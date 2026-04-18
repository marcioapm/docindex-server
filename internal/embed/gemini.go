// Package embed provides an Embedder interface and a Gemini implementation.
//
// Gemini uses task-asymmetric embeddings: documents are embedded with
// task type RETRIEVAL_DOCUMENT and queries with RETRIEVAL_QUERY. Getting
// this wrong silently degrades ranking quality — the vectors still parse,
// they are just miscalibrated. The output is Matryoshka-truncated to 768
// dimensions at the API boundary so we never round-trip the full 3072.
package embed

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

const (
	// TaskRetrievalDocument is used when embedding chunks for indexing.
	TaskRetrievalDocument = "RETRIEVAL_DOCUMENT"
	// TaskRetrievalQuery is used when embedding a user query for search.
	TaskRetrievalQuery = "RETRIEVAL_QUERY"
)

// Embedder produces float32 vectors for document chunks or user queries.
type Embedder interface {
	EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// Gemini is an Embedder backed by Google's Generative Language REST API.
type Gemini struct {
	APIKey  string
	Model   string // e.g. "gemini-embedding-001"
	Dim     int    // Matryoshka output dimensionality
	BaseURL string // override for tests; defaults to https://generativelanguage.googleapis.com
	Client  *http.Client
	// MaxRetries on 429/5xx. Zero means no retries.
	MaxRetries int
	// BaseDelay is the first retry backoff; subsequent retries double it.
	BaseDelay time.Duration
}

// NewGemini constructs a Gemini embedder with sensible defaults.
func NewGemini(apiKey, model string, dim int, timeout time.Duration) *Gemini {
	return &Gemini{
		APIKey:     apiKey,
		Model:      model,
		Dim:        dim,
		BaseURL:    "https://generativelanguage.googleapis.com",
		Client:     &http.Client{Timeout: timeout},
		MaxRetries: 3,
		BaseDelay:  200 * time.Millisecond,
	}
}

// EmbedDocuments embeds with task type RETRIEVAL_DOCUMENT.
func (g *Gemini) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := g.embed(ctx, t, TaskRetrievalDocument)
		if err != nil {
			return nil, fmt.Errorf("embed doc[%d]: %w", i, err)
		}
		out[i] = v
	}
	return out, nil
}

// EmbedQuery embeds with task type RETRIEVAL_QUERY.
func (g *Gemini) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return g.embed(ctx, text, TaskRetrievalQuery)
}

// Request/response shapes mirror
// POST {base}/v1beta/models/{model}:embedContent?key={APIKey}
type embedRequest struct {
	Content              embedContent `json:"content"`
	TaskType             string       `json:"taskType"`
	OutputDimensionality int          `json:"outputDimensionality"`
}

type embedContent struct {
	Parts []embedPart `json:"parts"`
}

type embedPart struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

type apiError struct {
	Err struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

func (g *Gemini) embed(ctx context.Context, text, taskType string) ([]float32, error) {
	if g.APIKey == "" {
		return nil, errors.New("gemini: APIKey is empty")
	}
	if g.Model == "" {
		return nil, errors.New("gemini: Model is empty")
	}
	if g.Dim <= 0 {
		return nil, errors.New("gemini: Dim must be > 0")
	}

	body, err := json.Marshal(embedRequest{
		Content:              embedContent{Parts: []embedPart{{Text: text}}},
		TaskType:             taskType,
		OutputDimensionality: g.Dim,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:embedContent", g.BaseURL, g.Model)

	var vec []float32
	var lastErr error
	delay := g.BaseDelay
	attempts := g.MaxRetries + 1
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
			delay *= 2
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("new request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-goog-api-key", g.APIKey)

		resp, err := g.Client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		vec, lastErr = readResponse(resp)
		if lastErr == nil {
			return vec, nil
		}
		// Only retry on retryable status; non-retryable fail fast.
		if !isRetryable(resp.StatusCode) {
			return nil, lastErr
		}
	}
	return nil, fmt.Errorf("gemini: exhausted retries: %w", lastErr)
}

func readResponse(resp *http.Response) ([]float32, error) {
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var ae apiError
		_ = json.Unmarshal(raw, &ae)
		msg := ae.Err.Message
		if msg == "" {
			msg = string(raw)
		}
		return nil, fmt.Errorf("gemini: status %d: %s", resp.StatusCode, msg)
	}
	var out embedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if len(out.Embedding.Values) == 0 {
		return nil, errors.New("gemini: empty embedding values")
	}
	return out.Embedding.Values, nil
}

func isRetryable(status int) bool {
	return status == http.StatusTooManyRequests || (status >= 500 && status <= 599)
}

// Fake is a deterministic, offline Embedder for tests. Vectors are derived
// from sha256(text) and then L2-normalized so cosine distance behaves.
// The output dimension is configurable.
type Fake struct {
	Dim int
}

// NewFake returns a Fake embedder with the given dimension.
func NewFake(dim int) *Fake {
	if dim <= 0 {
		dim = 768
	}
	return &Fake{Dim: dim}
}

// EmbedDocuments implements Embedder.
func (f *Fake) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = f.vector(t + "|doc")
	}
	return out, nil
}

// EmbedQuery implements Embedder.
func (f *Fake) EmbedQuery(_ context.Context, text string) ([]float32, error) {
	return f.vector(text + "|query"), nil
}

func (f *Fake) vector(seed string) []float32 {
	out := make([]float32, f.Dim)
	// Expand sha256 by hashing (seed, counter) until we have enough bytes.
	for i := 0; i < f.Dim; i += 8 {
		h := sha256.Sum256(fmt.Appendf(nil, "%s:%d", seed, i))
		for j := 0; j < 8 && i+j < f.Dim; j++ {
			// Two bytes per float, map to [-1, 1].
			u := uint16(h[2*j])<<8 | uint16(h[2*j+1])
			out[i+j] = float32(int16(u)) / 32768.0
		}
	}
	return l2Normalize(out)
}

func l2Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	norm := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= norm
	}
	return v
}
