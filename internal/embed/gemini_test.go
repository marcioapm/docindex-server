package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func okResponse(vec []float32) *http.Response {
	body := map[string]any{
		"embedding": map[string]any{"values": vec},
	}
	b, _ := json.Marshal(body)
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader(b)),
		Header:     make(http.Header),
	}
}

func errResponse(status int, msg string) *http.Response {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{"code": status, "message": msg, "status": "ERROR"},
	})
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     make(http.Header),
	}
}

func TestGemini_EmbedDocuments_URLHeadersBody(t *testing.T) {
	var gotURL, gotAPIKey, gotCT string
	var gotBody map[string]any
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotURL = req.URL.String()
		gotAPIKey = req.Header.Get("x-goog-api-key")
		gotCT = req.Header.Get("Content-Type")
		raw, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(raw, &gotBody)
		return okResponse([]float32{0.1, 0.2, 0.3}), nil
	})

	g := &Gemini{
		APIKey:  "test-key",
		Model:   "gemini-embedding-001",
		Dim:     768,
		BaseURL: "https://example.test",
		Client:  &http.Client{Transport: rt},
	}
	got, err := g.EmbedDocuments(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(got) != 1 || len(got[0]) != 3 {
		t.Fatalf("bad shape: %+v", got)
	}
	if !strings.Contains(gotURL, "/v1beta/models/gemini-embedding-001:embedContent") {
		t.Errorf("URL = %q", gotURL)
	}
	if gotAPIKey != "test-key" {
		t.Errorf("api key = %q", gotAPIKey)
	}
	if gotCT != "application/json" {
		t.Errorf("content-type = %q", gotCT)
	}
	if gotBody["taskType"] != "RETRIEVAL_DOCUMENT" {
		t.Errorf("taskType = %v", gotBody["taskType"])
	}
	if int(gotBody["outputDimensionality"].(float64)) != 768 {
		t.Errorf("outputDimensionality = %v", gotBody["outputDimensionality"])
	}
}

func TestGemini_EmbedQuery_TaskType(t *testing.T) {
	var got map[string]any
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		raw, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(raw, &got)
		return okResponse([]float32{1}), nil
	})
	g := &Gemini{APIKey: "k", Model: "m", Dim: 1, BaseURL: "http://x", Client: &http.Client{Transport: rt}}
	if _, err := g.EmbedQuery(context.Background(), "q"); err != nil {
		t.Fatal(err)
	}
	if got["taskType"] != "RETRIEVAL_QUERY" {
		t.Errorf("taskType = %v", got["taskType"])
	}
}

func TestGemini_RetryOn429(t *testing.T) {
	var calls atomic.Int32
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		n := calls.Add(1)
		if n < 3 {
			return errResponse(429, "rate limited"), nil
		}
		return okResponse([]float32{0.5}), nil
	})
	g := &Gemini{
		APIKey: "k", Model: "m", Dim: 1, BaseURL: "http://x",
		Client: &http.Client{Transport: rt}, MaxRetries: 3, BaseDelay: time.Millisecond,
	}
	v, err := g.EmbedQuery(context.Background(), "q")
	if err != nil {
		t.Fatalf("expected success after retry: %v", err)
	}
	if len(v) != 1 {
		t.Fatalf("bad vec: %+v", v)
	}
	if calls.Load() != 3 {
		t.Errorf("calls = %d, want 3", calls.Load())
	}
}

func TestGemini_RetryOn500(t *testing.T) {
	var calls atomic.Int32
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls.Add(1)
		if calls.Load() == 1 {
			return errResponse(503, "down"), nil
		}
		return okResponse([]float32{1}), nil
	})
	g := &Gemini{APIKey: "k", Model: "m", Dim: 1, BaseURL: "http://x",
		Client: &http.Client{Transport: rt}, MaxRetries: 2, BaseDelay: time.Millisecond}
	if _, err := g.EmbedQuery(context.Background(), "q"); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
}

func TestGemini_FailFastOn4xx(t *testing.T) {
	var calls atomic.Int32
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls.Add(1)
		return errResponse(400, "bad"), nil
	})
	g := &Gemini{APIKey: "k", Model: "m", Dim: 1, BaseURL: "http://x",
		Client: &http.Client{Transport: rt}, MaxRetries: 5, BaseDelay: time.Millisecond}
	_, err := g.EmbedQuery(context.Background(), "q")
	if err == nil {
		t.Fatalf("expected error")
	}
	if calls.Load() != 1 {
		t.Errorf("should not retry 4xx; calls = %d", calls.Load())
	}
}

func TestGemini_ValidationErrors(t *testing.T) {
	g := &Gemini{Model: "m", Dim: 1, Client: &http.Client{}}
	if _, err := g.EmbedQuery(context.Background(), "q"); err == nil {
		t.Errorf("expected error for empty APIKey")
	}
	g = &Gemini{APIKey: "k", Dim: 1, Client: &http.Client{}}
	if _, err := g.EmbedQuery(context.Background(), "q"); err == nil {
		t.Errorf("expected error for empty Model")
	}
	g = &Gemini{APIKey: "k", Model: "m", Dim: 0, Client: &http.Client{}}
	if _, err := g.EmbedQuery(context.Background(), "q"); err == nil {
		t.Errorf("expected error for Dim <= 0")
	}
}

func TestFake_Deterministic(t *testing.T) {
	f := NewFake(8)
	a, _ := f.EmbedQuery(context.Background(), "hello")
	b, _ := f.EmbedQuery(context.Background(), "hello")
	if len(a) != 8 || len(b) != 8 {
		t.Fatalf("wrong dim: %d %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("not deterministic at %d", i)
		}
	}
}

func TestFake_DocVsQueryDiffer(t *testing.T) {
	f := NewFake(16)
	q, _ := f.EmbedQuery(context.Background(), "x")
	docs, _ := f.EmbedDocuments(context.Background(), []string{"x"})
	same := true
	for i := range q {
		if q[i] != docs[0][i] {
			same = false
			break
		}
	}
	if same {
		t.Errorf("doc and query vectors should differ for same text")
	}
}

func TestFake_Normalized(t *testing.T) {
	f := NewFake(32)
	v, _ := f.EmbedQuery(context.Background(), "abc")
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("not normalized: ||v||^2 = %v", sum)
	}
}
