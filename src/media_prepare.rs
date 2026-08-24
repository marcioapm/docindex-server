//! Provider-aware, provider-neutral media preparation.
//!
//! This module validates media from its bytes (not its filename), normalizes
//! image inputs, and turns PDFs into either native PDF ranges or raster page
//! images according to the selected model's [`PdfMode`]. It deliberately has
//! no indexer or store dependency: callers can persist the returned cache keys
//! and metadata using their existing pipeline.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::codecs::png::PngEncoder;
use image::{
    AnimationDecoder, DynamicImage, GenericImageView, ImageEncoder, ImageFormat, ImageReader,
};
use pdf_oxide::api::Pdf;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::rendering::RenderOptions;
use thiserror::Error;

use crate::embed::registry::{ModelInfo, PdfMode};
use crate::embed::{EmbedInput, MediaPart};
use crate::media::{MediaType, media_embedding_cache_key};

/// The largest decoded image accepted without downscaling.
pub const MAX_IMAGE_PIXELS: u64 = 16_000_000;

/// Settings that affect the prepared representation and therefore cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepareOptions {
    /// Consecutive pages combined into each native PDF input.
    pub pdf_pages_per_chunk: u8,
    /// Rasterization resolution for PDF pages supplied to raster providers.
    pub pdf_dpi: u16,
    /// Maximum decoded pixels for any resulting image.
    pub max_image_pixels: u64,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            pdf_pages_per_chunk: 1,
            pdf_dpi: 150,
            max_image_pixels: MAX_IMAGE_PIXELS,
        }
    }
}

/// Metadata a caller can attach to the chunk it stores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChunkMetadata {
    /// Zero-based output chunk index within the source file.
    pub chunk_index: usize,
    /// Zero-based, end-exclusive PDF page range, if this chunk came from a PDF.
    pub page_range: Option<(usize, usize)>,
    /// Whether an animated image was reduced to its first frame.
    pub truncated_animation: bool,
    /// MIME type of the source media for persistence.
    pub mime_type: String,
    /// Stable key over the exact prepared provider parts.
    pub cache_key: String,
}

/// One prepared embedding payload plus provider-neutral metadata.
#[derive(Clone)]
pub struct PreparedMediaChunk {
    pub input: EmbedInput,
    pub metadata: PreparedChunkMetadata,
}

/// Fully prepared media for a single path.
#[derive(Clone)]
pub struct PreparedMedia {
    pub path: PathBuf,
    pub media_type: MediaType,
    pub chunks: Vec<PreparedMediaChunk>,
}

/// A preparation failure. Paths are included for operational diagnosis; source
/// bytes and decoded content are never included in errors or logs.
#[derive(Debug, Error)]
pub enum MediaPrepareError {
    #[error("media preparation for {path}: unsupported media bytes")]
    Unsupported { path: PathBuf },
    #[error("media preparation for {path}: model {model:?} does not accept media")]
    MediaUnsupported { path: PathBuf, model: String },
    #[error("media preparation for {path}: model {model:?} does not accept PDFs")]
    PdfUnsupported { path: PathBuf, model: String },
    #[error("media preparation for {path}: invalid preparation option {name}={value}")]
    InvalidOption {
        path: PathBuf,
        name: &'static str,
        value: u64,
    },
    #[error("media preparation for {path}: image decode failed")]
    ImageDecode { path: PathBuf },
    #[error("media preparation for {path}: image encoding failed")]
    ImageEncode { path: PathBuf },
    #[error("media preparation for {path}: PDF parse failed")]
    PdfParse { path: PathBuf },
    #[error("media preparation for {path}: PDF contains no pages")]
    EmptyPdf { path: PathBuf },
    #[error("media preparation for {path}: PDF page range {start}..{end} could not be extracted")]
    PdfExtract {
        path: PathBuf,
        start: usize,
        end: usize,
    },
    #[error("media preparation for {path}: PDF page {page} could not be rasterized")]
    PdfRender { path: PathBuf, page: usize },
}

/// Prepares byte-identified image or PDF content for `model`.
///
/// A filename extension is intentionally not consulted. Unsupported byte
/// sequences and media disallowed by the model return a path-specific error.
pub fn prepare_media(
    path: impl AsRef<Path>,
    bytes: &[u8],
    model: &ModelInfo,
    options: PrepareOptions,
) -> Result<PreparedMedia, MediaPrepareError> {
    let path = path.as_ref().to_path_buf();
    validate_options(&path, options)?;
    if !model.media_capable {
        return Err(MediaPrepareError::MediaUnsupported {
            path,
            model: model.model.to_owned(),
        });
    }

    if is_pdf(bytes) {
        return prepare_pdf(path, bytes, model, options);
    }
    prepare_image(path, bytes, options)
}

fn validate_options(path: &Path, options: PrepareOptions) -> Result<(), MediaPrepareError> {
    if options.pdf_pages_per_chunk == 0 {
        return Err(MediaPrepareError::InvalidOption {
            path: path.to_path_buf(),
            name: "pdf_pages_per_chunk",
            value: 0,
        });
    }
    if options.pdf_dpi == 0 {
        return Err(MediaPrepareError::InvalidOption {
            path: path.to_path_buf(),
            name: "pdf_dpi",
            value: 0,
        });
    }
    if options.max_image_pixels == 0 {
        return Err(MediaPrepareError::InvalidOption {
            path: path.to_path_buf(),
            name: "max_image_pixels",
            value: 0,
        });
    }
    Ok(())
}

fn prepare_image(
    path: PathBuf,
    bytes: &[u8],
    options: PrepareOptions,
) -> Result<PreparedMedia, MediaPrepareError> {
    let format = image::guess_format(bytes)
        .map_err(|_| MediaPrepareError::Unsupported { path: path.clone() })?;
    let (image, truncated_animation) = decode_image(bytes, format, &path)?;
    let original_dimensions = image.dimensions();
    let resized = limit_pixels(image, options.max_image_pixels);
    let must_encode_png = matches!(format, ImageFormat::Gif | ImageFormat::WebP)
        || resized.dimensions() != original_dimensions;

    let (prepared, payload_mime_type) = if must_encode_png {
        (encode_png(&resized, &path)?, "image/png")
    } else {
        (bytes.to_vec(), mime_for_image(format))
    };
    let input = media_input(payload_mime_type, prepared);
    let metadata = metadata(
        MediaType::Image,
        0,
        None,
        payload_mime_type,
        &input,
        truncated_animation,
    );
    Ok(PreparedMedia {
        path,
        media_type: MediaType::Image,
        chunks: vec![PreparedMediaChunk { input, metadata }],
    })
}

fn decode_image(
    bytes: &[u8],
    format: ImageFormat,
    path: &Path,
) -> Result<(DynamicImage, bool), MediaPrepareError> {
    match format {
        ImageFormat::Gif => {
            let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
                .map_err(|_| MediaPrepareError::ImageDecode { path: path.into() })?;
            let frames = decoder
                .into_frames()
                .collect_frames()
                .map_err(|_| MediaPrepareError::ImageDecode { path: path.into() })?;
            let animated = frames.len() > 1;
            let first = frames
                .into_iter()
                .next()
                .ok_or_else(|| MediaPrepareError::ImageDecode { path: path.into() })?;
            Ok((DynamicImage::ImageRgba8(first.into_buffer()), animated))
        }
        ImageFormat::WebP => {
            let decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
                .map_err(|_| MediaPrepareError::ImageDecode { path: path.into() })?;
            if !decoder.has_animation() {
                return DynamicImage::from_decoder(decoder)
                    .map(|image| (image, false))
                    .map_err(|_| MediaPrepareError::ImageDecode { path: path.into() });
            }
            let first = decoder
                .into_frames()
                .next()
                .transpose()
                .map_err(|_| MediaPrepareError::ImageDecode { path: path.into() })?
                .ok_or_else(|| MediaPrepareError::ImageDecode { path: path.into() })?;
            Ok((DynamicImage::ImageRgba8(first.into_buffer()), true))
        }
        _ => ImageReader::with_format(Cursor::new(bytes), format)
            .decode()
            .map(|image| (image, false))
            .map_err(|_| MediaPrepareError::ImageDecode { path: path.into() }),
    }
}

fn prepare_pdf(
    path: PathBuf,
    bytes: &[u8],
    model: &ModelInfo,
    options: PrepareOptions,
) -> Result<PreparedMedia, MediaPrepareError> {
    match model.pdf_mode {
        PdfMode::None => Err(MediaPrepareError::PdfUnsupported {
            path,
            model: model.model.to_owned(),
        }),
        PdfMode::Native => prepare_native_pdf(path, bytes, options),
        PdfMode::Raster => prepare_raster_pdf(path, bytes, options),
    }
}

fn prepare_native_pdf(
    path: PathBuf,
    bytes: &[u8],
    options: PrepareOptions,
) -> Result<PreparedMedia, MediaPrepareError> {
    let mut editor = DocumentEditor::from_bytes(bytes.to_vec())
        .map_err(|_| MediaPrepareError::PdfParse { path: path.clone() })?;
    let page_count = editor.current_page_count();
    if page_count == 0 {
        return Err(MediaPrepareError::EmptyPdf { path });
    }
    let ranges = page_ranges(page_count, options.pdf_pages_per_chunk as usize);
    let chunks = ranges
        .into_iter()
        .enumerate()
        .map(|(chunk_index, (start, end))| {
            let pages: Vec<_> = (start..end).collect();
            let prepared = editor.extract_pages_to_bytes(&pages).map_err(|_| {
                MediaPrepareError::PdfExtract {
                    path: path.clone(),
                    start,
                    end,
                }
            })?;
            let mime_type = "application/pdf";
            let input = media_input(mime_type, prepared);
            Ok(PreparedMediaChunk {
                metadata: metadata(
                    MediaType::Pdf,
                    chunk_index,
                    Some((start, end)),
                    mime_type,
                    &input,
                    false,
                ),
                input,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedMedia {
        path,
        media_type: MediaType::Pdf,
        chunks,
    })
}

fn prepare_raster_pdf(
    path: PathBuf,
    bytes: &[u8],
    options: PrepareOptions,
) -> Result<PreparedMedia, MediaPrepareError> {
    let mut pdf = Pdf::from_bytes(bytes.to_vec())
        .map_err(|_| MediaPrepareError::PdfParse { path: path.clone() })?;
    let page_count = pdf
        .page_count()
        .map_err(|_| MediaPrepareError::PdfParse { path: path.clone() })?;
    if page_count == 0 {
        return Err(MediaPrepareError::EmptyPdf { path });
    }
    let render_options = RenderOptions::with_dpi(u32::from(options.pdf_dpi));
    let chunks = page_ranges(page_count, options.pdf_pages_per_chunk as usize)
        .into_iter()
        .enumerate()
        .map(|(chunk_index, (start, end))| {
            let parts = (start..end)
                .map(|page| {
                    let rendered = pdf
                        .render_page_with_options(page, &render_options)
                        .map_err(|_| MediaPrepareError::PdfRender {
                            path: path.clone(),
                            page,
                        })?;
                    let image =
                        ImageReader::with_format(Cursor::new(rendered.data), ImageFormat::Png)
                            .decode()
                            .map_err(|_| MediaPrepareError::PdfRender {
                                path: path.clone(),
                                page,
                            })?;
                    Ok(MediaPart {
                        mime_type: "image/png".to_owned(),
                        bytes: encode_png(&limit_pixels(image, options.max_image_pixels), &path)?,
                    })
                })
                .collect::<Result<Vec<_>, MediaPrepareError>>()?;
            let input = EmbedInput::Media(parts);
            Ok(PreparedMediaChunk {
                metadata: metadata(
                    MediaType::Pdf,
                    chunk_index,
                    Some((start, end)),
                    "application/pdf",
                    &input,
                    false,
                ),
                input,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedMedia {
        path,
        media_type: MediaType::Pdf,
        chunks,
    })
}

fn media_input(mime_type: &str, bytes: Vec<u8>) -> EmbedInput {
    EmbedInput::Media(vec![MediaPart {
        mime_type: mime_type.to_owned(),
        bytes,
    }])
}

fn metadata(
    media_type: MediaType,
    chunk_index: usize,
    page_range: Option<(usize, usize)>,
    mime_type: &str,
    input: &EmbedInput,
    truncated_animation: bool,
) -> PreparedChunkMetadata {
    PreparedChunkMetadata {
        chunk_index,
        page_range,
        mime_type: mime_type.to_owned(),
        cache_key: cache_key_for_input(media_type, input),
        truncated_animation,
    }
}

/// Hashes every provider part with length-delimited MIME and payload fields.
/// This prevents ambiguity between differently partitioned multipart inputs.
fn cache_key_for_input(media_type: MediaType, input: &EmbedInput) -> String {
    let EmbedInput::Media(parts) = input else {
        unreachable!("media preparation only creates media inputs");
    };
    let mut framed = Vec::new();
    framed.extend_from_slice(&(parts.len() as u64).to_le_bytes());
    for part in parts {
        framed.extend_from_slice(&(part.mime_type.len() as u64).to_le_bytes());
        framed.extend_from_slice(part.mime_type.as_bytes());
        framed.extend_from_slice(&(part.bytes.len() as u64).to_le_bytes());
        framed.extend_from_slice(&part.bytes);
    }
    media_embedding_cache_key(media_type, "multipart", &framed)
}

fn encode_png(image: &DynamicImage, path: &Path) -> Result<Vec<u8>, MediaPrepareError> {
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut output = Vec::new();
    PngEncoder::new(&mut output)
        .write_image(rgba.as_raw(), width, height, image::ColorType::Rgba8.into())
        .map_err(|_| MediaPrepareError::ImageEncode { path: path.into() })?;
    Ok(output)
}

fn limit_pixels(image: DynamicImage, cap: u64) -> DynamicImage {
    let (width, height) = image.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= cap {
        return image;
    }
    let scale = (cap as f64 / pixels as f64).sqrt();
    let resized_width = ((f64::from(width) * scale).floor() as u32).max(1);
    let resized_height = ((f64::from(height) * scale).floor() as u32).max(1);
    image.resize_exact(
        resized_width,
        resized_height,
        image::imageops::FilterType::Lanczos3,
    )
}

fn page_ranges(page_count: usize, pages_per_chunk: usize) -> Vec<(usize, usize)> {
    (0..page_count)
        .step_by(pages_per_chunk)
        .map(|start| (start, (start + pages_per_chunk).min(page_count)))
        .collect()
}

fn is_pdf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF-")
}

fn mime_for_image(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Gif => "image/gif",
        ImageFormat::WebP => "image/webp",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::registry::{EmbedProvider, PdfMode};

    fn media_model(pdf_mode: PdfMode) -> ModelInfo {
        ModelInfo {
            provider: EmbedProvider::Gemini,
            model: "test-media",
            native_dim: 1,
            allowed_dims: &[],
            doc_task: "document",
            query_task: "query",
            media_capable: true,
            pdf_mode,
        }
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(width, height, |x, y| {
            image::Rgba([(x % 255) as u8, (y % 255) as u8, 0, 255])
        }));
        encode_png(&image, Path::new("generated.png")).unwrap()
    }

    fn animated_gif() -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = image::codecs::gif::GifEncoder::new(&mut bytes);
            for color in [image::Rgba([255, 0, 0, 255]), image::Rgba([0, 0, 255, 255])] {
                let frame = image::Frame::new(image::RgbaImage::from_pixel(2, 2, color));
                encoder.encode_frame(frame).unwrap();
            }
        }
        bytes
    }

    #[test]
    fn detects_png_bytes_without_consulting_extension() {
        let prepared = prepare_media(
            "misleading.pdf",
            &png(2, 3),
            &media_model(PdfMode::Native),
            PrepareOptions::default(),
        )
        .unwrap();
        assert_eq!(prepared.media_type, MediaType::Image);
        assert_eq!(prepared.chunks[0].metadata.mime_type, "image/png");
    }

    #[test]
    fn animated_gif_uses_first_frame_png() {
        let prepared = prepare_media(
            "animation.dat",
            &animated_gif(),
            &media_model(PdfMode::Native),
            PrepareOptions::default(),
        )
        .unwrap();
        assert!(prepared.chunks[0].metadata.truncated_animation);
        let EmbedInput::Media(parts) = &prepared.chunks[0].input else {
            panic!("expected media input");
        };
        let pixel = image::load_from_memory(&parts[0].bytes).unwrap().to_rgba8();
        assert_eq!(pixel.get_pixel(0, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn oversized_generated_image_is_resized_under_cap() {
        let prepared = prepare_media(
            "large.bin",
            &png(500, 500),
            &media_model(PdfMode::Native),
            PrepareOptions {
                max_image_pixels: 10_000,
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        let EmbedInput::Media(parts) = &prepared.chunks[0].input else {
            panic!("expected media input");
        };
        let decoded = image::load_from_memory(&parts[0].bytes).unwrap();
        assert!(u64::from(decoded.width()) * u64::from(decoded.height()) <= 10_000);
    }

    #[test]
    fn original_png_passes_through_exact_bytes() {
        let bytes = png(10, 10);
        let prepared = prepare_media(
            "image.png",
            &bytes,
            &media_model(PdfMode::Native),
            PrepareOptions::default(),
        )
        .unwrap();
        let EmbedInput::Media(parts) = &prepared.chunks[0].input else {
            panic!("expected media input");
        };
        assert_eq!(parts[0].mime_type, "image/png");
        assert_eq!(parts[0].bytes, bytes);
        assert!(!prepared.chunks[0].metadata.truncated_animation);
    }

    #[test]
    fn cache_key_uses_exact_framed_provider_parts() {
        let one_part = EmbedInput::Media(vec![MediaPart {
            mime_type: "image/png".to_owned(),
            bytes: b"abc".to_vec(),
        }]);
        let split_parts = EmbedInput::Media(vec![
            MediaPart {
                mime_type: "image/png".to_owned(),
                bytes: b"a".to_vec(),
            },
            MediaPart {
                mime_type: "image/png".to_owned(),
                bytes: b"bc".to_vec(),
            },
        ]);
        assert_ne!(
            cache_key_for_input(MediaType::Pdf, &one_part),
            cache_key_for_input(MediaType::Pdf, &split_parts)
        );
    }

    #[test]
    fn cache_key_does_not_duplicate_ineffective_profile_settings() {
        let bytes = png(10, 10);
        let model = media_model(PdfMode::Native);
        let default = prepare_media("image", &bytes, &model, PrepareOptions::default()).unwrap();
        let changed = prepare_media(
            "image",
            &bytes,
            &model,
            PrepareOptions {
                pdf_dpi: 72,
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            default.chunks[0].metadata.cache_key,
            changed.chunks[0].metadata.cache_key
        );
    }

    #[test]
    fn page_ranges_are_zero_based_and_end_exclusive() {
        assert_eq!(page_ranges(5, 2), vec![(0, 2), (2, 4), (4, 5)]);
        assert!(page_ranges(0, 1).is_empty());
    }

    #[test]
    fn pdf_bytes_are_rejected_for_pdf_none_with_path_only_error() {
        let error = match prepare_media(
            "private/paper.pdf",
            b"%PDF-1.7\nnot-a-real-document",
            &media_model(PdfMode::None),
            PrepareOptions::default(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("PDFs must be rejected for PdfMode::None"),
        };
        let display = error.to_string();
        assert!(display.contains("private/paper.pdf"));
        assert!(!display.contains("not-a-real-document"));
    }
}
