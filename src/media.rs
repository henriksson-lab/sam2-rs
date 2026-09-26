//! Optional image and video decoding for predictor inputs.

use std::{fs, path::Path};

use crate::{Error, Result};

/// One owned, tightly packed interleaved RGB8 frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

impl DecodedFrame {
    pub fn new(pixels: Vec<u8>, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 || pixels.len() != width * height * 3 {
            return Err(media_error(format!(
                "expected {} RGB bytes for {width}x{height}, got {}",
                width * height * 3,
                pixels.len()
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn as_input(&self) -> (&[u8], usize, usize) {
        (&self.pixels, self.width, self.height)
    }

    pub fn into_parts(self) -> (Vec<u8>, usize, usize) {
        (self.pixels, self.width, self.height)
    }
}

/// A decoded frame sequence plus its original display dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedVideo {
    frames: Vec<DecodedFrame>,
    original_width: usize,
    original_height: usize,
}

impl DecodedVideo {
    fn new(
        frames: Vec<DecodedFrame>,
        original_width: usize,
        original_height: usize,
    ) -> Result<Self> {
        if frames.is_empty() {
            return Err(media_error("video contains no decodable frames"));
        }
        if frames
            .iter()
            .any(|frame| frame.width != original_width || frame.height != original_height)
        {
            return Err(media_error(
                "all video frames must have the same dimensions",
            ));
        }
        Ok(Self {
            frames,
            original_width,
            original_height,
        })
    }

    pub fn frames(&self) -> &[DecodedFrame] {
        &self.frames
    }

    pub fn into_frames(self) -> Vec<DecodedFrame> {
        self.frames
    }

    pub const fn original_width(&self) -> usize {
        self.original_width
    }

    pub const fn original_height(&self) -> usize {
        self.original_height
    }

    pub fn inputs(&self) -> Vec<(&[u8], usize, usize)> {
        self.frames.iter().map(DecodedFrame::as_input).collect()
    }
}

/// Decodes one image and converts it to tightly packed RGB8.
pub fn decode_image(path: impl AsRef<Path>) -> Result<DecodedFrame> {
    let path = path.as_ref();
    let image = image::ImageReader::open(path)
        .map_err(|error| media_error(format!("could not open {}: {error}", path.display())))?
        .with_guessed_format()
        .map_err(|error| media_error(format!("could not inspect {}: {error}", path.display())))?
        .decode()
        .map_err(|error| media_error(format!("could not decode {}: {error}", path.display())))?
        .to_rgb8();
    let width = image.width() as usize;
    let height = image.height() as usize;
    DecodedFrame::new(image.into_raw(), width, height)
}

/// Loads `<integer>.jpg`/`.jpeg` frames in numeric order, matching upstream.
pub fn load_jpeg_directory(path: impl AsRef<Path>) -> Result<DecodedVideo> {
    let path = path.as_ref();
    let entries = fs::read_dir(path)
        .map_err(|error| media_error(format!("could not read {}: {error}", path.display())))?;
    let mut frames = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| media_error(error.to_string()))?;
        let entry_path = entry.path();
        let Some(extension) = entry_path.extension().and_then(|value| value.to_str()) else {
            continue;
        };
        if !extension.eq_ignore_ascii_case("jpg") && !extension.eq_ignore_ascii_case("jpeg") {
            continue;
        }
        let stem = entry_path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| media_error(format!("invalid frame name {}", entry_path.display())))?;
        let index = stem.parse::<u64>().map_err(|_| {
            media_error(format!(
                "JPEG frame name must have an integer stem: {}",
                entry_path.display()
            ))
        })?;
        frames.push((index, entry_path));
    }
    frames.sort_by_key(|(index, _)| *index);
    if frames.is_empty() {
        return Err(media_error(format!(
            "no JPEG frames found in {}",
            path.display()
        )));
    }
    if frames.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(media_error("JPEG frame indices must be unique"));
    }
    let decoded = frames
        .into_iter()
        .map(|(_, path)| decode_image(path))
        .collect::<Result<Vec<_>>>()?;
    let width = decoded[0].width;
    let height = decoded[0].height;
    DecodedVideo::new(decoded, width, height)
}

/// Loads either an encoded video file or a JPEG frame directory.
#[cfg(feature = "video-decoding")]
pub fn load_video(path: impl AsRef<Path>) -> Result<DecodedVideo> {
    let path = path.as_ref();
    if path.is_dir() {
        load_jpeg_directory(path)
    } else {
        decode_video(path)
    }
}

/// Decodes an encoded video through FFmpeg into tightly packed RGB8 frames.
#[cfg(feature = "video-decoding")]
pub fn decode_video(path: impl AsRef<Path>) -> Result<DecodedVideo> {
    decode_video_impl(path.as_ref(), None)
}

/// Decodes at most the first `max_frames` frames of an encoded video.
///
/// This is useful for bounded previews and tests without allocating the whole
/// sequence. A zero limit is rejected instead of returning an invalid empty
/// video.
#[cfg(feature = "video-decoding")]
pub fn decode_video_prefix(path: impl AsRef<Path>, max_frames: usize) -> Result<DecodedVideo> {
    if max_frames == 0 {
        return Err(media_error("video frame limit must be greater than zero"));
    }
    decode_video_impl(path.as_ref(), Some(max_frames))
}

#[cfg(feature = "video-decoding")]
fn decode_video_impl(path: &Path, max_frames: Option<usize>) -> Result<DecodedVideo> {
    use ffmpeg::{format::Pixel, media::Type, software::scaling};
    use ffmpeg_next as ffmpeg;

    ffmpeg::init().map_err(ffmpeg_error)?;
    let mut input = ffmpeg::format::input(path).map_err(ffmpeg_error)?;
    let stream = input
        .streams()
        .best(Type::Video)
        .ok_or_else(|| media_error(format!("{} has no video stream", path.display())))?;
    let stream_index = stream.index();
    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .map_err(ffmpeg_error)?;
    let mut decoder = context.decoder().video().map_err(ffmpeg_error)?;
    let width = decoder.width() as usize;
    let height = decoder.height() as usize;
    if width == 0 || height == 0 {
        return Err(media_error("video reports zero-sized frames"));
    }
    let mut scaler = scaling::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        scaling::flag::Flags::BILINEAR,
    )
    .map_err(ffmpeg_error)?;
    let mut frames = Vec::new();

    for (packet_stream, packet) in input.packets() {
        if packet_stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet).map_err(ffmpeg_error)?;
        receive_frames(&mut decoder, &mut scaler, width, height, &mut frames)?;
        if max_frames.is_some_and(|limit| frames.len() >= limit) {
            frames.truncate(max_frames.unwrap());
            return DecodedVideo::new(frames, width, height);
        }
    }
    decoder.send_eof().map_err(ffmpeg_error)?;
    receive_frames(&mut decoder, &mut scaler, width, height, &mut frames)?;
    DecodedVideo::new(frames, width, height)
}

#[cfg(feature = "video-decoding")]
fn receive_frames(
    decoder: &mut ffmpeg_next::decoder::Video,
    scaler: &mut ffmpeg_next::software::scaling::Context,
    width: usize,
    height: usize,
    output: &mut Vec<DecodedFrame>,
) -> Result<()> {
    use ffmpeg_next::{Error as FfmpegError, util::frame::video::Video};

    loop {
        let mut decoded = Video::empty();
        match decoder.receive_frame(&mut decoded) {
            Ok(()) => {
                let mut rgb = Video::empty();
                scaler.run(&decoded, &mut rgb).map_err(ffmpeg_error)?;
                let row_bytes = width * 3;
                let stride = rgb.stride(0);
                let source = rgb.data(0);
                let mut pixels = vec![0u8; row_bytes * height];
                for row in 0..height {
                    pixels[row * row_bytes..(row + 1) * row_bytes]
                        .copy_from_slice(&source[row * stride..row * stride + row_bytes]);
                }
                output.push(DecodedFrame::new(pixels, width, height)?);
            }
            Err(FfmpegError::Other { errno }) if errno == ffmpeg_next::error::EAGAIN => break,
            Err(FfmpegError::Eof) => break,
            Err(error) => return Err(ffmpeg_error(error)),
        }
    }
    Ok(())
}

fn media_error(message: impl Into<String>) -> Error {
    Error::Integration {
        operation: "media decoding failed",
        message: message.into(),
    }
}

#[cfg(feature = "video-decoding")]
fn ffmpeg_error(error: ffmpeg_next::Error) -> Error {
    media_error(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn jpeg_directories_use_numeric_order() {
        let temp = tempfile::tempdir().unwrap();
        for (name, value) in [("10.jpg", 10u8), ("2.JPEG", 2u8), ("1.jpg", 1u8)] {
            ImageBuffer::from_pixel(3, 2, Rgb([value, value + 1, value + 2]))
                .save(temp.path().join(name))
                .unwrap();
        }
        let video = load_jpeg_directory(temp.path()).unwrap();
        assert_eq!(video.frames().len(), 3);
        assert_eq!((video.original_width(), video.original_height()), (3, 2));
        // JPEG is lossy, but these flat fields retain their ordering.
        assert!(video.frames()[0].pixels()[0] < video.frames()[1].pixels()[0]);
        assert!(video.frames()[1].pixels()[0] < video.frames()[2].pixels()[0]);
    }

    #[test]
    fn mismatched_frame_dimensions_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        ImageBuffer::from_pixel(3, 2, Rgb([1u8, 2, 3]))
            .save(temp.path().join("0.jpg"))
            .unwrap();
        ImageBuffer::from_pixel(4, 2, Rgb([1u8, 2, 3]))
            .save(temp.path().join("1.jpg"))
            .unwrap();
        assert!(load_jpeg_directory(temp.path()).is_err());
    }

    #[cfg(feature = "video-decoding")]
    #[test]
    fn decodes_the_vendored_sam2_mp4() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("vendor/segment-anything-2/demo/data/gallery/05_default_juggle.mp4");
        if !path.exists() {
            return;
        }
        let video = decode_video_prefix(path, 2).unwrap();
        assert_eq!(video.frames().len(), 2);
        assert_eq!(
            (video.original_width(), video.original_height()),
            (1280, 720)
        );
        assert!(
            video
                .frames()
                .iter()
                .all(|frame| frame.pixels().len() == frame.width() * frame.height() * 3)
        );
    }
}
