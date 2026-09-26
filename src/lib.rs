//! Native, Candle-backed translation of Meta's Segment Anything Model 2.
//!
//! The public API deliberately keeps checkpoint loading separate from model
//! construction so callers do not need to care whether a checkpoint originated
//! as a PyTorch archive or a converted safetensors file.

pub mod checkpoint;
pub mod config;
mod cpu_allocator;
pub mod error;
#[cfg(feature = "huggingface")]
pub mod huggingface;
#[cfg(feature = "image-decoding")]
pub mod media;
pub mod modeling;
pub mod ops;
pub mod predictor;
pub mod transforms;
pub mod video_state;

pub use checkpoint::{CheckpointFormat, CheckpointIndex, CheckpointManifest, TensorInfo};
pub use config::{HieraConfig, Sam2Config, Sam2Variant};
pub use error::{Error, Result};
#[cfg(feature = "huggingface")]
pub use huggingface::{HuggingFaceCheckpoint, download_checkpoint};
#[cfg(feature = "image-decoding")]
pub use media::{DecodedFrame, DecodedVideo, decode_image, load_jpeg_directory};
#[cfg(feature = "video-decoding")]
pub use media::{decode_video, decode_video_prefix, load_video};
pub use modeling::{
    Attention, BackboneOutput, FpnNeck, Hiera, HieraProfile, ImageEncoder, ImageEncoderProfile,
    MaskDecoder, MaskDecoderConfig, MaskDecoderOutput, MaskDecoderProfile, MemoryAttention,
    MemoryEncoder, MemoryEncoderOutput, PromptEncoder, PromptEncoderOutput, RopeAttention,
    Sam2Base, Sam2FrameOutput, TwoWayTransformer, TwoWayTransformerProfile,
};
pub use ops::MaskRle;
pub use predictor::{
    AutomaticMask, AutomaticMaskConfig, AutomaticMaskOutputMode, AutomaticMaskSegmentation,
    BoxPrompt, ImagePrediction, ImagePredictorConfig, ImagePromptBatch, InteractiveVideoOutput,
    InteractiveVideoSession, PointPrompt, RemoveObjectOutput, Sam2AutomaticMaskGenerator,
    Sam2ImagePredictor, Sam2InteractiveVideoPredictor, Sam2MultiObjectVideoPredictor,
    Sam2VideoPredictor, VideoFramePrediction, VideoPredictorConfig,
};
pub use transforms::{ImageGeometry, PreparedImage, Sam2ImageTransforms};
pub use video_state::{
    MemoryFrame, ObjectInferenceState, TrackedFrame, VideoInferenceState, VideoPromptInput,
};
