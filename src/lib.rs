//! Native, Candle-backed translation of Meta's Segment Anything Model 2.
//!
//! The public API deliberately keeps checkpoint loading separate from model
//! construction so callers do not need to care whether a checkpoint originated
//! as a PyTorch archive or a converted safetensors file.

pub mod checkpoint;
pub mod config;
mod cpu_allocator;
pub mod error;
pub mod modeling;
pub mod ops;
pub mod predictor;
pub mod transforms;
pub mod video_state;

pub use checkpoint::{CheckpointFormat, CheckpointIndex, CheckpointManifest, TensorInfo};
pub use config::{HieraConfig, Sam2Config, Sam2Variant};
pub use error::{Error, Result};
pub use modeling::{
    Attention, BackboneOutput, FpnNeck, Hiera, HieraProfile, ImageEncoder, ImageEncoderProfile,
    MaskDecoder, MaskDecoderOutput, MaskDecoderProfile, MemoryAttention, MemoryEncoder,
    MemoryEncoderOutput, PromptEncoder, PromptEncoderOutput, RopeAttention, Sam2Base,
    Sam2FrameOutput, TwoWayTransformer, TwoWayTransformerProfile,
};
pub use predictor::{
    AutomaticMask, AutomaticMaskConfig, BoxPrompt, ImagePrediction, MaskRle, PointPrompt,
    Sam2AutomaticMaskGenerator, Sam2ImagePredictor, Sam2MultiObjectVideoPredictor,
    Sam2VideoPredictor, VideoFramePrediction,
};
pub use transforms::{ImageGeometry, PreparedImage, Sam2ImageTransforms};
pub use video_state::{MemoryFrame, VideoInferenceState};
