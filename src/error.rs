use std::path::PathBuf;

use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O while reading checkpoint {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("could not parse safetensors checkpoint {path}: {source}")]
    SafeTensors {
        path: PathBuf,
        source: safetensors::SafeTensorError,
    },

    #[error(
        "unsupported checkpoint format for {0}; convert the PyTorch state_dict to safetensors first"
    )]
    UnsupportedCheckpointFormat(PathBuf),

    #[error("checkpoint is missing required tensor `{key}`")]
    MissingTensor { key: String },

    #[error("checkpoint tensor `{key}` has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("invalid SAM2 configuration: {0}")]
    InvalidConfig(String),
}
