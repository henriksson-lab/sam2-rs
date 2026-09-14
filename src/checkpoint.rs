//! Checkpoint inspection and validation.
//!
//! SAM2 publishes PyTorch archives. Candle consumes safetensors efficiently, so
//! this boundary indexes converted safetensors now and makes unsupported archive
//! input explicit instead of leaking format assumptions into model modules.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use safetensors::{Dtype, SafeTensors};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    SafeTensors,
    PyTorch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckpointIndex {
    tensors: BTreeMap<String, TensorInfo>,
}

impl CheckpointIndex {
    /// Reads names, shapes, and dtypes without constructing a model.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match checkpoint_format(path) {
            CheckpointFormat::SafeTensors => Self::from_safetensors(path),
            CheckpointFormat::PyTorch => Err(Error::UnsupportedCheckpointFormat(path.into())),
        }
    }

    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| Error::Io {
            path: path.into(),
            source,
        })?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|source| Error::SafeTensors {
                path: path.into(),
                source,
            })?
            .tensors()
            .into_iter()
            .map(|(key, view)| {
                (
                    normalize_key(&key),
                    TensorInfo {
                        shape: view.shape().to_vec(),
                        dtype: view.dtype(),
                    },
                )
            })
            .collect();
        Ok(Self { tensors })
    }

    pub fn get(&self, key: &str) -> Option<&TensorInfo> {
        self.tensors.get(&normalize_key(key))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    pub fn validate(&self, manifest: &CheckpointManifest) -> Result<()> {
        for requirement in &manifest.required {
            let Some(found) = self.get(&requirement.key) else {
                return Err(Error::MissingTensor {
                    key: requirement.key.clone(),
                });
            };
            if let Some(expected) = &requirement.shape {
                if found.shape != *expected {
                    return Err(Error::TensorShape {
                        key: requirement.key.clone(),
                        expected: expected.clone(),
                        actual: found.shape.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirement {
    pub key: String,
    /// `None` verifies presence only, useful for repeated module-list keys.
    pub shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckpointManifest {
    required: Vec<TensorRequirement>,
}

impl CheckpointManifest {
    pub fn new(required: impl IntoIterator<Item = TensorRequirement>) -> Self {
        Self {
            required: required.into_iter().collect(),
        }
    }

    pub fn required(&self) -> &[TensorRequirement] {
        &self.required
    }

    /// Minimal architectural sentinels. The completed model will extend this to
    /// every parameter key; validating these before construction catches a v1,
    /// incomplete, or incorrectly-prefixed checkpoint immediately.
    pub fn sam2_1_hiera_tiny() -> Self {
        Self::new([
            required(
                "image_encoder.trunk.patch_embed.proj.weight",
                &[96, 3, 7, 7],
            ),
            required("image_encoder.trunk.blocks.0.attn.qkv.weight", &[288, 96]),
            required("image_encoder.neck.convs.0.conv.weight", &[256, 768, 1, 1]),
            required("sam_prompt_encoder.not_a_point_embed.weight", &[1, 256]),
            required("sam_mask_decoder.iou_token.weight", &[1, 256]),
            required(
                "memory_encoder.mask_downsampler.encoder.0.weight",
                &[4, 1, 3, 3],
            ),
            required("memory_attention.layers.0.linear1.weight", &[2048, 256]),
            required("no_mem_embed", &[1, 1, 256]),
        ])
    }
}

fn required(key: &str, shape: &[usize]) -> TensorRequirement {
    TensorRequirement {
        key: key.into(),
        shape: Some(shape.into()),
    }
}

pub fn checkpoint_format(path: &Path) -> CheckpointFormat {
    match path.extension().and_then(|value| value.to_str()) {
        Some("safetensors") => CheckpointFormat::SafeTensors,
        _ => CheckpointFormat::PyTorch,
    }
}

/// Normalizes common wrappers emitted by PyTorch training and conversion tools.
/// Native SAM2 key paths themselves are preserved unchanged.
pub fn normalize_key(key: &str) -> String {
    key.strip_prefix("model.")
        .or_else(|| key.strip_prefix("module."))
        .unwrap_or(key)
        .to_owned()
}

pub fn converted_path(source: &Path) -> PathBuf {
    source.with_extension("safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_only_outer_wrapper() {
        assert_eq!(
            normalize_key("model.image_encoder.trunk.pos_embed"),
            "image_encoder.trunk.pos_embed"
        );
        assert_eq!(normalize_key("module.no_mem_embed"), "no_mem_embed");
        assert_eq!(
            normalize_key("image_encoder.neck.convs.0.conv.weight"),
            "image_encoder.neck.convs.0.conv.weight"
        );
    }

    #[test]
    fn tiny_config_is_valid() {
        crate::config::Sam2Config::hiera_tiny().validate().unwrap();
    }
}
