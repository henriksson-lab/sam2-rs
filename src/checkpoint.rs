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

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{
    Sam2Variant,
    error::{Error, Result},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    SafeTensors,
    PyTorch,
}

/// Opens either a converted safetensors file or a published SAM2 PyTorch
/// archive. Published `.pt` files wrap the actual state dict under `model`.
pub fn load_var_builder(
    path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    let path = path.as_ref();
    match checkpoint_format(path) {
        CheckpointFormat::SafeTensors => unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], dtype, device)
        },
        CheckpointFormat::PyTorch => VarBuilder::from_pth_with_state(path, dtype, "model", device),
    }
}

pub fn validate_sam2_1(path: impl AsRef<Path>, variant: Sam2Variant) -> candle_core::Result<()> {
    CheckpointIndex::from_path(path)
        .and_then(|index| index.validate(&CheckpointManifest::sam2_1(variant)))
        .map_err(|error| candle_core::Error::Msg(error.to_string()))
}

/// Validate a decoder-only child checkpoint emitted by native fine-tuning. It must contain every
/// translated `sam_mask_decoder.*` tensor with its upstream shape and no unrelated tensors.
pub fn validate_sam2_1_mask_decoder(
    path: impl AsRef<Path>,
    variant: Sam2Variant,
) -> candle_core::Result<()> {
    let decoder = CheckpointManifest::new(
        CheckpointManifest::sam2_1(variant)
            .required()
            .iter()
            .filter(|tensor| tensor.key.starts_with("sam_mask_decoder."))
            .cloned(),
    );
    CheckpointIndex::from_path(path)
        .and_then(|index| index.validate(&decoder))
        .map_err(|error| candle_core::Error::Msg(error.to_string()))
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
            CheckpointFormat::PyTorch => Self::from_pytorch(path),
        }
    }

    pub fn from_pytorch(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let archive =
            candle_core::pickle::PthTensors::new(path, Some("model")).map_err(|source| {
                Error::PyTorch {
                    path: path.into(),
                    source,
                }
            })?;
        let tensors = archive
            .tensor_infos()
            .iter()
            .map(|(key, info)| {
                let dtype = match info.dtype {
                    DType::U8 => Dtype::U8,
                    DType::U32 => Dtype::U32,
                    DType::I16 => Dtype::I16,
                    DType::I32 => Dtype::I32,
                    DType::I64 => Dtype::I64,
                    DType::BF16 => Dtype::BF16,
                    DType::F16 => Dtype::F16,
                    DType::F32 => Dtype::F32,
                    DType::F64 => Dtype::F64,
                    DType::F8E4M3 => Dtype::F8_E4M3,
                    DType::F6E2M3 => Dtype::F6_E2M3,
                    DType::F6E3M2 => Dtype::F6_E3M2,
                    DType::F4 => Dtype::F4,
                    DType::F8E8M0 => Dtype::F8_E8M0,
                    _ => Dtype::F32,
                };
                (
                    normalize_key(key),
                    TensorInfo {
                        shape: info.layout.shape().dims().to_vec(),
                        dtype,
                    },
                )
            })
            .collect();
        Ok(Self { tensors })
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
        let mut problems = Vec::new();
        for requirement in &manifest.required {
            let Some(found) = self.get(&requirement.key) else {
                problems.push(format!("missing `{}`", requirement.key));
                continue;
            };
            if let Some(expected) = &requirement.shape {
                if found.shape != *expected {
                    problems.push(format!(
                        "shape `{}` is {:?}, expected {:?}",
                        requirement.key, found.shape, expected
                    ));
                }
            }
        }
        let expected = manifest
            .required
            .iter()
            .map(|item| item.key.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for key in self.tensors.keys() {
            if !expected.contains(key.as_str()) {
                problems.push(format!("unexpected `{key}`"));
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Error::CheckpointManifest(problems.join("; ")))
        }
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

    /// Complete published state-dict manifest for a SAM2.1 Hiera variant.
    pub fn sam2_1(variant: Sam2Variant) -> Self {
        let source = match variant {
            Sam2Variant::HieraTiny => include_str!("../manifests/sam2.1_hiera_tiny.manifest"),
            Sam2Variant::HieraSmall => include_str!("../manifests/sam2.1_hiera_small.manifest"),
            Sam2Variant::HieraBasePlus => {
                include_str!("../manifests/sam2.1_hiera_base_plus.manifest")
            }
            Sam2Variant::HieraLarge => include_str!("../manifests/sam2.1_hiera_large.manifest"),
        };
        Self::new(source.lines().map(|line| {
            let mut fields = line.split('\t');
            let key = fields.next().expect("manifest key").to_owned();
            let shape = fields
                .next()
                .expect("manifest shape")
                .split(',')
                .filter(|value| !value.is_empty())
                .map(|value| value.parse().expect("numeric manifest dimension"))
                .collect();
            TensorRequirement {
                key,
                shape: Some(shape),
            }
        }))
    }

    /// Minimal architectural sentinels. The completed model will extend this to
    /// every parameter key; validating these before construction catches a v1,
    /// incomplete, or incorrectly-prefixed checkpoint immediately.
    pub fn sam2_1_hiera_tiny() -> Self {
        Self::sam2_1(Sam2Variant::HieraTiny)
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
    use std::io::Write;

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

    #[test]
    fn complete_manifests_cover_every_published_variant() {
        assert_eq!(
            CheckpointManifest::sam2_1(Sam2Variant::HieraTiny)
                .required()
                .len(),
            471
        );
        assert_eq!(
            CheckpointManifest::sam2_1(Sam2Variant::HieraSmall)
                .required()
                .len(),
            519
        );
        assert_eq!(
            CheckpointManifest::sam2_1(Sam2Variant::HieraBasePlus)
                .required()
                .len(),
            615
        );
        assert_eq!(
            CheckpointManifest::sam2_1(Sam2Variant::HieraLarge)
                .required()
                .len(),
            903
        );
    }

    #[test]
    fn manifest_error_reports_missing_wrong_shape_and_unexpected_keys() {
        let index = CheckpointIndex {
            tensors: BTreeMap::from([
                (
                    "wrong_shape".into(),
                    TensorInfo {
                        shape: vec![3],
                        dtype: Dtype::F32,
                    },
                ),
                (
                    "extra".into(),
                    TensorInfo {
                        shape: vec![1],
                        dtype: Dtype::F32,
                    },
                ),
            ]),
        };
        let manifest = CheckpointManifest::new([
            TensorRequirement {
                key: "missing".into(),
                shape: Some(vec![1]),
            },
            TensorRequirement {
                key: "wrong_shape".into(),
                shape: Some(vec![2]),
            },
        ]);
        let error = index.validate(&manifest).unwrap_err().to_string();
        assert!(error.contains("missing `missing`"));
        assert!(error.contains("shape `wrong_shape` is [3], expected [2]"));
        assert!(error.contains("unexpected `extra`"));
    }

    #[test]
    fn truncated_pytorch_archive_fails_during_indexing() {
        let mut archive = tempfile::Builder::new().suffix(".pt").tempfile().unwrap();
        archive.write_all(b"not a pytorch archive").unwrap();
        let error = CheckpointIndex::from_path(archive.path()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("could not parse PyTorch checkpoint")
        );
    }
}
