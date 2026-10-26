//! Trainable-variable boundary for Blockflow-owned native training.
//!
//! This module does not define losses, sampling, an optimizer, or a training loop. It constructs
//! the faithfully translated SAM2 mask decoder in a caller-owned [`VarMap`] and copies the exact
//! upstream checkpoint tensors into it. Blockflow owns how those variables are optimized.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::{MaskDecoder, MaskDecoderConfig, Sam2Variant};

/// Frozen encoder/prompt outputs consumed by a trainable translated mask decoder.
#[derive(Clone, Debug)]
pub struct FrozenDecoderInputs {
    pub image_embeddings: Tensor,
    pub image_pe: Tensor,
    pub sparse_prompt_embeddings: Tensor,
    pub dense_prompt_embeddings: Tensor,
    /// Unprojected 256x256 and 128x128 FPN features in upstream order.
    pub high_res_features: [Tensor; 2],
    pub geometry: crate::ImageGeometry,
}

impl TrainableMaskDecoder {
    pub fn forward(
        &self,
        inputs: &FrozenDecoderInputs,
        multimask_output: bool,
    ) -> Result<crate::MaskDecoderOutput> {
        self.decoder.forward_differentiable(
            &inputs.image_embeddings,
            &inputs.image_pe,
            &inputs.sparse_prompt_embeddings,
            &inputs.dense_prompt_embeddings,
            multimask_output,
            &inputs.high_res_features,
        )
    }
}

/// A translated mask decoder whose named tensors are mutable and serializable.
pub struct TrainableMaskDecoder {
    decoder: MaskDecoder,
    variables: VarMap,
    variant: Sam2Variant,
}

impl TrainableMaskDecoder {
    /// Validate a complete upstream SAM2.1 checkpoint, create decoder variables with upstream
    /// names, and copy only `sam_mask_decoder.*` tensors into owned mutable storage.
    pub fn load_sam2_1(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        config: MaskDecoderConfig,
    ) -> Result<Self> {
        Self::load_sam2_1_with_mask_decoder(variant, checkpoint, None::<&Path>, device, config)
    }

    /// Initialize from a complete immutable base checkpoint and, when supplied, overlay a
    /// validated decoder-only child artifact. This permits another fine-tuning generation without
    /// modifying the faithful encoder or accepting unrelated tensors from the child bundle.
    pub fn load_sam2_1_with_mask_decoder(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        mask_decoder: Option<impl AsRef<Path>>,
        device: &Device,
        config: MaskDecoderConfig,
    ) -> Result<Self> {
        let checkpoint = checkpoint.as_ref();
        crate::checkpoint::validate_sam2_1(checkpoint, variant)?;
        let source_path = if let Some(mask_decoder) = mask_decoder {
            let mask_decoder = mask_decoder.as_ref().to_path_buf();
            crate::checkpoint::validate_sam2_1_mask_decoder(&mask_decoder, variant)?;
            mask_decoder
        } else {
            checkpoint.to_path_buf()
        };
        let source = crate::checkpoint::load_var_builder(&source_path, DType::F32, device)?;
        let variables = VarMap::new();
        let decoder = MaskDecoder::load_with_config(
            VarBuilder::from_varmap(&variables, DType::F32, device).pp("sam_mask_decoder"),
            config,
        )?;
        {
            let variable_data = variables.data().lock().map_err(|_| {
                candle_core::Error::Msg("SAM2 decoder variable map is poisoned".into())
            })?;
            for (name, variable) in variable_data.iter() {
                let checkpoint_value = source.get(variable.shape().clone(), name)?;
                variable.set(&checkpoint_value)?;
            }
        }
        Ok(Self {
            decoder,
            variables,
            variant,
        })
    }

    pub fn variant(&self) -> Sam2Variant {
        self.variant
    }

    pub fn decoder(&self) -> &MaskDecoder {
        &self.decoder
    }

    pub fn variables(&self) -> &VarMap {
        &self.variables
    }

    /// Save a decoder-only child artifact using the original `sam_mask_decoder.*` tensor names.
    pub fn save_decoder(&self, path: impl AsRef<Path>) -> Result<()> {
        self.variables.save(path)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn differentiable_decoder_reaches_named_variables() -> Result<()> {
        let device = Device::Cpu;
        let variables = VarMap::new();
        let decoder = MaskDecoder::load_with_config(
            VarBuilder::from_varmap(&variables, DType::F32, &device).pp("sam_mask_decoder"),
            MaskDecoderConfig::raw(),
        )?;
        let image = Tensor::zeros((1, 256, 2, 2), DType::F32, &device)?;
        let image_pe = Tensor::zeros((1, 256, 2, 2), DType::F32, &device)?;
        let sparse = Tensor::zeros((1, 1, 256), DType::F32, &device)?;
        let dense = Tensor::zeros((1, 256, 2, 2), DType::F32, &device)?;
        let high_res = [
            Tensor::zeros((1, 256, 8, 8), DType::F32, &device)?,
            Tensor::zeros((1, 256, 4, 4), DType::F32, &device)?,
        ];
        let output =
            decoder.forward_differentiable(&image, &image_pe, &sparse, &dense, false, &high_res)?;
        let gradients = output.masks.mean_all()?.backward()?;
        let data = variables.data().lock().unwrap();
        let variable = data
            .get("sam_mask_decoder.mask_tokens.weight")
            .expect("named upstream decoder variable");
        let gradient = gradients
            .get(variable.as_tensor())
            .expect("mask-token gradient")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient.iter().any(|value| *value != 0.0));
        Ok(())
    }
}
