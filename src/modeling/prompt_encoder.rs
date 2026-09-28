//! Translation of `sam2/modeling/sam/prompt_encoder.py`.

use std::time::Instant;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};

use crate::{
    config::Sam2Config,
    ops::{LayerNorm2d, PositionEmbeddingRandom},
};

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn sam2_conv1x1_nchw_f32(
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        output: *mut f32,
        output_elements: u64,
        input_channels: u32,
        output_channels: u32,
        spatial: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaConv1x1Nchw;

#[cfg(feature = "cuda")]
impl candle_core::CustomOp3 for CudaConv1x1Nchw {
    fn name(&self) -> &'static str {
        "sam2-conv1x1-nchw-f32"
    }

    fn cpu_fwd(
        &self,
        _input: &candle_core::CpuStorage,
        _input_layout: &candle_core::Layout,
        _weight: &candle_core::CpuStorage,
        _weight_layout: &candle_core::Layout,
        _bias: &candle_core::CpuStorage,
        _bias_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("NCHW 1x1 convolution is only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &candle_core::Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &candle_core::Layout,
        bias: &candle_core::CudaStorage,
        bias_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        if input.dtype() != DType::F32 || weight.dtype() != DType::F32 || bias.dtype() != DType::F32
        {
            candle_core::bail!("NCHW 1x1 convolution requires f32")
        }
        let (batch, input_channels, height, width) = input_layout.shape().dims4()?;
        let (output_channels, weight_channels, kernel_h, kernel_w) =
            weight_layout.shape().dims4()?;
        if input_channels != weight_channels
            || kernel_h != 1
            || kernel_w != 1
            || bias_layout.shape().dims() != [output_channels]
        {
            candle_core::bail!("invalid NCHW 1x1 convolution shape")
        }
        let Some((input_start, _)) = input_layout.contiguous_offsets() else {
            candle_core::bail!("NCHW 1x1 convolution input must be contiguous")
        };
        let Some((weight_start, _)) = weight_layout.contiguous_offsets() else {
            candle_core::bail!("NCHW 1x1 convolution weight must be contiguous")
        };
        let Some((bias_start, _)) = bias_layout.contiguous_offsets() else {
            candle_core::bail!("NCHW 1x1 convolution bias must be contiguous")
        };
        let output_shape = candle_core::Shape::from_dims(&[batch, output_channels, height, width]);
        let output_elements = output_shape.elem_count();
        let dev = input.device();
        let stream = dev.cuda_stream();
        let input = input.as_cuda_slice::<f32>()?.slice(input_start..);
        let weight = weight.as_cuda_slice::<f32>()?.slice(weight_start..);
        let bias = bias.as_cuda_slice::<f32>()?.slice(bias_start..);
        // SAFETY: the external kernel writes each element of this exact shape.
        let mut output = unsafe { dev.alloc::<f32>(output_elements)? };
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (weight_ptr, weight_guard) = weight.device_ptr(&stream);
        let (bias_ptr, bias_guard) = bias.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_conv1x1_nchw_f32(
                input_ptr as *const f32,
                weight_ptr as *const f32,
                bias_ptr as *const f32,
                output_ptr as *mut f32,
                output_elements as u64,
                input_channels as u32,
                output_channels as u32,
                (height * width) as u32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("NCHW 1x1 convolution CUDA launch failed with status {status}")
        }
        drop((input_guard, weight_guard, bias_guard, output_guard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            output_shape,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct PromptEncoderOutput {
    pub sparse_embeddings: Tensor,
    pub dense_embeddings: Tensor,
}

#[derive(Clone, Debug)]
pub struct PromptEncoder {
    embed_dim: usize,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    pe_layer: PositionEmbeddingRandom,
    cached_dense_pe: Option<Tensor>,
    cached_no_mask_dense: Option<Tensor>,
    point_embeddings: [Tensor; 4],
    not_a_point_embed: Tensor,
    mask_conv_0: Conv2d,
    mask_norm_1: LayerNorm2d,
    mask_conv_3: Conv2d,
    mask_norm_4: LayerNorm2d,
    mask_conv_6: Conv2d,
    no_mask_embed: Tensor,
}

impl PromptEncoder {
    pub fn load(config: &Sam2Config, vb: VarBuilder) -> Result<Self> {
        let mask_channels = 16;
        let image_embedding_size = (
            config.image_size / config.backbone_stride,
            config.image_size / config.backbone_stride,
        );
        let pe_layer = PositionEmbeddingRandom::load(config.hidden_dim / 2, vb.pp("pe_layer"))?;
        let no_mask_embed = vb.get((1, config.hidden_dim), "no_mask_embed.weight")?;
        let cached_dense_pe = if no_mask_embed.device().is_cpu()
            && std::env::var_os("SAM2_CPU_CACHE_DENSE_PE").is_none_or(|value| value != "0")
        {
            Some(
                pe_layer
                    .forward(image_embedding_size.0, image_embedding_size.1)?
                    .unsqueeze(0)?,
            )
        } else {
            None
        };
        let cached_no_mask_dense = if no_mask_embed.device().is_cpu() {
            Some(
                no_mask_embed
                    .reshape((1, config.hidden_dim, 1, 1))?
                    .broadcast_as((
                        1,
                        config.hidden_dim,
                        image_embedding_size.0,
                        image_embedding_size.1,
                    ))?,
            )
        } else {
            None
        };
        Ok(Self {
            embed_dim: config.hidden_dim,
            image_embedding_size,
            input_image_size: (config.image_size, config.image_size),
            pe_layer,
            cached_dense_pe,
            cached_no_mask_dense,
            point_embeddings: [
                vb.pp("point_embeddings.0")
                    .get((1, config.hidden_dim), "weight")?,
                vb.pp("point_embeddings.1")
                    .get((1, config.hidden_dim), "weight")?,
                vb.pp("point_embeddings.2")
                    .get((1, config.hidden_dim), "weight")?,
                vb.pp("point_embeddings.3")
                    .get((1, config.hidden_dim), "weight")?,
            ],
            not_a_point_embed: vb.get((1, config.hidden_dim), "not_a_point_embed.weight")?,
            mask_conv_0: conv2d(
                1,
                mask_channels / 4,
                2,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("mask_downscaling.0"),
            )?,
            mask_norm_1: LayerNorm2d::load(mask_channels / 4, 1e-6, vb.pp("mask_downscaling.1"))?,
            mask_conv_3: conv2d(
                mask_channels / 4,
                mask_channels,
                2,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("mask_downscaling.3"),
            )?,
            mask_norm_4: LayerNorm2d::load(mask_channels, 1e-6, vb.pp("mask_downscaling.4"))?,
            mask_conv_6: conv2d(
                mask_channels,
                config.hidden_dim,
                1,
                Conv2dConfig::default(),
                vb.pp("mask_downscaling.6"),
            )?,
            no_mask_embed,
        })
    }

    pub fn get_dense_pe(&self) -> Result<Tensor> {
        if let Some(cached) = &self.cached_dense_pe {
            return Ok(cached.clone());
        }
        self.pe_layer
            .forward(self.image_embedding_size.0, self.image_embedding_size.1)?
            .unsqueeze(0)
    }

    /// `points` has BxNx2 pixels and `labels` BxN values: -1 padding, 0
    /// negative, 1 positive, 2 top-left box corner, and 3 bottom-right.
    pub fn forward(
        &self,
        points: Option<(&Tensor, &Tensor)>,
        boxes: Option<&Tensor>,
        masks: Option<&Tensor>,
    ) -> Result<PromptEncoderOutput> {
        let batch = match (points, boxes, masks) {
            (Some((coordinates, _)), _, _) => coordinates.dim(0)?,
            (_, Some(boxes), _) => boxes.dim(0)?,
            (_, _, Some(masks)) => masks.dim(0)?,
            _ => 1,
        };
        let mut sparse = Tensor::zeros(
            (batch, 0usize, self.embed_dim),
            DType::F32,
            self.no_mask_embed.device(),
        )?;
        if let Some((coordinates, labels)) = points {
            let embedded = self.embed_points(coordinates, labels, boxes.is_none())?;
            sparse = Tensor::cat(&[&sparse, &embedded], 1)?;
        }
        if let Some(boxes) = boxes {
            sparse = Tensor::cat(&[&sparse, &self.embed_boxes(boxes)?], 1)?;
        }
        let dense_embeddings = match masks {
            Some(mask) => self.embed_masks(mask)?,
            None if batch == 1 && self.cached_no_mask_dense.is_some() => {
                self.cached_no_mask_dense.as_ref().unwrap().clone()
            }
            None => self
                .no_mask_embed
                .reshape((1, self.embed_dim, 1, 1))?
                .broadcast_as((
                    batch,
                    self.embed_dim,
                    self.image_embedding_size.0,
                    self.image_embedding_size.1,
                ))?,
        };
        Ok(PromptEncoderOutput {
            sparse_embeddings: sparse,
            dense_embeddings,
        })
    }

    fn embed_points(&self, points: &Tensor, labels: &Tensor, pad: bool) -> Result<Tensor> {
        let mut points = points.affine(1., 0.5)?;
        let mut labels = labels.clone();
        if pad {
            let batch = points.dim(0)?;
            let padding_points =
                Tensor::zeros((batch, 1usize, 2usize), points.dtype(), points.device())?;
            let padding_labels =
                Tensor::full(-1i64, (batch, 1usize), labels.device())?.to_dtype(labels.dtype())?;
            points = Tensor::cat(&[&points, &padding_points], 1)?;
            labels = Tensor::cat(&[&labels, &padding_labels], 1)?;
        }
        let mut embedding = self.pe_layer.forward_with_coords(
            &points,
            self.input_image_size.0,
            self.input_image_size.1,
        )?;
        let shape = embedding.dims().to_vec();
        for (label, replacement) in [
            (-1i64, &self.not_a_point_embed),
            (0, &self.point_embeddings[0]),
            (1, &self.point_embeddings[1]),
            (2, &self.point_embeddings[2]),
            (3, &self.point_embeddings[3]),
        ] {
            let condition = labels
                .eq(label)?
                .unsqueeze(2)?
                .broadcast_as(shape.clone())?;
            let value = if label == -1 {
                replacement.broadcast_as(shape.clone())?
            } else {
                embedding.broadcast_add(&replacement.broadcast_as(shape.clone())?)?
            };
            embedding = condition.where_cond(&value, &embedding)?;
        }
        Ok(embedding)
    }

    fn embed_boxes(&self, boxes: &Tensor) -> Result<Tensor> {
        let batch = boxes.dim(0)?;
        let coords = boxes.affine(1., 0.5)?.reshape((batch, 2usize, 2usize))?;
        let embedding = self.pe_layer.forward_with_coords(
            &coords,
            self.input_image_size.0,
            self.input_image_size.1,
        )?;
        let first = embedding
            .narrow(1, 0, 1)?
            .broadcast_add(&self.point_embeddings[2])?;
        let second = embedding
            .narrow(1, 1, 1)?
            .broadcast_add(&self.point_embeddings[3])?;
        Tensor::cat(&[&first, &second], 1)
    }

    fn embed_masks(&self, masks: &Tensor) -> Result<Tensor> {
        let profiling = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
        if profiling {
            masks.device().synchronize()?;
        }
        let mut start = Instant::now();
        macro_rules! stage {
            ($name:expr) => {
                if profiling {
                    masks.device().synchronize()?;
                    eprintln!(
                        "SAM2_MASK_PROMPT_STAGE stage={} batch={} ms={:.6}",
                        $name,
                        masks.dim(0)?,
                        start.elapsed().as_secs_f64() * 1000.
                    );
                    start = Instant::now();
                }
            };
        }
        let x = self.mask_conv_0.forward(masks)?;
        stage!("conv_0");
        let x = self.mask_norm_1.forward(&x)?.gelu_erf()?;
        stage!("norm_gelu_1");
        let x = self.mask_conv_3.forward(&x)?;
        stage!("conv_3");
        let x = self.mask_norm_4.forward(&x)?.gelu_erf()?;
        stage!("norm_gelu_4");
        #[cfg(feature = "cuda")]
        let x = if x.device().is_cuda()
            && std::env::var_os("SAM2_CUDA_MASK_CONV1X1").is_none_or(|value| value != "0")
        {
            let bias = self
                .mask_conv_6
                .bias()
                .ok_or_else(|| candle_core::Error::Msg("mask projection requires a bias".into()))?;
            x.apply_op3_no_bwd(self.mask_conv_6.weight(), bias, &CudaConv1x1Nchw)?
        } else {
            self.mask_conv_6.forward(&x)?
        };
        #[cfg(not(feature = "cuda"))]
        let x = self.mask_conv_6.forward(&x)?;
        stage!("conv_6");
        let _ = start;
        Ok(x)
    }
}
