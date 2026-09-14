//! Translation of `sam2/modeling/backbones/hieradet.py`.

use std::time::Instant;

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder, linear, ops::softmax};

use crate::{
    config::HieraConfig,
    ops::{Activation, Mlp, PatchEmbed, WindowPartition, window_partition, window_unpartition},
};

#[derive(Clone, Debug)]
struct MultiScaleAttention {
    dim_out: usize,
    num_heads: usize,
    q_pool: bool,
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
}

impl MultiScaleAttention {
    fn load(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        q_pool: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            dim_out,
            num_heads,
            q_pool,
            qkv: linear(dim, dim_out * 3, vb.pp("qkv"))?,
            proj: linear(dim_out, dim_out, vb.pp("proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, mut height, mut width, _) = x.dims4()?;
        let qkv =
            self.qkv
                .forward(x)?
                .reshape((batch, height * width, 3usize, self.num_heads, ()))?;
        let mut q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        if self.q_pool {
            q = pool_nhwc(&q.reshape((batch, height, width, ()))?)?;
            height = q.dim(1)?;
            width = q.dim(2)?;
            q = q.reshape((batch, height * width, self.num_heads, ()))?;
        }
        let head_dim = self.dim_out / self.num_heads;
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;
        let attention = softmax(
            &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(1. / (head_dim as f64).sqrt(), 0.)?,
            3,
        )?;
        self.proj
            .forward(&attention.matmul(&v)?.permute((0, 2, 1, 3))?.reshape((
                batch,
                height,
                width,
                self.dim_out,
            ))?)
    }
}

#[derive(Clone, Debug)]
struct MultiScaleBlock {
    dim: usize,
    dim_out: usize,
    norm1: LayerNorm,
    attn: MultiScaleAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    proj: Option<candle_nn::Linear>,
    q_pool: bool,
    window_size: usize,
}

impl MultiScaleBlock {
    fn load(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        q_pool: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            dim,
            dim_out,
            norm1: candle_nn::layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: MultiScaleAttention::load(dim, dim_out, num_heads, q_pool, vb.pp("attn"))?,
            norm2: candle_nn::layer_norm(dim_out, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::load(
                dim_out,
                dim_out * 4,
                dim_out,
                2,
                Activation::Gelu,
                false,
                vb.pp("mlp"),
            )?,
            proj: (dim != dim_out)
                .then(|| linear(dim, dim_out, vb.pp("proj")))
                .transpose()?,
            q_pool,
            window_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut shortcut = x.clone();
        let x = self.norm1.forward(x)?;
        if self.dim != self.dim_out {
            shortcut = pool_nhwc(&self.proj.as_ref().unwrap().forward(&x)?)?;
        }
        let (x, partition) = if self.window_size > 0 {
            let (windows, partition) = window_partition(&x, self.window_size)?;
            (windows, Some(partition))
        } else {
            (x, None)
        };
        let x = self.attn.forward(&x)?;
        let x = if self.window_size > 0 {
            let (window_size, partition) = if self.q_pool {
                let window_size = self.window_size / 2;
                let h = shortcut.dim(1)?;
                let w = shortcut.dim(2)?;
                let partition = WindowPartition {
                    padded_height: h + (window_size - h % window_size) % window_size,
                    padded_width: w + (window_size - w % window_size) % window_size,
                };
                (window_size, partition)
            } else {
                (self.window_size, partition.unwrap())
            };
            window_unpartition(
                &x,
                window_size,
                partition,
                shortcut.dim(1)?,
                shortcut.dim(2)?,
            )?
        } else {
            x
        };
        let x = shortcut.broadcast_add(&x)?;
        x.broadcast_add(&self.mlp.forward(&self.norm2.forward(&x)?)?)
    }
}

/// Hiera image trunk. Its `forward` returns stage features in fine-to-coarse
/// order, exactly as the reference loop appends them.
#[derive(Clone, Debug)]
pub struct Hiera {
    patch_embed: PatchEmbed,
    pos_embed: Tensor,
    pos_embed_window: Tensor,
    blocks: Vec<MultiScaleBlock>,
    stage_ends: [usize; 4],
}

/// Synchronized component timings for one Hiera image encoding.
#[derive(Clone, Debug)]
pub struct HieraProfile {
    pub setup_ms: f64,
    pub block_ms: Vec<f64>,
}

impl Hiera {
    pub fn load(config: &HieraConfig, vb: VarBuilder) -> Result<Self> {
        let depth: usize = config.stages.iter().sum();
        let stage_ends = [
            config.stages[0] - 1,
            config.stages[0] + config.stages[1] - 1,
            config.stages[0] + config.stages[1] + config.stages[2] - 1,
            depth - 1,
        ];
        let q_pool_blocks = [stage_ends[0] + 1, stage_ends[1] + 1, stage_ends[2] + 1];
        let mut blocks = Vec::with_capacity(depth);
        let mut dim = config.embed_dim;
        let mut heads = config.num_heads;
        let mut stage = 0;
        for index in 0..depth {
            let mut dim_out = dim;
            let mut window_size = config.window_spec[stage];
            if config.global_att_blocks.contains(&index) {
                window_size = 0;
            }
            if index > 0 && stage_ends[..3].contains(&(index - 1)) {
                dim_out *= 2;
                heads *= 2;
                stage += 1;
            }
            blocks.push(MultiScaleBlock::load(
                dim,
                dim_out,
                heads,
                q_pool_blocks.contains(&index),
                window_size,
                vb.pp(format!("blocks.{index}")),
            )?);
            dim = dim_out;
        }
        Ok(Self {
            patch_embed: PatchEmbed::load(3, config.embed_dim, 7, 4, 3, vb.pp("patch_embed"))?,
            pos_embed: vb.get(
                (
                    1,
                    config.embed_dim,
                    config.window_pos_embed_bkg_spatial_size[0],
                    config.window_pos_embed_bkg_spatial_size[1],
                ),
                "pos_embed",
            )?,
            pos_embed_window: vb.get(
                (
                    1,
                    config.embed_dim,
                    config.window_spec[0],
                    config.window_spec[0],
                ),
                "pos_embed_window",
            )?,
            blocks,
            stage_ends,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Vec<Tensor>> {
        let mut x = self.patch_embed.forward(image)?;
        x = x.broadcast_add(&self.position_embedding(x.dim(1)?, x.dim(2)?)?)?;
        let mut output = Vec::with_capacity(4);
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if self.stage_ends.contains(&index) {
                output.push(x.permute((0, 3, 1, 2))?);
            }
        }
        Ok(output)
    }

    /// Runs Hiera while recording the patch/position setup and every block.
    /// Synchronization makes these timings valid for either CPU or CUDA.
    pub fn forward_profiled(&self, image: &Tensor) -> Result<(Vec<Tensor>, HieraProfile)> {
        let start = Instant::now();
        let mut x = self.patch_embed.forward(image)?;
        x = x.broadcast_add(&self.position_embedding(x.dim(1)?, x.dim(2)?)?)?;
        x.device().synchronize()?;
        let setup_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut output = Vec::with_capacity(4);
        let mut block_ms = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let start = Instant::now();
            x = block.forward(&x)?;
            x.device().synchronize()?;
            block_ms.push(start.elapsed().as_secs_f64() * 1000.);
            if self.stage_ends.contains(&index) {
                output.push(x.permute((0, 3, 1, 2))?);
            }
        }
        Ok((output, HieraProfile { setup_ms, block_ms }))
    }

    fn position_embedding(&self, height: usize, width: usize) -> Result<Tensor> {
        // PyTorch F.interpolate(..., mode="bicubic", align_corners=False).
        let pos = bicubic_resize(&self.pos_embed, height, width)?;
        let repeats_h = height / self.pos_embed_window.dim(2)?;
        let repeats_w = width / self.pos_embed_window.dim(3)?;
        let tiled = self.pos_embed_window.repeat((1, 1, repeats_h, repeats_w))?;
        pos.broadcast_add(&tiled)?.permute((0, 2, 3, 1))
    }
}

fn pool_nhwc(x: &Tensor) -> Result<Tensor> {
    x.permute((0, 3, 1, 2))?
        .max_pool2d_with_stride(2, 2)?
        .permute((0, 2, 3, 1))
}

fn cubic_weight(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1. {
        // PyTorch bicubic interpolation uses Keys' cubic convolution with
        // a=-0.75 (not Catmull-Rom's a=-0.5).
        (1.25 * x - 2.25) * x * x + 1.
    } else if x < 2. {
        ((-0.75 * x + 3.75) * x - 6.) * x + 3.
    } else {
        0.
    }
}

fn bicubic_resize(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (_, _, in_h, in_w) = x.dims4()?;
    let (indices_h, weights_h) = cubic_indices(in_h, out_h, x)?;
    let (indices_w, weights_w) = cubic_indices(in_w, out_w, x)?;
    let mut rows = Tensor::zeros((x.dim(0)?, x.dim(1)?, out_h, in_w), x.dtype(), x.device())?;
    for n in 0..4 {
        rows = rows.broadcast_add(
            &x.index_select(&indices_h[n], 2)?
                .broadcast_mul(&weights_h[n].reshape((1, 1, out_h, 1))?)?,
        )?;
    }
    let mut output = Tensor::zeros((x.dim(0)?, x.dim(1)?, out_h, out_w), x.dtype(), x.device())?;
    for n in 0..4 {
        output = output.broadcast_add(
            &rows
                .index_select(&indices_w[n], 3)?
                .broadcast_mul(&weights_w[n].reshape((1, 1, 1, out_w))?)?,
        )?;
    }
    Ok(output)
}

fn cubic_indices(input: usize, output: usize, x: &Tensor) -> Result<([Tensor; 4], [Tensor; 4])> {
    let mut indices: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::with_capacity(output));
    let mut weights: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(output));
    for out in 0..output {
        let source = (out as f64 + 0.5) * input as f64 / output as f64 - 0.5;
        let base = source.floor() as isize;
        for n in 0..4 {
            let index = (base + n as isize - 1).clamp(0, input as isize - 1) as u32;
            indices[n].push(index);
            weights[n].push(cubic_weight(source - (base + n as isize - 1) as f64) as f32);
        }
    }
    Ok((
        std::array::from_fn(|n| Tensor::from_vec(indices[n].clone(), output, x.device()).unwrap()),
        std::array::from_fn(|n| Tensor::from_vec(weights[n].clone(), output, x.device()).unwrap()),
    ))
}
