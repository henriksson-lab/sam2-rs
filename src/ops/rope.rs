//! Rotary position encoding used by SAM2 temporal attention.

use candle_core::{Result, Tensor};

/// Builds axial RoPE cosine/sine frequencies equivalent to
/// `compute_axial_cis`, represented as real tensors to avoid complex dtypes.
pub fn axial_frequencies(
    head_dim: usize,
    width: usize,
    height: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    assert_eq!(head_dim % 4, 0);
    let quarter = head_dim / 4;
    let mut cos = Vec::with_capacity(width * height * head_dim / 2);
    let mut sin = Vec::with_capacity(width * height * head_dim / 2);
    for y in 0..height {
        for x in 0..width {
            for axis in [x, y] {
                for index in 0..quarter {
                    let angle = axis as f64 / theta.powf((index * 4) as f64 / head_dim as f64);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (width * height, head_dim / 2), device)?,
        Tensor::from_vec(sin, (width * height, head_dim / 2), device)?,
    ))
}

/// Applies rotary encoding to BxHeadsxTokensxHeadDim tensors. `cos`/`sin`
/// have Tokensx(HeadDim/2) layout.
pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, heads, tokens, dim) = x.dims4()?;
    assert_eq!(cos.dims(), &[tokens, dim / 2]);
    let even_indexes = Tensor::from_vec(
        (0..dim).step_by(2).map(|v| v as u32).collect::<Vec<_>>(),
        dim / 2,
        x.device(),
    )?;
    let odd_indexes = Tensor::from_vec(
        (1..dim).step_by(2).map(|v| v as u32).collect::<Vec<_>>(),
        dim / 2,
        x.device(),
    )?;
    let even = x.index_select(&even_indexes, 3)?;
    let odd = x.index_select(&odd_indexes, 3)?;
    let cos = cos.reshape((1, 1, tokens, dim / 2))?;
    let sin = sin.reshape((1, 1, tokens, dim / 2))?;
    let first = even
        .broadcast_mul(&cos)?
        .broadcast_sub(&odd.broadcast_mul(&sin)?)?;
    let second = even
        .broadcast_mul(&sin)?
        .broadcast_add(&odd.broadcast_mul(&cos)?)?;
    Tensor::stack(&[&first, &second], 4)?.reshape((batch, heads, tokens, dim))
}
