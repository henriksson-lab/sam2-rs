//! Rotary position encoding used by SAM2 temporal attention.

use candle_core::{DType, Result, Storage, Tensor};
use rayon::prelude::*;

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
    if x.device().is_cpu()
        && x.dtype() == DType::F32
        && cos.dtype() == DType::F32
        && sin.dtype() == DType::F32
        && let (Some((x_start, x_end)), Some((cos_start, cos_end)), Some((sin_start, sin_end))) = (
            x.layout().contiguous_offsets(),
            cos.layout().contiguous_offsets(),
            sin.layout().contiguous_offsets(),
        )
    {
        let (x_storage, _) = x.storage_and_layout();
        let (cos_storage, _) = cos.storage_and_layout();
        let (sin_storage, _) = sin.storage_and_layout();
        if let (Storage::Cpu(x_storage), Storage::Cpu(cos_storage), Storage::Cpu(sin_storage)) =
            (&*x_storage, &*cos_storage, &*sin_storage)
        {
            let input = &x_storage.as_slice::<f32>()?[x_start..x_end];
            let cosine = &cos_storage.as_slice::<f32>()?[cos_start..cos_end];
            let sine = &sin_storage.as_slice::<f32>()?[sin_start..sin_end];
            if input.len() == batch * heads * tokens * dim
                && cosine.len() == tokens * dim / 2
                && sine.len() == cosine.len()
            {
                let mut output = vec![0f32; input.len()];
                output
                    .par_chunks_mut(dim)
                    .enumerate()
                    .for_each(|(row, destination)| {
                        let source = &input[row * dim..(row + 1) * dim];
                        let token = row % tokens;
                        let frequency = token * dim / 2;
                        for pair in 0..dim / 2 {
                            let even = source[pair * 2];
                            let odd = source[pair * 2 + 1];
                            let cos = cosine[frequency + pair];
                            let sin = sine[frequency + pair];
                            destination[pair * 2] = even * cos - odd * sin;
                            destination[pair * 2 + 1] = even * sin + odd * cos;
                        }
                    });
                return Tensor::from_vec(output, (batch, heads, tokens, dim), x.device());
            }
        }
    }
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

/// Applies a base RoPE table repeatedly to spatial key grids while copying an
/// optional unrotated token tail directly into the same output allocation.
///
/// SAM2 temporal cross-attention stores several 64x64 memory grids followed by
/// object-pointer tokens. The reference repeats its frequency table for those
/// grids, then concatenates the unchanged pointer tail. This CPU path indexes
/// the base table modulo one grid and writes both regions once, avoiding the
/// repeated cosine/sine tensors and the concatenation copy.
pub fn apply_rotary_repeated(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    unrotated_tail: usize,
) -> Result<Tensor> {
    let (batch, heads, tokens, dim) = x.dims4()?;
    let base_tokens = cos.dim(0)?;
    assert_eq!(cos.dims(), &[base_tokens, dim / 2]);
    assert_eq!(sin.dims(), cos.dims());
    let rotated_tokens = tokens
        .checked_sub(unrotated_tail)
        .ok_or_else(|| candle_core::Error::Msg("RoPE tail exceeds key tokens".into()))?;
    if base_tokens == 0 || rotated_tokens % base_tokens != 0 {
        candle_core::bail!(
            "RoPE key count {rotated_tokens} is not a whole number of {base_tokens}-token grids"
        )
    }
    if x.device().is_cpu()
        && x.dtype() == DType::F32
        && cos.dtype() == DType::F32
        && sin.dtype() == DType::F32
        && let (Some((x_start, x_end)), Some((cos_start, cos_end)), Some((sin_start, sin_end))) = (
            x.layout().contiguous_offsets(),
            cos.layout().contiguous_offsets(),
            sin.layout().contiguous_offsets(),
        )
    {
        let (x_storage, _) = x.storage_and_layout();
        let (cos_storage, _) = cos.storage_and_layout();
        let (sin_storage, _) = sin.storage_and_layout();
        if let (Storage::Cpu(x_storage), Storage::Cpu(cos_storage), Storage::Cpu(sin_storage)) =
            (&*x_storage, &*cos_storage, &*sin_storage)
        {
            let input = &x_storage.as_slice::<f32>()?[x_start..x_end];
            let cosine = &cos_storage.as_slice::<f32>()?[cos_start..cos_end];
            let sine = &sin_storage.as_slice::<f32>()?[sin_start..sin_end];
            if input.len() == batch * heads * tokens * dim
                && cosine.len() == base_tokens * dim / 2
                && sine.len() == cosine.len()
            {
                let mut output = vec![0f32; input.len()];
                output
                    .par_chunks_mut(dim)
                    .enumerate()
                    .for_each(|(row, destination)| {
                        let source = &input[row * dim..(row + 1) * dim];
                        let token = row % tokens;
                        if token >= rotated_tokens {
                            destination.copy_from_slice(source);
                            return;
                        }
                        let frequency = (token % base_tokens) * dim / 2;
                        for pair in 0..dim / 2 {
                            let even = source[pair * 2];
                            let odd = source[pair * 2 + 1];
                            let cos = cosine[frequency + pair];
                            let sin = sine[frequency + pair];
                            destination[pair * 2] = even * cos - odd * sin;
                            destination[pair * 2 + 1] = even * sin + odd * cos;
                        }
                    });
                return Tensor::from_vec(output, (batch, heads, tokens, dim), x.device());
            }
        }
    }
    let repeats = rotated_tokens / base_tokens;
    let repeated_cos = cos.repeat((repeats, 1))?;
    let repeated_sin = sin.repeat((repeats, 1))?;
    let rotated = apply_rotary(
        &x.narrow(2, 0, rotated_tokens)?,
        &repeated_cos,
        &repeated_sin,
    )?;
    if unrotated_tail == 0 {
        Ok(rotated)
    } else {
        Tensor::cat(
            &[&rotated, &x.narrow(2, rotated_tokens, unrotated_tail)?],
            2,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn cpu_rotary_matches_pairwise_reference() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 1, 2, 4),
            &device,
        )?;
        let cos = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &device)?;
        let sin = Tensor::from_vec(vec![0f32, 1., 1., 0.], (2, 2), &device)?;
        assert_eq!(
            apply_rotary(&x, &cos, &sin)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1., 2., -4., 3., -6., 5., 7., 8.]
        );
        Ok(())
    }

    #[test]
    fn repeated_cpu_rotary_matches_repeat_and_concatenate() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..1 * 2 * 6 * 4)
                .map(|index| index as f32 * 0.125 - 2.)
                .collect(),
            (1, 2, 6, 4),
            &device,
        )?;
        let cos = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &device)?;
        let sin = Tensor::from_vec(vec![0f32, 1., 1., 0.], (2, 2), &device)?;
        let expected = Tensor::cat(
            &[
                &apply_rotary(
                    &x.narrow(2, 0, 4)?.contiguous()?,
                    &cos.repeat((2, 1))?,
                    &sin.repeat((2, 1))?,
                )?,
                &x.narrow(2, 4, 2)?,
            ],
            2,
        )?;
        let actual = apply_rotary_repeated(&x, &cos, &sin, 2)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
