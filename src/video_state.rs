//! Typed ownership for SAM2 temporal/video (including z-slice) inference.

use std::collections::BTreeMap;

use candle_core::{Result, Tensor};

/// Memory produced after a conditioned or propagated frame.
#[derive(Clone, Debug)]
pub struct MemoryFrame {
    pub memory_features: Tensor,
    pub memory_pos_enc: Tensor,
    pub object_pointer: Tensor,
    pub low_res_mask_logits: Tensor,
}

/// Stateful temporal bank. Frame indices are caller-defined: ordinary video
/// uses time order, while z-stack callers may use slice order. `reset` is
/// deliberately explicit at decomposition boundaries.
#[derive(Clone, Debug, Default)]
pub struct VideoInferenceState {
    conditioning_frames: BTreeMap<usize, MemoryFrame>,
    non_conditioning_frames: BTreeMap<usize, MemoryFrame>,
}

impl VideoInferenceState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.conditioning_frames.clear();
        self.non_conditioning_frames.clear();
    }

    pub fn insert_conditioning(&mut self, frame_index: usize, frame: MemoryFrame) {
        self.conditioning_frames.insert(frame_index, frame);
    }

    pub fn insert_propagated(&mut self, frame_index: usize, frame: MemoryFrame) {
        self.non_conditioning_frames.insert(frame_index, frame);
    }

    pub fn conditioning(&self) -> &BTreeMap<usize, MemoryFrame> {
        &self.conditioning_frames
    }
    pub fn propagated(&self) -> &BTreeMap<usize, MemoryFrame> {
        &self.non_conditioning_frames
    }

    /// Returns conditioning frames nearest to `frame_index`, matching the
    /// ordering policy used before temporal attention assembles its memory.
    pub fn closest_conditioning(
        &self,
        frame_index: usize,
        limit: usize,
    ) -> Vec<(usize, &MemoryFrame)> {
        let mut frames: Vec<_> = self
            .conditioning_frames
            .iter()
            .map(|(&index, frame)| (index, frame))
            .collect();
        frames.sort_by_key(|(index, _)| index.abs_diff(frame_index));
        frames.truncate(limit);
        frames
    }

    /// Ordered temporal memory candidates: nearest conditioning frames first,
    /// followed by the most recent propagated frames in the requested direction.
    pub fn temporal_candidates(
        &self,
        frame_index: usize,
        max_conditioning: usize,
        max_propagated: usize,
        reverse: bool,
    ) -> Vec<(usize, &MemoryFrame)> {
        let mut result = self.closest_conditioning(frame_index, max_conditioning);
        let mut propagated: Vec<_> = self
            .non_conditioning_frames
            .iter()
            .filter(|(index, _)| {
                if reverse {
                    **index > frame_index
                } else {
                    **index < frame_index
                }
            })
            .map(|(&index, frame)| (index, frame))
            .collect();
        propagated.sort_by_key(|(index, _)| index.abs_diff(frame_index));
        propagated.truncate(max_propagated);
        result.extend(propagated);
        result
    }

    /// Flattens selected spatial memories into SAM2's `(sequence, batch,
    /// channels)` attention layout. Object pointers remain a separate suffix
    /// because their RoPE exclusion count is significant to cross-attention.
    pub fn assemble_spatial_memory(
        &self,
        frame_index: usize,
        max_conditioning: usize,
        max_propagated: usize,
        reverse: bool,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let candidates =
            self.temporal_candidates(frame_index, max_conditioning, max_propagated, reverse);
        if candidates.is_empty() {
            return Ok(None);
        }
        let mut features = Vec::with_capacity(candidates.len());
        let mut positions = Vec::with_capacity(candidates.len());
        for (_, frame) in candidates {
            let (batch, channels, height, width) = frame.memory_features.dims4()?;
            features.push(
                frame
                    .memory_features
                    .reshape((batch, channels, height * width))?
                    .permute((2, 0, 1))?,
            );
            positions.push(
                frame
                    .memory_pos_enc
                    .reshape((batch, channels, height * width))?
                    .permute((2, 0, 1))?,
            );
        }
        let feature_refs: Vec<_> = features.iter().collect();
        let position_refs: Vec<_> = positions.iter().collect();
        Ok(Some((
            Tensor::cat(&feature_refs, 0)?,
            Tensor::cat(&position_refs, 0)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_clears_both_memory_banks() {
        let mut state = VideoInferenceState::new();
        state.reset();
        assert!(state.conditioning().is_empty());
        assert!(state.propagated().is_empty());
    }
}
