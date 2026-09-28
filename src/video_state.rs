//! Typed ownership for SAM2 temporal/video (including z-slice) inference.

use std::collections::{BTreeMap, VecDeque};

use candle_core::{Result, Tensor};

use crate::{BackboneOutput, ops::concat_points};

/// Memory produced after a conditioned or propagated frame.
#[derive(Clone, Debug)]
pub struct MemoryFrame {
    pub memory_features: Tensor,
    pub memory_pos_enc: Tensor,
    pub object_pointer: Tensor,
    pub low_res_mask_logits: Tensor,
}

#[derive(Clone, Debug)]
pub struct VideoPromptInput {
    pub coordinates: Tensor,
    pub labels: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackedFrame {
    pub reverse: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ObjectInferenceState {
    pub point_inputs: BTreeMap<usize, VideoPromptInput>,
    pub mask_inputs: BTreeMap<usize, Tensor>,
    pub conditioning_outputs: BTreeMap<usize, MemoryFrame>,
    pub non_conditioning_outputs: BTreeMap<usize, MemoryFrame>,
    pub temporary_conditioning_outputs: BTreeMap<usize, MemoryFrame>,
    pub temporary_non_conditioning_outputs: BTreeMap<usize, MemoryFrame>,
    pub tracked_frames: BTreeMap<usize, TrackedFrame>,
}

/// Stateful temporal bank. Frame indices are caller-defined: ordinary video
/// uses time order, while z-stack callers may use slice order. `reset` is
/// deliberately explicit at decomposition boundaries.
#[derive(Clone, Debug, Default)]
pub struct VideoInferenceState {
    conditioning_frames: BTreeMap<usize, MemoryFrame>,
    non_conditioning_frames: BTreeMap<usize, MemoryFrame>,
    video_width: Option<usize>,
    video_height: Option<usize>,
    num_frames: Option<usize>,
    object_ids: Vec<u32>,
    object_indices: BTreeMap<u32, usize>,
    objects: Vec<ObjectInferenceState>,
    cached_features: BTreeMap<usize, BackboneOutput>,
    feature_cache_order: VecDeque<usize>,
    feature_cache_capacity: usize,
    mask_memory_position_encoding: Option<Vec<Tensor>>,
}

impl VideoInferenceState {
    pub fn new() -> Self {
        Self {
            feature_cache_capacity: 1,
            ..Self::default()
        }
    }

    pub fn with_video(width: usize, height: usize, num_frames: usize) -> Result<Self> {
        if width == 0 || height == 0 || num_frames == 0 {
            candle_core::bail!("video dimensions and frame count must be positive")
        }
        Ok(Self {
            video_width: Some(width),
            video_height: Some(height),
            num_frames: Some(num_frames),
            ..Self::new()
        })
    }

    pub fn reset(&mut self) {
        self.conditioning_frames.clear();
        self.non_conditioning_frames.clear();
        self.object_ids.clear();
        self.object_indices.clear();
        self.objects.clear();
        self.cached_features.clear();
        self.feature_cache_order.clear();
        self.mask_memory_position_encoding = None;
    }

    fn reuse_memory_position_encoding(&mut self, frame: &mut MemoryFrame) {
        if let Some(position) = self
            .mask_memory_position_encoding
            .as_ref()
            .and_then(|positions| positions.first())
            .filter(|position| position.dims() == frame.memory_pos_enc.dims())
        {
            frame.memory_pos_enc = position.clone();
        } else {
            self.mask_memory_position_encoding = Some(vec![frame.memory_pos_enc.clone()]);
        }
    }

    pub fn insert_conditioning(&mut self, frame_index: usize, mut frame: MemoryFrame) {
        self.reuse_memory_position_encoding(&mut frame);
        self.conditioning_frames.insert(frame_index, frame);
    }

    pub fn insert_propagated(&mut self, frame_index: usize, mut frame: MemoryFrame) {
        self.reuse_memory_position_encoding(&mut frame);
        self.non_conditioning_frames.insert(frame_index, frame);
    }

    pub fn conditioning(&self) -> &BTreeMap<usize, MemoryFrame> {
        &self.conditioning_frames
    }
    pub fn propagated(&self) -> &BTreeMap<usize, MemoryFrame> {
        &self.non_conditioning_frames
    }

    pub fn video_dimensions(&self) -> Option<(usize, usize)> {
        self.video_width.zip(self.video_height)
    }

    pub fn num_frames(&self) -> Option<usize> {
        self.num_frames
    }

    pub fn object_ids(&self) -> &[u32] {
        &self.object_ids
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    pub fn object_id_for_index(&self, object_index: usize) -> Option<u32> {
        self.object_ids.get(object_index).copied()
    }

    pub fn object_index(&mut self, object_id: u32) -> usize {
        if let Some(&index) = self.object_indices.get(&object_id) {
            return index;
        }
        let index = self.objects.len();
        self.object_ids.push(object_id);
        self.object_indices.insert(object_id, index);
        self.objects.push(ObjectInferenceState::default());
        index
    }

    pub fn object(&self, object_id: u32) -> Option<&ObjectInferenceState> {
        self.object_indices
            .get(&object_id)
            .and_then(|&index| self.objects.get(index))
    }

    pub fn object_mut(&mut self, object_id: u32) -> &mut ObjectInferenceState {
        let index = self.object_index(object_id);
        &mut self.objects[index]
    }

    pub fn add_points(
        &mut self,
        object_id: u32,
        frame_index: usize,
        coordinates: &Tensor,
        labels: &Tensor,
        clear_old_points: bool,
    ) -> Result<()> {
        let object = self.object_mut(object_id);
        let old = (!clear_old_points)
            .then(|| object.point_inputs.get(&frame_index))
            .flatten()
            .map(|prompt| (&prompt.coordinates, &prompt.labels));
        let (coordinates, labels) = concat_points(old, coordinates, labels)?;
        object.point_inputs.insert(
            frame_index,
            VideoPromptInput {
                coordinates,
                labels,
            },
        );
        object.mask_inputs.remove(&frame_index);
        Ok(())
    }

    pub fn add_mask(&mut self, object_id: u32, frame_index: usize, mask: Tensor) {
        let object = self.object_mut(object_id);
        object.mask_inputs.insert(frame_index, mask);
        object.point_inputs.remove(&frame_index);
    }

    pub fn mark_tracked(&mut self, object_id: u32, frame_index: usize, reverse: bool) {
        self.object_mut(object_id)
            .tracked_frames
            .insert(frame_index, TrackedFrame { reverse });
    }

    /// Moves the latest per-object interaction output into the consolidated
    /// bank used for propagation, removing a stale output of the other kind.
    pub fn commit_temporary_frame(&mut self, frame_index: usize, conditioning: bool) {
        for object in &mut self.objects {
            let output = if conditioning {
                object.temporary_conditioning_outputs.remove(&frame_index)
            } else {
                object
                    .temporary_non_conditioning_outputs
                    .remove(&frame_index)
            };
            if let Some(output) = output {
                if conditioning {
                    object.conditioning_outputs.insert(frame_index, output);
                    object.non_conditioning_outputs.remove(&frame_index);
                } else {
                    object.non_conditioning_outputs.insert(frame_index, output);
                    object.conditioning_outputs.remove(&frame_index);
                }
            }
        }
    }

    pub fn clear_prompts_in_frame(&mut self, object_id: u32, frame_index: usize) -> bool {
        let Some(object) = self
            .object_indices
            .get(&object_id)
            .and_then(|&index| self.objects.get_mut(index))
        else {
            return false;
        };
        let changed = object.point_inputs.remove(&frame_index).is_some()
            | object.mask_inputs.remove(&frame_index).is_some();
        object.temporary_conditioning_outputs.remove(&frame_index);
        object
            .temporary_non_conditioning_outputs
            .remove(&frame_index);
        if let Some(output) = object.conditioning_outputs.remove(&frame_index) {
            object.non_conditioning_outputs.insert(frame_index, output);
            object.tracked_frames.remove(&frame_index);
        }
        changed
    }

    pub fn validate_for_propagation(&self) -> Result<()> {
        if self.objects.is_empty() {
            candle_core::bail!("no object prompts have been added")
        }
        for (index, object) in self.objects.iter().enumerate() {
            if object.conditioning_outputs.is_empty()
                && object.temporary_conditioning_outputs.is_empty()
            {
                candle_core::bail!(
                    "object {} has no conditioning input",
                    self.object_ids[index]
                )
            }
        }
        Ok(())
    }

    pub fn earliest_conditioning_frame(&self) -> Option<usize> {
        self.objects
            .iter()
            .flat_map(|object| {
                object
                    .conditioning_outputs
                    .keys()
                    .chain(object.temporary_conditioning_outputs.keys())
            })
            .copied()
            .min()
    }

    pub fn propagation_order(
        &self,
        start_frame: Option<usize>,
        maximum_frames: Option<usize>,
        reverse: bool,
    ) -> Result<Vec<usize>> {
        self.validate_for_propagation()?;
        let frame_count = self
            .num_frames
            .ok_or_else(|| candle_core::Error::Msg("video frame count is not set".into()))?;
        let start = start_frame
            .or_else(|| self.earliest_conditioning_frame())
            .ok_or_else(|| candle_core::Error::Msg("no conditioning frame exists".into()))?;
        if start >= frame_count {
            candle_core::bail!("start frame is outside the video")
        }
        let maximum = maximum_frames.unwrap_or(frame_count);
        if reverse {
            if start == 0 {
                return Ok(Vec::new());
            }
            let end = start.saturating_sub(maximum);
            Ok((end..=start).rev().collect())
        } else {
            let end = start.saturating_add(maximum).min(frame_count - 1);
            Ok((start..=end).collect())
        }
    }

    pub fn consolidated_mask_logits(
        &self,
        frame_index: usize,
        conditioning: bool,
        height: usize,
        width: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let mut masks = Vec::with_capacity(self.objects.len());
        for object in &self.objects {
            let temporary = if conditioning {
                &object.temporary_conditioning_outputs
            } else {
                &object.temporary_non_conditioning_outputs
            };
            let output = temporary
                .get(&frame_index)
                .or_else(|| object.conditioning_outputs.get(&frame_index))
                .or_else(|| object.non_conditioning_outputs.get(&frame_index));
            let mask = match output {
                Some(output) => {
                    let mask = &output.low_res_mask_logits;
                    if mask.dim(mask.rank() - 2)? == height && mask.dim(mask.rank() - 1)? == width {
                        mask.clone()
                    } else {
                        mask.to_dtype(candle_core::DType::F32)?
                            .upsample_bilinear2d(height, width, false)?
                    }
                }
                None => Tensor::full(-1024f32, (1, 1, height, width), device)?,
            };
            masks.push(mask);
        }
        let references = masks.iter().collect::<Vec<_>>();
        Tensor::cat(&references, 0)
    }

    pub fn remove_object(&mut self, object_id: u32) -> Option<ObjectInferenceState> {
        let index = self.object_indices.remove(&object_id)?;
        self.object_ids.remove(index);
        let removed = self.objects.remove(index);
        for value in self.object_indices.values_mut() {
            if *value > index {
                *value -= 1;
            }
        }
        Some(removed)
    }

    pub fn set_feature_cache_capacity(&mut self, capacity: usize) -> Result<()> {
        if capacity == 0 {
            candle_core::bail!("feature cache capacity must be positive")
        }
        self.feature_cache_capacity = capacity;
        while self.feature_cache_order.len() > capacity {
            let index = self.feature_cache_order.pop_front().unwrap();
            self.cached_features.remove(&index);
        }
        Ok(())
    }

    pub fn cache_feature(&mut self, frame_index: usize, feature: BackboneOutput) {
        if let Some(position) = self
            .feature_cache_order
            .iter()
            .position(|&index| index == frame_index)
        {
            self.feature_cache_order.remove(position);
        }
        self.feature_cache_order.push_back(frame_index);
        self.cached_features.insert(frame_index, feature);
        while self.feature_cache_order.len() > self.feature_cache_capacity.max(1) {
            let oldest = self.feature_cache_order.pop_front().unwrap();
            self.cached_features.remove(&oldest);
        }
    }

    pub fn cached_feature(&self, frame_index: usize) -> Option<&BackboneOutput> {
        self.cached_features.get(&frame_index)
    }

    pub fn cached_feature_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.feature_cache_order.iter().copied()
    }

    pub fn set_mask_memory_position_encoding(&mut self, encoding: Vec<Tensor>) {
        self.mask_memory_position_encoding = Some(encoding);
    }

    pub fn mask_memory_position_encoding(&self) -> Option<&[Tensor]> {
        self.mask_memory_position_encoding.as_deref()
    }

    pub fn clear_non_conditioning_around(&mut self, frame_index: usize, radius: usize) {
        let start = frame_index.saturating_sub(radius);
        let end = frame_index.saturating_add(radius);
        for object in &mut self.objects {
            object
                .non_conditioning_outputs
                .retain(|&index, _| index < start || index > end);
        }
        self.non_conditioning_frames
            .retain(|&index, _| index < start || index > end);
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

    fn memory(value: f32) -> MemoryFrame {
        let tensor = Tensor::new(&[[[[value]]]], &candle_core::Device::Cpu).unwrap();
        MemoryFrame {
            memory_features: tensor.clone(),
            memory_pos_enc: tensor.clone(),
            object_pointer: Tensor::new(&[[value]], &candle_core::Device::Cpu).unwrap(),
            low_res_mask_logits: tensor,
        }
    }
    #[test]
    fn reset_clears_both_memory_banks() {
        let mut state = VideoInferenceState::new();
        state.reset();
        assert!(state.conditioning().is_empty());
        assert!(state.propagated().is_empty());
    }

    #[test]
    fn object_indices_are_stable_then_compact_after_removal() {
        let mut state = VideoInferenceState::with_video(640, 480, 3).unwrap();
        assert_eq!(state.object_index(42), 0);
        assert_eq!(state.object_index(7), 1);
        assert_eq!(state.object_index(42), 0);
        assert_eq!(state.object_ids(), &[42, 7]);
        state.remove_object(42).unwrap();
        assert_eq!(state.object_ids(), &[7]);
        assert_eq!(state.object_index(7), 0);
        assert_eq!(state.video_dimensions(), Some((640, 480)));
        assert_eq!(state.num_frames(), Some(3));
    }

    #[test]
    fn propagation_order_and_missing_object_placeholders_match_reference() {
        let mut state = VideoInferenceState::with_video(4, 3, 6).unwrap();
        state
            .object_mut(11)
            .conditioning_outputs
            .insert(2, memory(1.));
        state.object_index(22);
        state
            .object_mut(22)
            .temporary_conditioning_outputs
            .insert(3, memory(2.));
        assert_eq!(
            state.propagation_order(None, Some(2), false).unwrap(),
            vec![2, 3, 4]
        );
        assert_eq!(
            state.propagation_order(Some(4), Some(2), true).unwrap(),
            vec![4, 3, 2]
        );
        let masks = state
            .consolidated_mask_logits(2, true, 1, 1, &candle_core::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(masks, vec![1., -1024.]);
    }
}
