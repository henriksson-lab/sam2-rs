//! Leaf tensor operations translated before higher-level SAM2 modules.

pub mod amg;
pub mod conv;
pub mod layer_norm_2d;
pub mod layer_norm_residual;
pub mod mask;
pub mod mlp;
pub mod position_encoding;
pub mod rope;

pub use amg::{
    MaskRle, SmallRegionMask, batch_ranges, box_xyxy_to_xywh, build_all_layer_point_grids,
    build_point_grid, calculate_stability_score, generate_crop_boxes, is_box_near_crop_edge,
    mask_to_rles, postprocess_small_regions, uncrop_boxes_xyxy, uncrop_masks, uncrop_points,
    uncrop_rle,
};
pub use conv::{
    PatchEmbed, WindowPartition, window_partition, window_unpartition,
    window_unpartition_add_residual,
};
pub use layer_norm_2d::LayerNorm2d;
pub use layer_norm_residual::layer_norm_add_residual;
pub use mask::{
    SmallRegionMode, apply_non_overlapping_constraints, batched_mask_to_box, concat_points,
    connected_components_8, fill_holes_in_mask_scores, mask_to_box, remove_small_regions,
    restore_video_masks,
};
pub use mlp::{Activation, Mlp};
pub use position_encoding::{PositionEmbeddingRandom, PositionEmbeddingSine};
pub use rope::{apply_rotary, apply_rotary_repeated, axial_frequencies};
