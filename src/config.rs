//! Typed equivalents of the published SAM2.1 Hydra configurations.

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sam2Variant {
    HieraTiny,
    HieraSmall,
    HieraBasePlus,
    HieraLarge,
}

impl Sam2Variant {
    pub const ALL: [Self; 4] = [
        Self::HieraTiny,
        Self::HieraSmall,
        Self::HieraBasePlus,
        Self::HieraLarge,
    ];

    pub const fn checkpoint_file(self) -> &'static str {
        match self {
            Self::HieraTiny => "sam2.1_hiera_tiny.pt",
            Self::HieraSmall => "sam2.1_hiera_small.pt",
            Self::HieraBasePlus => "sam2.1_hiera_base_plus.pt",
            Self::HieraLarge => "sam2.1_hiera_large.pt",
        }
    }

    pub const fn hugging_face_model_id(self) -> &'static str {
        match self {
            Self::HieraTiny => "facebook/sam2.1-hiera-tiny",
            Self::HieraSmall => "facebook/sam2.1-hiera-small",
            Self::HieraBasePlus => "facebook/sam2.1-hiera-base-plus",
            Self::HieraLarge => "facebook/sam2.1-hiera-large",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HieraConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub stages: [usize; 4],
    pub global_att_blocks: Vec<usize>,
    pub window_pos_embed_bkg_spatial_size: [usize; 2],
    pub window_spec: [usize; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam2Config {
    pub variant: Sam2Variant,
    pub image_size: usize,
    pub backbone_stride: usize,
    pub hidden_dim: usize,
    pub hierarchy: HieraConfig,
    pub backbone_channel_list: [usize; 4],
    pub fpn_top_down_levels: Vec<usize>,
    pub num_maskmem: usize,
    pub sigmoid_scale_for_mem_enc: f64,
    pub sigmoid_bias_for_mem_enc: f64,
}

impl Sam2Config {
    pub fn sam2_1(variant: Sam2Variant) -> Self {
        let (embed_dim, num_heads, stages, global_att_blocks, pos_size, window_spec) = match variant
        {
            Sam2Variant::HieraTiny => (96, 1, [1, 2, 7, 2], vec![5, 7, 9], [7, 7], [8, 4, 14, 7]),
            Sam2Variant::HieraSmall => {
                (96, 1, [1, 2, 11, 2], vec![7, 10, 13], [7, 7], [8, 4, 14, 7])
            }
            Sam2Variant::HieraBasePlus => (
                112,
                2,
                [2, 3, 16, 3],
                vec![12, 16, 20],
                [14, 14],
                [8, 4, 14, 7],
            ),
            Sam2Variant::HieraLarge => (
                144,
                2,
                [2, 6, 36, 4],
                vec![23, 33, 43],
                [7, 7],
                [8, 4, 16, 8],
            ),
        };
        Self {
            variant,
            image_size: 1024,
            backbone_stride: 16,
            hidden_dim: 256,
            hierarchy: HieraConfig {
                embed_dim,
                num_heads,
                stages,
                global_att_blocks,
                window_pos_embed_bkg_spatial_size: pos_size,
                window_spec,
            },
            backbone_channel_list: [embed_dim * 8, embed_dim * 4, embed_dim * 2, embed_dim],
            fpn_top_down_levels: vec![2, 3],
            num_maskmem: 7,
            sigmoid_scale_for_mem_enc: 20.0,
            sigmoid_bias_for_mem_enc: -10.0,
        }
    }

    pub fn hiera_tiny() -> Self {
        Self::sam2_1(Sam2Variant::HieraTiny)
    }

    pub fn validate(&self) -> Result<()> {
        if self.image_size == 0 || self.image_size % self.backbone_stride != 0 {
            return Err(Error::InvalidConfig(
                "image_size must be a positive multiple of backbone_stride".into(),
            ));
        }
        if self.hidden_dim != 256 {
            return Err(Error::InvalidConfig(
                "published SAM2.1 heads require hidden_dim = 256".into(),
            ));
        }
        if self.hierarchy.stages.contains(&0) {
            return Err(Error::InvalidConfig(
                "every Hiera stage must contain at least one block".into(),
            ));
        }
        Ok(())
    }
}

impl Default for Sam2Config {
    fn default() -> Self {
        Self::hiera_tiny()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_published_sam2_1_hiera_variant_has_a_typed_valid_configuration() {
        for variant in Sam2Variant::ALL {
            let config = Sam2Config::sam2_1(variant);
            config.validate().unwrap();
            assert_eq!(
                config.backbone_channel_list[0],
                config.hierarchy.embed_dim * 8
            );
            assert!(config.hierarchy.window_spec.iter().all(|&size| size > 0));
        }
    }

    #[test]
    fn base_plus_and_large_keep_their_non_default_hiera_geometry() {
        assert_eq!(
            Sam2Config::sam2_1(Sam2Variant::HieraBasePlus)
                .hierarchy
                .window_pos_embed_bkg_spatial_size,
            [14, 14]
        );
        assert_eq!(
            Sam2Config::sam2_1(Sam2Variant::HieraLarge)
                .hierarchy
                .window_spec,
            [8, 4, 16, 8]
        );
    }
}
