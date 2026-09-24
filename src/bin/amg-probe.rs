//! Exercises automatic mask generation including mask-to-mask refinement.

use std::{env, path::PathBuf, process::ExitCode};

use sam2_rs::{AutomaticMaskConfig, Sam2AutomaticMaskGenerator};

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: amg-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let result = (|| -> candle_core::Result<()> {
        let generator = Sam2AutomaticMaskGenerator::load_sam2_1_hiera_tiny(
            path,
            AutomaticMaskConfig {
                points_per_side: Some(1),
                pred_iou_thresh: 0.,
                stability_score_thresh: 0.,
                use_m2m: true,
                ..Default::default()
            },
        )?;
        let masks = generator.generate_rgb8(&vec![0; 32 * 32 * 3], 32, 32)?;
        println!("automatic refined proposals: {}", masks.len());
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
