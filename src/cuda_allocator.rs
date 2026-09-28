//! Bounded CUDA memory-pool policy for inference workloads.

use candle_core::Device;

#[cfg(feature = "cuda")]
const DEFAULT_RETAIN_MIB: u64 = 256;

pub(crate) fn configure_cuda_allocator_for_inference(device: &Device) -> candle_core::Result<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = device {
        let retain_mib = match std::env::var("SAM2_CUDA_POOL_RETAIN_MIB") {
            Ok(value) => value.parse::<u64>().map_err(|_| {
                candle_core::Error::Msg(
                    "SAM2_CUDA_POOL_RETAIN_MIB must be a nonnegative integer".into(),
                )
            })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_RETAIN_MIB,
            Err(error) => {
                return Err(candle_core::Error::Msg(format!(
                    "invalid SAM2_CUDA_POOL_RETAIN_MIB: {error}"
                )));
            }
        };
        cuda.set_memory_pool_release_threshold(retain_mib.saturating_mul(1024 * 1024))?;
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(())
}
