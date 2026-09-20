//! Keep the large, short-lived CPU attention buffers on the glibc heap.
//!
//! glibc otherwise maps and unmaps them on each inference, which is expensive
//! for Hiera's early QKV projections. This is process-global allocator tuning;
//! callers can opt out with `SAM2_CPU_KEEP_LARGE_ALLOCATIONS=0` or supply
//! their own `MALLOC_MMAP_THRESHOLD_` / `MALLOC_TRIM_THRESHOLD_` settings.

pub(crate) fn configure_cpu_allocator_for_inference() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        use std::sync::Once;

        if std::env::var_os("SAM2_CPU_KEEP_LARGE_ALLOCATIONS").is_some_and(|value| value == "0")
            || std::env::var_os("MALLOC_MMAP_THRESHOLD_").is_some()
            || std::env::var_os("MALLOC_TRIM_THRESHOLD_").is_some()
        {
            return;
        }

        static CONFIGURE: Once = Once::new();
        CONFIGURE.call_once(|| {
            unsafe extern "C" {
                fn mallopt(param: std::ffi::c_int, value: std::ffi::c_int) -> std::ffi::c_int;
            }

            const M_TRIM_THRESHOLD: std::ffi::c_int = -1;
            const M_MMAP_THRESHOLD: std::ffi::c_int = -3;
            const LARGE_BUFFER_THRESHOLD: std::ffi::c_int = 256 * 1024 * 1024;

            // SAFETY: These are glibc mallopt constants and a positive value
            // within c_int's range. Once serializes the process-wide change.
            unsafe {
                mallopt(M_MMAP_THRESHOLD, LARGE_BUFFER_THRESHOLD);
                mallopt(M_TRIM_THRESHOLD, LARGE_BUFFER_THRESHOLD);
            }
        });
    }
}
