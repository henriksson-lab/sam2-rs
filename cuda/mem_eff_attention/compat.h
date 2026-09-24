/*
 * Minimal inference-only compatibility definitions for the BSD-licensed
 * PyTorch memory-efficient attention kernel. SAM2 never enables dropout in
 * this path, but the upstream parameter ABI retains the Philox state fields.
 */
#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

#define TORCH_CHECK(CONDITION, ...) assert(CONDITION)

namespace at {
struct PhiloxCudaState {
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  PhiloxCudaState() = default;
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }

  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

namespace cuda::philox {
__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(
    PhiloxCudaState state) {
  if (state.captured_) {
    return std::make_tuple(
        static_cast<uint64_t>(*state.seed_.ptr),
        static_cast<uint64_t>(*state.offset_.ptr + state.offset_intragraph_));
  }
  return std::make_tuple(state.seed_.val, state.offset_.val);
}
}  // namespace cuda::philox
}  // namespace at
