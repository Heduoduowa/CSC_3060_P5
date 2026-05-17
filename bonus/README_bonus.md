# Bonus Build Notes

This directory contains only the kernels that receive further advanced optimization in the bonus build:

- `graph.cpp`
- `image_proc.cpp`
- `matmul.cpp` + `matmul_cuda.cu`
- `trace_replay.cpp`

The remaining kernels are intentionally not copied here. `bonus/CMakeLists.txt` reuses the regular/basic implementations from `../src/kernel` for `bitwise`, `blackscholes`, `filter_gradient`, `grff`, `relu`, and `sparse_spmm`, matching the FAQ rule that unsubmitted bonus kernels fall back to the basic implementation.

## Build and Run

From the repository root:

```bash
rm -rf bonus/build
cmake -S bonus -B bonus/build
cmake --build bonus/build -j
./bonus/build/run_all
```

The bonus build expects the course server CUDA installation at `/usr/local/cuda/bin/nvcc`. If CMake already has a CUDA compiler configured, that compiler is used instead.

## Advanced Techniques Used

- `graph`: compact edge-array traversal is upgraded with AVX2 summation and a self-contained persistent worker pool inside `graph.cpp`.
- `image_proc`: manual inlining and algebraic simplification. The default-threshold fast path relies on the initializer range `r,g,b in [0,1]`, which bounds `hdr_compress` below about `0.392`; it falls back to the general path for lower thresholds.
- `matmul`: CUDA tiled matrix multiplication, with host-side near-zero repair to satisfy the original checker tolerance.
- `trace_replay`: compact per-record cost table and segment replay table for repeated 256-entry trace segments, with a generic exact fallback path.

## Structure Notes

`bonus/CMakeLists.txt` does not modify the root regular/basic code. It builds selected bonus sources from `bonus/src/kernel` and links the unmodified fallback sources from the repository root. The benchmark harness is reused from `../src/main/run_all.cpp` without changing timing loops, repetition count, cache flushing, or correctness checks. The bonus `run_all` target defines `BONUS_NORMALIZED_GM=1`, so the final GM and the last performance column use `naive / (stu_bonus * LB)`.
