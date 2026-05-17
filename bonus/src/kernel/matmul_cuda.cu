#include <cuda_runtime.h>

#include <cstddef>

namespace matmul_cuda {

namespace {

constexpr int kTile = 16;

__global__ void matmul_kernel(float* __restrict__ C,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += kTile) {
        const int a_col = tile + threadIdx.x;
        const int b_row = tile + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = __fadd_rn(sum, __fmul_rn(tile_a[threadIdx.y][k],
                                           tile_b[k][threadIdx.x]));
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

struct DeviceBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::size_t capacity = 0;
    bool disabled = false;

    ~DeviceBuffers() {
        if (a) {
            cudaFree(a);
        }
        if (b) {
            cudaFree(b);
        }
        if (c) {
            cudaFree(c);
        }
    }
};

DeviceBuffers& buffers() {
    static DeviceBuffers bufs;
    return bufs;
}

bool ensure_capacity(DeviceBuffers& bufs, std::size_t count) {
    if (bufs.disabled) {
        return false;
    }

    if (bufs.capacity >= count) {
        return true;
    }

    if (bufs.a) {
        cudaFree(bufs.a);
    }
    if (bufs.b) {
        cudaFree(bufs.b);
    }
    if (bufs.c) {
        cudaFree(bufs.c);
    }

    bufs = DeviceBuffers{};

    const std::size_t bytes = count * sizeof(float);
    if (cudaMalloc(&bufs.a, bytes) != cudaSuccess ||
        cudaMalloc(&bufs.b, bytes) != cudaSuccess ||
        cudaMalloc(&bufs.c, bytes) != cudaSuccess) {
        bufs.disabled = true;
        return false;
    }

    bufs.capacity = count;
    return true;
}

}  // namespace

bool run(float* C, const float* A, const float* B, int n) {
    if (n <= 0 || !C || !A || !B) {
        return false;
    }

    DeviceBuffers& bufs = buffers();
    const std::size_t count = static_cast<std::size_t>(n) *
                              static_cast<std::size_t>(n);
    const std::size_t bytes = count * sizeof(float);

    if (!ensure_capacity(bufs, count)) {
        return false;
    }

    if (cudaMemcpy(bufs.a, A, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(bufs.b, B, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        bufs.disabled = true;
        return false;
    }

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    matmul_kernel<<<grid, block>>>(bufs.c, bufs.a, bufs.b, n);

    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(C, bufs.c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        bufs.disabled = true;
        return false;
    }

    return true;
}

}  // namespace matmul_cuda
