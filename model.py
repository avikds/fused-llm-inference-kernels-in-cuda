"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // Reduce the value across all 32 lanes of the warp.
    // After the loop, every lane contains the warp-wide sum.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Broadcast the final sum from lane 0 to every lane.
    val = __shfl_sync(0xFFFFFFFF, val, 0);

    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // Reduce the maximum across all 32 lanes of the warp.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    // Broadcast the maximum from lane 0 to every lane.
    val = __shfl_sync(0xFFFFFFFF, val, 0);

    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // First, reduce values within each warp.
    val = warp_reduce_sum(val);

    // Lane 0 of each warp writes its partial sum to shared memory.
    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    // Have the first warp reduce the per-warp partial sums.
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);

        // The complete block sum is returned only by thread 0.
        if (lane == 0) {
            shared[0] = val;
        }
    }

    __syncthreads();

    return (threadIdx.x == 0) ? shared[0] : 0.0f;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // First, reduce values within each warp.
    val = warp_reduce_max(val);

    // Lane 0 of each warp stores its partial maximum.
    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    // The first warp reduces the per-warp maxima.
    if (warp_id == 0) {
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            // Use a valid value for inactive lanes.
            // shared[0] is already available after __syncthreads().
            val = shared[0];
        }

        val = warp_reduce_max(val);

        // Only thread 0 needs to return the final result.
        if (lane == 0) {
            shared[0] = val;
        }
    }

    __syncthreads();

    return (threadIdx.x == 0) ? shared[0] : 0.0f;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i] + residual[i];
    }
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float val = x[i];

        // Standard tanh approximation of GELU:
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        float x3 = val * val * val;
        float inner = sqrt_2_over_pi * (val + coeff * x3);

        out[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float val = x[i];
        out[i] = val / (1.0f + expf(-val));
    }
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float g = gate[i];

        // SiLU(g) = g * sigmoid(g)
        float silu = g / (1.0f + expf(-g));

        out[i] = silu * up[i];
    }
}

# Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(const float* x, const float* weight,
                               float* out, int n, float eps) {
    // One block processes one row.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + row * n;
    float* row_out = out + row * n;

    // Accumulate the sum of squares for this row.
    float sum_sq = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float v = row_x[i];
        sum_sq += v * v;
    }

    // Warp-level reduction of the sum of squares.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Store one partial sum per warp.
    __shared__ float warp_sums[32];

    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }

    __syncthreads();

    // First warp reduces the warp-level partial sums.
    if (warp_id == 0) {
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }

        if (lane == 0) {
            warp_sums[0] = sum_sq;
        }
    }

    __syncthreads();

    // RMS = sqrt(mean(x^2) + eps)
    float rms = sqrtf(warp_sums[0] / static_cast<float>(n) + eps);

    // Normalize and scale by the learned weight.
    for (int i = tid; i < n; i += blockDim.x) {
        row_out[i] = (row_x[i] / rms) * weight[i];
    }
}

# Step 10 - layernorm_kernel
__global__ void layernorm_kernel(const float* x, const float* weight,
                                 const float* bias, float* out,
                                 int n, float eps) {
    // One block processes one row.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + row * n;
    float* row_out = out + row * n;

    // Shared memory: one float per possible warp.
    __shared__ float shared[32];

    // Compute the local sum for this thread.
    float local_sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += row_x[i];
    }

    // Reduce the sum across the entire block.
    float block_sum = block_reduce_sum(local_sum, shared);

    // Broadcast the mean to all threads.
    if (tid == 0) {
        shared[0] = block_sum / static_cast<float>(n);
    }

    __syncthreads();

    float mean = shared[0];

    // Compute the local sum of squared deviations.
    float local_var = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float diff = row_x[i] - mean;
        local_var += diff * diff;
    }

    // Reduce the variance sum across the block.
    float block_var = block_reduce_sum(local_var, shared);

    // Compute the standard deviation on thread 0.
    if (tid == 0) {
        shared[0] = block_var / static_cast<float>(n);
    }

    __syncthreads();

    float variance = shared[0];
    float inv_std = rsqrtf(variance + eps);

    // Normalize, scale, and shift.
    for (int i = tid; i < n; i += blockDim.x) {
        float normalized = (row_x[i] - mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

# Step 11 - fused_add_rmsnorm_kernel
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + row * n;
    const float* row_residual = residual + row * n;
    float* row_out = out + row * n;
    float* row_residual_out = residual_out + row * n;

    // One float per warp for the block reduction.
    __shared__ float shared[32];

    // Step 1: Fused residual addition and accumulation of squares.
    float local_sum_sq = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float value = row_x[i] + row_residual[i];

        // Write residual_out = x + residual.
        row_residual_out[i] = value;

        // Accumulate squared values for RMSNorm.
        local_sum_sq += value * value;
    }

    // Step 2: Reduce the sum of squares across the block.
    float sum_sq = block_reduce_sum(local_sum_sq, shared);

    // Thread 0 computes the inverse RMS and stores it in shared memory.
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(n);
        shared[0] = rsqrtf(mean_sq + eps);
    }

    __syncthreads();

    float inv_rms = shared[0];

    // Step 3: RMSNorm and learned per-feature scaling.
    for (int i = tid; i < n; i += blockDim.x) {
        row_out[i] = row_residual_out[i] * inv_rms * weight[i];
    }
}

# Step 12 - softmax_row_kernel (not yet solved)
# TODO: implement

# Step 13 - causal_softmax_kernel (not yet solved)
# TODO: implement

# Step 14 - embedding_lookup_kernel (not yet solved)
# TODO: implement

# Step 15 - rope_kernel (not yet solved)
# TODO: implement

# Step 16 - linear_kernel (not yet solved)
# TODO: implement

# Step 17 - fused_linear_bias_gelu_kernel (not yet solved)
# TODO: implement

# Step 18 - mlp_swiglu_forward (not yet solved)
# TODO: implement

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

