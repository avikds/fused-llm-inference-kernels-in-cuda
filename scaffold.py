"""
Fused LLM Inference Kernels in CUDA scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

// scaffold.cu — smoke-test harness for fused LLM inference kernels.
// Student kernels/host fns are concatenated above; main only drives them.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static void fill(float* a, int n) {
    for (int i = 0; i < n; i++) a[i] = (float)(rand() % 100) / 50.0f - 1.0f;
}

int main() {
    srand(0);
    const int M = 4, H = 64, I = 128, cols = 16, V = 32, heads = 4, hd = 16;
    const int seq = 4, n = H;
    const float eps = 1e-5f;
    const int threads = 256;

    // Host buffers
    std::vector<float> h_x(M * H), h_res(M * H), h_w(H), h_b(H), h_out(M * H);
    std::vector<float> h_gate(M * I), h_up(M * I), h_scores(M * cols);
    std::vector<float> h_wgate(I * H), h_wup(I * H), h_wdown(H * I), h_bias(I);
    std::vector<float> h_emb(V * H), h_cos(seq * hd), h_sin(seq * hd);
    std::vector<float> h_q(seq * heads * hd), h_k(seq * heads * hd);
    std::vector<int>   h_ids(seq);
    fill(h_x.data(), M * H); fill(h_res.data(), M * H); fill(h_w.data(), H);
    fill(h_b.data(), H); fill(h_scores.data(), M * cols);
    fill(h_wgate.data(), I * H); fill(h_wup.data(), I * H);
    fill(h_wdown.data(), H * I); fill(h_bias.data(), I);
    fill(h_emb.data(), V * H); fill(h_cos.data(), seq * hd); fill(h_sin.data(), seq * hd);
    fill(h_q.data(), seq * heads * hd); fill(h_k.data(), seq * heads * hd);
    for (int i = 0; i < seq; i++) h_ids[i] = rand() % V;

    // Device buffers
    float *d_x, *d_res, *d_w, *d_b, *d_out, *d_rout, *d_gate, *d_up;
    float *d_scores, *d_sout, *d_wgate, *d_wup, *d_wdown, *d_bias;
    float *d_emb, *d_cos, *d_sin, *d_q, *d_k, *d_lin;
    int *d_ids;
    CUDA_CHECK(cudaMalloc(&d_x, M*H*4)); CUDA_CHECK(cudaMalloc(&d_res, M*H*4));
    CUDA_CHECK(cudaMalloc(&d_w, H*4));   CUDA_CHECK(cudaMalloc(&d_b, H*4));
    CUDA_CHECK(cudaMalloc(&d_out, M*H*4)); CUDA_CHECK(cudaMalloc(&d_rout, M*H*4));
    CUDA_CHECK(cudaMalloc(&d_gate, M*I*4)); CUDA_CHECK(cudaMalloc(&d_up, M*I*4));
    CUDA_CHECK(cudaMalloc(&d_scores, M*cols*4)); CUDA_CHECK(cudaMalloc(&d_sout, M*cols*4));
    CUDA_CHECK(cudaMalloc(&d_wgate, I*H*4)); CUDA_CHECK(cudaMalloc(&d_wup, I*H*4));
    CUDA_CHECK(cudaMalloc(&d_wdown, H*I*4)); CUDA_CHECK(cudaMalloc(&d_bias, I*4));
    CUDA_CHECK(cudaMalloc(&d_emb, V*H*4)); CUDA_CHECK(cudaMalloc(&d_cos, seq*hd*4));
    CUDA_CHECK(cudaMalloc(&d_sin, seq*hd*4)); CUDA_CHECK(cudaMalloc(&d_q, seq*heads*hd*4));
    CUDA_CHECK(cudaMalloc(&d_k, seq*heads*hd*4)); CUDA_CHECK(cudaMalloc(&d_lin, M*I*4));
    CUDA_CHECK(cudaMalloc(&d_ids, seq*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), M*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), M*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), M*cols*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wgate, h_wgate.data(), I*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wup, h_wup.data(), I*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wdown, h_wdown.data(), H*I*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), I*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb, h_emb.data(), V*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), seq*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), seq*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), seq*heads*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), seq*heads*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), seq*sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (M * H + threads - 1) / threads;
    add_residual_kernel<<<blocks, threads>>>(d_x, d_res, d_out, M * H);
    gelu_kernel<<<blocks, threads>>>(d_x, d_out, M * H);
    silu_kernel<<<blocks, threads>>>(d_x, d_out, M * H);
    // reuse gate/up as intermediate activations
    CUDA_CHECK(cudaMemcpy(d_gate, h_x.data(), M*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_res.data(), M*H*4, cudaMemcpyHostToDevice));
    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_out, M * H);

    rmsnorm_kernel<<<M, threads>>>(d_x, d_w, d_out, n, eps);
    layernorm_kernel<<<M, threads>>>(d_x, d_w, d_b, d_out, n, eps);
    fused_add_rmsnorm_kernel<<<M, threads>>>(d_x, d_res, d_w, d_out, d_rout, n, eps);

    softmax_row_kernel<<<M, threads>>>(d_scores, d_sout, M, cols);
    causal_softmax_kernel<<<M, threads>>>(d_scores, d_sout, M, cols);

    embedding_lookup_kernel<<<seq, threads>>>(d_ids, d_emb, d_out, seq, V, H);
    rope_kernel<<<seq * heads, threads>>>(d_q, d_k, d_cos, d_sin, seq, heads, hd);

    linear_kernel<<<(M*I+threads-1)/threads, threads>>>(d_x, d_wup, d_bias, d_lin, M, I, H);
    fused_linear_bias_gelu_kernel<<<(M*I+threads-1)/threads, threads>>>(
        d_x, d_wup, d_bias, d_lin, M, I, H);

    mlp_swiglu_forward(d_x, d_wgate, d_wup, d_wdown, d_out, M, H, I);
    rmsnorm_residual_block(d_x, d_res, d_w, d_out, d_rout, M, n, eps);
    run_transformer_ffn(d_x, d_res, d_w, d_wgate, d_wup, d_wdown, d_out, M, H, I, eps);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, M * H * sizeof(float), cudaMemcpyDeviceToHost));

    printf("FFN out[0..3]: %.6f %.6f %.6f %.6f\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);
    printf("FFN out[last]: %.6f\n", h_out[M * H - 1]);
    printf("scaffold OK\n");

    cudaFree(d_x); cudaFree(d_res); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_out); cudaFree(d_rout); cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_scores); cudaFree(d_sout); cudaFree(d_wgate); cudaFree(d_wup);
    cudaFree(d_wdown); cudaFree(d_bias); cudaFree(d_emb); cudaFree(d_cos);
    cudaFree(d_sin); cudaFree(d_q); cudaFree(d_k); cudaFree(d_lin); cudaFree(d_ids);
    return 0;
}
