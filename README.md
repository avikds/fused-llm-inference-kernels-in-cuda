# Fused LLM Inference Kernels in CUDA

Implement high-performance CUDA kernels for LLM inference, from warp/block reductions and activations through fused RMSNorm, Softmax, RoPE, and SwiGLU MLP blocks. Learn the GPU primitives that make modern transformer inference efficient.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** warp_reduce_sum
- [x] **2.** warp_reduce_max
- [x] **3.** block_reduce_sum
- [ ] **4.** block_reduce_max
- [ ] **5.** add_residual_kernel
- [ ] **6.** gelu_kernel
- [ ] **7.** silu_kernel
- [ ] **8.** swiglu_kernel
- [ ] **9.** rmsnorm_kernel
- [ ] **10.** layernorm_kernel
- [ ] **11.** fused_add_rmsnorm_kernel
- [ ] **12.** softmax_row_kernel
- [ ] **13.** causal_softmax_kernel
- [ ] **14.** embedding_lookup_kernel
- [ ] **15.** rope_kernel
- [ ] **16.** linear_kernel
- [ ] **17.** fused_linear_bias_gelu_kernel
- [ ] **18.** mlp_swiglu_forward
- [ ] **19.** rmsnorm_residual_block
- [ ] **20.** run_transformer_ffn

---

Built on Deep-ML.
