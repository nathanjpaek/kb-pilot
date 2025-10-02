# ~14,000 tokens
# Layernorm + Attention
# https://github.com/theunnecessarythings/llm-ptx

PTX_WALKTHROUGH_PROMPT = """
Here's a walkthrough of translating a CUDA Layernorm kernel (using shared memory) to PTX. 

**Original CUDA Kernel:**

__global__ void layernorm_fwd_kernel(float *out, const float *inp,
                                   const float *weight, const float *bias,
                                   int C, float eps) {
  extern __shared__ float shared_buffer[];

  int bt = blockIdx.x;
  const float *x = inp + bt * C;
  float *y = out + bt * C;

  int tid = threadIdx.x;
  int block_size = blockDim.x;

  // --- Parallel Mean Calculation ---
  float sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sum += x[i];
  }
  shared_buffer[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_buffer[tid] += shared_buffer[tid + stride];
    }
    __syncthreads();
  }
  float mean = shared_buffer[0] / C;
  __syncthreads(); // Ensure mean is visible to all

  // --- Parallel Variance Calculation ---
  sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    float diff = x[i] - mean;
    sum += diff * diff;
  }
  shared_buffer[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_buffer[tid] += shared_buffer[tid + stride];
    }
    __syncthreads();
  }
  float var = shared_buffer[0] / C;
  float rstd = rsqrtf(var + eps);

  // --- Final Application ---
  for (int i = tid; i < C; i += block_size) {
    float n = (x[i] - mean) * rstd;
    y[i] = n * weight[i] + bias[i];
  }
}

void layernorm_forward(float *out, const float *inp, const float *weight,
                     const float *bias, int B, int T, int C,
                     cudaStream_t stream = 0) {
  const float eps = 1e-5f;
  int N = B * T;

  int blocks = N;
  // Threads must be a power of 2 for the reduction to work. 256 is good.
  int threads = 1024;

  // Shared memory for the reduction (one float per thread)
  size_t shared_mem_size = threads * sizeof(float);

  layernorm_fwd_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      out, inp, weight, bias, C, eps);

  cudaErrorCheck();
}

**Line by line breakdown:**

Kernel Declaration
CUDA C++:
__global__ void layernorm_fwd_kernel(float *out, const float *inp, 
                                     const float *weight, const float *bias, 
                                     int C, float eps)
{
PTX Assembly:
.visible .entry layernorm_fwd_kernel(
  .param .u64 out_param,
  .param .u64 inp_param,
  .param .u64 weight_param,
  .param .u64 bias_param,
  .param .s32 N_param,
  .param .s32 C_param
)
Explanation: The kernel header becomes a PTX entry point with six parameters. Pointers are passed as 64-bit (.u64) and the integers N (the number of rows, called bt in CUDA) and C are 32-bit signed (.s32).

Shared Memory Allocation
CUDA C++:
  extern __shared__ float shared_buffer[];
PTX Assembly:
.shared .align 4 .b8 %shared_sum_arr[128];
.shared .align 4 .b8 %shared_sum2_arr[128];
Explanation: The opening brace begins the body. Two statically-sized shared arrays are allocated for the running sums of x and x². 128 bytes is enough for 32 floats (32 * 4 = 128). The .align 4 directive ensures 4-byte alignment. extern shared requests dynamic shared memory, but we replaced it with two static 128-byte buffers (%shared_sum_arr, %shared_sum2_arr).

Calculate Block Index
CUDA C++:
  int bt = blockIdx.x;
PTX Assembly:
mov.s32 %idx, %ctaid.x;
Explanation: %ctaid.x is copied into %idx, giving the row ('batch-time') index bt.

Calculate Input Pointer
CUDA C++:
  const float *x = inp + bt * C;
PTX Assembly:
shl.b32   %C4,  %C, 2;                 // C*4 = bytes per row
mad.wide.s32 %x_ptr, %idx, %C4, %inp_ptr;  // x = inp + bt*C
Explanation: The element count C is converted to bytes (×4) then combined with bt to compute the base address %x_ptr for this row of x.

Calculate Output Pointer
CUDA C++:
  float *y = out + bt * C;
PTX Assembly:
mad.wide.s32 %out_ptr, %idx, %C4, %out_ptr;
Explanation: A similar mad.wide.s32 prepares the base pointer for the output row y.

Get Thread ID
CUDA C++:
  int tid = threadIdx.x;
PTX Assembly:
mov.s32 %tidx, %tid.x;
Explanation: %tid.x (thread-ID) is stored in %tidx.

Get Block Size
CUDA C++:
  int block_size = blockDim.x;
PTX Assembly:
mov.s32 %ntidx, %ntid.x;
Explanation: %ntid.x gives the number of threads per block (blockDim.x). It is saved as %ntidx.

Initialize Sum Accumulator
CUDA C++:
  float sum = 0.0f;
PTX Assembly:
mov.f32 %thread_sum, 0f00000000;
Explanation: Initialises the per-thread running sum register to +0.0.

Mean Computation Loop - Prologue
CUDA C++:
  for (int i = tid; i < C; i += block_size) {
PTX Assembly:
mov.s32 %i, %tidx;
bra $thread_local_cond;
Explanation: Loop prologue: the loop index %i starts at tid and jumps to a condition check label $thread_local_cond.

Mean Computation Loop - Accumulate
CUDA C++:
    sum += x[i];
PTX Assembly:
mad.wide.s32 %xi_ptr, %i, 4, %x_ptr;  // &x[i]
ld.global.f32 %xi, [%xi_ptr];
add.f32       %thread_sum, %thread_sum, %xi;
Explanation: Loads x[i] and accumulates it into %thread_sum.

Mean Computation Loop - Increment and Check
CUDA C++:
  }
PTX Assembly:
add.s32 %i, %i, %ntidx;
$thread_local_cond:
setp.lt.s32 %cond, %i, %C;
@%cond bra $thread_local_loop;
Explanation: Increments i by block_size, tests i < C, and branches back for the next loop iteration – exactly mirroring the CUDA for.

Store Partial Sum to Shared Memory
CUDA C++:
  shared_buffer[tid] = sum;
PTX Assembly:
// stores only warp-leader results:
setp.eq.s32 %cond, %lane_id, 0;
@!%cond bra $after_shared_write;
mad.lo.s32 %shared_sum_ptr, %warp_id, 4, %shared_sum;
st.shared.f32 [%shared_sum_ptr], %thread_sum;
$after_shared_write:
Explanation: Rather than every thread writing, the we perform an intra-warp reduction (using shfl.sync.down) and only the lane-0 thread of each warp writes the partial sum to shared memory.

Barrier After Mean Accumulation
CUDA C++:
  __syncthreads();
PTX Assembly:
bar.sync 0;
Explanation: A barrier so all partial sums are visible before the next stage.

Reduction Loop for Mean
CUDA C++:
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_buffer[tid] += shared_buffer[tid + stride];
    }
    __syncthreads();
  }
PTX Assembly:
// implemented in two levels:
  // 1) intra-warp shuffle (already done)
  // 2) warp-leaders read & reduce again (below)
Explanation: The high-level shared-memory reduction is translated into a second warp-level shfl.sync.down reduction across the warp-leader values that were written to shared memory. Predicate tests inside the CUDA loop disappear because the two-stage reduction uses warps and shuffles instead of an explicit if. This pairwise addition is replaced by shfl.sync.down operations. The compiler rewrites the reduction to avoid a barrier inside the loop.

Compute Mean
CUDA C++:
  float mean = shared_buffer[0] / C;
PTX Assembly:
shfl.sync.idx.b32 %block_sumf32, %warp_sum, 0, 0x1f, 0xffffffff;
cvt.rn.f32.s32 %Cf32, %C;
div.rn.f32     %m, %block_sumf32, %Cf32;
Explanation: The final block-wide sum is broadcast with shfl.sync.idx, converted to float, and divided by C to obtain mean (%m).

Barrier After Mean Computation
CUDA C++:
  __syncthreads(); // Ensure mean is visible to all
PTX Assembly:
bar.sync 0;
Explanation: Barrier so every thread has the computed mean before proceeding.

Initialize Variance Accumulator
CUDA C++:
  sum = 0.0f;
PTX Assembly:
mov.f32 %thread_sum2, 0f00000000;
Explanation: %thread_sum2 is reused to accumulate (x-mean)².

Variance Computation Loop - Prologue
CUDA C++:
  for (int i = tid; i < C; i += block_size) {
PTX Assembly:
mov.s32 %i, %tidx;
bra $thread_local_cond_var;
Explanation: Second thread loop prologue for the variance pass.

Variance Computation Loop - Calculate Difference
CUDA C++:
    float diff = x[i] - mean;
PTX Assembly:
mad.wide.s32 %xi_ptr, %i, 4, %x_ptr;
ld.global.f32 %xi, [%xi_ptr];
sub.f32       %n, %xi, %m;
Explanation: Loads x[i] and subtracts the previously computed mean, producing diff (%n).

Variance Computation Loop - Accumulate Squared Difference
CUDA C++:
    sum += diff * diff;
PTX Assembly:
mul.f32       %n, %n, %n;
add.f32       %thread_sum2, %thread_sum2, %n;
Explanation: Squares the difference and accumulates into %thread_sum2.

Variance Computation Loop - Increment and Check
CUDA C++:
  }
PTX Assembly:
add.s32 %i, %i, %ntidx;
$thread_local_cond_var:
setp.lt.s32 %cond, %i, %C;
@%cond bra $thread_local_loop;
Explanation: Loop increment, test, and branch exactly like the mean loop.

Store Variance Partial Sum to Shared Memory
CUDA C++:
  shared_buffer[tid] = sum;
PTX Assembly:
setp.eq.s32 %cond, %lane_id, 0;
@!%cond bra $after_shared_write2;
mad.lo.s32 %shared_sum2_ptr, %warp_id, 4, %shared_sum2;
st.shared.f32 [%shared_sum2_ptr], %thread_sum2;
$after_shared_write2:
Explanation: Warp-leaders write their (diff²) sums to the second shared buffer.

Barrier Before Variance Reduction
CUDA C++:
  __syncthreads();
PTX Assembly:
bar.sync 0;
Explanation: Barrier before the second reduction stage.

Reduction Loop for Variance
CUDA C++:
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_buffer[tid] += shared_buffer[tid + stride];
    }
    __syncthreads();
  }
PTX Assembly:
// translated into a second warp-shuffle reduction (see PTX around $warp_reduce_loop2)
Explanation: As with the mean, the loop is lowered to warp-level shuffles + one barrier. Removed. Similar to the mean reduction. Replaced by the shuffle arithmetic visible in $warp_reduce_loop2. Removed; the warp-shuffle pattern needs only one barrier afterwards.

Compute Variance
CUDA C++:
  float var = shared_buffer[0] / C;
PTX Assembly:
shfl.sync.idx.b32 %block_sum2f32, %warp_sum2, 0, 0x1f, 0xffffffff;
div.rn.f32     %block_sum2f32, %block_sum2f32, %Cf32;
sub.f32         %var, %block_sum2f32, %m2;
Explanation: The block-wide sum of squares is divided by C, minus mean², giving the variance %var.

Compute Reciprocal Standard Deviation
CUDA C++:
  float rstd = rsqrtf(var + eps);
PTX Assembly:
add.f32        %var, %var, 0f3727C5AC;   // eps     (0x3727C5AC ≈ 1e-5)
rsqrt.approx.f32 %s,  %var;
Explanation: Adds the epsilon and computes the reciprocal square-root (rsqrt.approx.f32) to get rstd (%s).

Normalization Loop - Prologue
CUDA C++:
  for (int i = tid; i < C; i += block_size) {
PTX Assembly:
mov.s32 %i, %tidx;
bra $normalize_cond;
Explanation: Third per-thread loop prologue: iterate over channels for output.

Normalization Loop - Compute Normalized Value
CUDA C++:
    float n = (x[i] - mean) * rstd;
PTX Assembly:
mad.wide.s32 %xi_ptr, %i, 4, %x_ptr;
ld.global.cs.f32 %xi, [%xi_ptr];
sub.f32           %n,  %xi, %m;
mul.f32           %n,  %n,  %s;
Explanation: Loads x[i], subtracts the mean, multiplies by rstd – computing the normalised value n.

Normalization Loop - Apply Scale and Bias, Store Result
CUDA C++:
    y[i] = n * weight[i] + bias[i];
PTX Assembly:
mad.wide.s32 %weight_ptr_i, %i, 4, %weight_ptr;
ld.global.nc.f32  %weight_val, [%weight_ptr_i];
mad.wide.s32 %bias_ptr_i,   %i, 4, %bias_ptr;
ld.global.nc.f32  %bias_val,  [%bias_ptr_i];
fma.rn.f32        %n, %n, %weight_val, %bias_val;
mad.wide.s32 %out_ptr_i, %i, 4, %out_ptr;
st.global.cs.f32  [%out_ptr_i], %n;
Explanation: Fetches the scale (weight[i]) and bias (bias[i]), performs n*weight + bias with a fused‐multiply-add, and stores the result to y[i].

Normalization Loop - Increment and Check
CUDA C++:
  }
PTX Assembly:
add.s32 %i, %i, %ntidx;
$normalize_cond:
setp.lt.s32 %cond, %i, %C;
@%cond bra $normalize_loop;
Explanation: Loop increment, condition test, and branch to iterate until i ≥ C.

Kernel Exit
CUDA C++:
}
PTX Assembly:
ret;
Explanation: End of kernel: ret terminates the thread.

**Final PTX Kernel:**

.version 8.7
.target sm_80
.address_size 64


.visible .entry layernorm_fwd_kernel(
  .param .u64 out_param,
  .param .u64 inp_param,
  .param .u64 weight_param,
  .param .u64 bias_param,
  .param .s32 N_param,
  .param .s32 C_param
)
{
    .shared .align 4 .b8 %shared_sum_arr[128];   
    .shared .align 4 .b8 %shared_sum2_arr[128];  
    // Register Declarations
    .reg .pred %cond;
    .reg .f32 %thread_sum, %thread_sum2, %warp_sum, %warp_sum2,
            %block_sumf32, %block_sum2f32, %xi, %n, %m, %m2, %var, %s, %weight_val, 
            %bias_val, %shuffled_bits_f32, %Cf32;
    .reg .b32 %i, %idx, %warp_id, %lane_id, %offset, %N, %num_warps, %C4,
            %shuffled_bits, %C, %ntidx, %tidx, %block_sum, %block_sum2, %shared_sum,
            %shared_sum2, %shared_sum_ptr, %shared_sum2_ptr;
    .reg .b64 %out_ptr, %inp_ptr, %weight_ptr, %bias_ptr, %xi_ptr, 
            %out_ptr_i, %weight_ptr_i, %bias_ptr_i, %x_ptr;


    // Load params
    ld.param.u64 %out_ptr, [out_param];
    ld.param.u64 %inp_ptr, [inp_param];
    ld.param.u64 %weight_ptr, [weight_param];
    ld.param.u64 %bias_ptr, [bias_param];
    ld.param.s32 %N, [N_param];
    ld.param.s32 %C, [C_param];
    cvta.to.global.u64 %out_ptr, %out_ptr;
    cvta.to.global.u64 %inp_ptr, %inp_ptr;
    cvta.to.global.u64 %weight_ptr, %weight_ptr;
    cvta.to.global.u64 %bias_ptr, %bias_ptr;

    mov.s32 %shared_sum, %shared_sum_arr;
    mov.s32 %shared_sum2, %shared_sum2_arr;
    mov.s32 %ntidx, %ntid.x;
    mov.s32 %tidx, %tid.x;
    shr.b32 %num_warps, %ntidx, 5;
    shr.b32 %warp_id, %tidx, 5;
    rem.s32 %lane_id, %tidx, 32;
    mov.s32 %idx, %ctaid.x;
    shl.b32 %C4, %C, 2;
    mad.wide.s32 %x_ptr, %idx, %C4, %inp_ptr;


    // Guard
    setp.ge.s32 %cond, %idx, %N;
    @%cond ret;

    // Thread-local summation
    mov.f32 %thread_sum, 0f00000000;
    mov.f32 %thread_sum2, 0f00000000;

    mov.s32 %i, %tid.x;
    bra $thread_local_cond;
    $thread_local_loop:
        mad.wide.s32 %xi_ptr, %i, 4, %x_ptr;
        ld.global.f32 %xi, [%xi_ptr];
        add.f32 %thread_sum, %thread_sum, %xi;
        fma.rn.f32 %thread_sum2, %xi, %xi, %thread_sum2;
        add.s32 %i, %i, %ntidx;
    $thread_local_cond:
        setp.lt.s32 %cond, %i, %C;
        @%cond bra $thread_local_loop;

    // Warp-level reduction
    mov.s32 %offset, 16;
    bra $warp_reduce_cond;
    $warp_reduce_loop:
       shfl.sync.down.b32 %shuffled_bits_f32, %thread_sum, %offset, 0x1f, 0xffffffff;
       add.f32 %thread_sum, %thread_sum, %shuffled_bits_f32;
       shfl.sync.down.b32 %shuffled_bits_f32, %thread_sum2, %offset, 0x1f, 0xffffffff;
       add.f32 %thread_sum2, %thread_sum2, %shuffled_bits_f32;
       shr.s32 %offset, %offset, 1;
    $warp_reduce_cond:
        setp.gt.s32 %cond, %offset, 0;
        @%cond bra $warp_reduce_loop;

    // lane 0 - write partial sum to shared memory
    setp.eq.s32 %cond, %lane_id, 0;
    @!%cond bra $after_shared_write;
    mad.lo.s32 %shared_sum_ptr, %warp_id, 4, %shared_sum;
    mad.lo.s32 %shared_sum2_ptr, %warp_id, 4, %shared_sum2;
    st.shared.f32 [%shared_sum_ptr], %thread_sum;
    st.shared.f32 [%shared_sum2_ptr], %thread_sum2;

    $after_shared_write:
    bar.sync 0;

    // Each warp reads its partial sum
    mov.f32 %warp_sum, 0f00000000;
    mov.f32 %warp_sum2, 0f00000000;
    setp.lt.s32 %cond, %lane_id, %num_warps;
    @!%cond bra $after_warp_read; 
    mad.lo.s32 %shared_sum_ptr, %lane_id, 4, %shared_sum;
    mad.lo.s32 %shared_sum2_ptr, %lane_id, 4, %shared_sum2;
    ld.shared.f32 %warp_sum, [%shared_sum_ptr];
    ld.shared.f32 %warp_sum2, [%shared_sum2_ptr];

    $after_warp_read:
    // 2nd warp-level reduction
    mov.s32 %offset, 16;
    bra $warp_reduce_cond2;
    $warp_reduce_loop2:
       shfl.sync.down.b32 %shuffled_bits_f32, %warp_sum, %offset, 0x1f, 0xffffffff;
       add.f32 %warp_sum, %warp_sum, %shuffled_bits_f32;
       shfl.sync.down.b32 %shuffled_bits_f32, %warp_sum2, %offset, 0x1f, 0xffffffff;
       add.f32 %warp_sum2, %warp_sum2, %shuffled_bits_f32;
       shr.s32 %offset, %offset, 1;
    $warp_reduce_cond2:
        setp.gt.s32 %cond, %offset, 0;
        @%cond bra $warp_reduce_loop2;

    // Broadcast the final sum to all threads
    shfl.sync.idx.b32 %block_sumf32, %warp_sum, 0, 0x1f, 0xffffffff;
    shfl.sync.idx.b32 %block_sum2f32, %warp_sum2, 0, 0x1f, 0xffffffff;

    cvt.rn.f32.s32 %Cf32, %C;
    div.rn.f32 %m, %block_sumf32, %Cf32;
    div.rn.f32 %block_sum2f32, %block_sum2f32, %Cf32;
    mul.f32 %m2, %m, %m;
    sub.f32 %var, %block_sum2f32, %m2;
    add.f32 %var, %var, 0f3727C5AC;
    rsqrt.approx.f32 %s, %var;

    // Normalize 
    mad.wide.s32 %out_ptr, %idx, %C4, %out_ptr;
    mov.s32 %i, %tid.x;
    bra $normalize_cond;
    $normalize_loop:
        mad.wide.s32 %xi_ptr, %i, 4, %x_ptr;
        ld.global.cs.f32 %xi, [%xi_ptr];
        sub.f32 %n, %xi, %m;
        mul.f32 %n, %n, %s;
        mad.wide.s32 %weight_ptr_i, %i, 4, %weight_ptr;
        ld.global.nc.f32 %weight_val, [%weight_ptr_i];
        mad.wide.s32 %bias_ptr_i, %i, 4, %bias_ptr;
        ld.global.nc.f32 %bias_val, [%bias_ptr_i];
        fma.rn.f32 %n, %n, %weight_val, %bias_val;
        mad.wide.s32 %out_ptr_i, %i, 4, %out_ptr;
        st.global.cs.f32 [%out_ptr_i], %n;
        add.s32 %i, %i, %ntidx;
    $normalize_cond:
        setp.lt.s32 %cond, %i, %C;
        @%cond bra $normalize_loop;
    ret;
}

Here's a walkthrough of translating a CUDA Attention kernel to PTX. 

**Original CUDA Kernel:**

__global__ void attention_fwd_kernel(float *out, float *preatt, float *att,
                                   const float *inp, int B, int T, int C,
                                   int NH) {
  // Each thread block is for one head and one batch item: grid(NH, B)
  // Each thread is for one query token: block(T)
  int h = blockIdx.x;
  int b = blockIdx.y;
  int t = threadIdx.x;

  if (b >= B || h >= NH || t >= T)
    return;

  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0f / sqrtf((float)hs);

  // Pointer to the input for this batch item
  const float *inp_b = inp + b * T * C3;
  // Pointer to the query vector for this thread (b, t, h)
  const float *query_t = inp_b + t * C3 + h * hs;

  // Pointers to the output attention scores for this thread's row
  float *preatt_bth = preatt + (b * NH * T * T) + (h * T * T) + (t * T);
  float *att_bth = att + (b * NH * T * T) + (h * T * T) + (t * T);

  // --- Pass 1: Calculate Q.K^T and find maxval (causal attention) ---
  float maxval = -10000.0f;
  for (int t2 = 0; t2 <= t; t2++) {
    const float *key_t2 = inp_b + t2 * C3 + h * hs + C; // +C offset for key
    float val = 0.0f;
    for (int i = 0; i < hs; i++) {
      val += query_t[i] * key_t2[i];
    }
    val *= scale;
    if (val > maxval) {
      maxval = val;
    }
    preatt_bth[t2] = val;
  }

  // --- Pass 2: Calculate exponentials and sum for the softmax denominator
  float expsum = 0.0f;
  for (int t2 = 0; t2 <= t; t2++) {
    float expv = expf(preatt_bth[t2] - maxval);
    expsum += expv;
    att_bth[t2] = expv; // Store the numerator temporarily
  }
  float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

  // --- Pass 3: Normalize to get final softmax scores ---
  for (int t2 = 0; t2 <= t; t2++) {
    att_bth[t2] *= expsum_inv;
  }
  for (int t2 = t + 1; t2 < T; t2++) {
    att_bth[t2] = 0.0f;
  }

  // --- Pass 4: Accumulate weighted values into the output ---
  float *out_bth = out + (b * T * C) + (t * C) + (h * hs);
  for (int i = 0; i < hs; i++) {
    out_bth[i] = 0.0f;
  }
  for (int t2 = 0; t2 <= t; t2++) {
    const float *value_t2 =
        inp_b + t2 * C3 + h * hs + C * 2; // +2C offset for value
    float att_score = att_bth[t2];
    for (int i = 0; i < hs; i++) {
      out_bth[i] += att_score * value_t2[i];
    }
  }
}

void attention_forward(float *out, float *preatt, float *att, const float *inp,
                     int B, int T, int C, int NH, cudaStream_t stream = 0) {
  dim3 grid(NH, B);
  dim3 block(T);
  attention_fwd_kernel<<<grid, block, 0, stream>>>(out, preatt, att, inp, B, T,
                                                   C, NH);
  cudaErrorCheck();
}

**Line by line breakdown:**

Kernel Declaration
CUDA C++:
__global__ void attention_fwd_kernel(...) {
PTX Assembly:
.visible .entry attention_fwd_kernel(
  .param .u64 out_param, ...
) {
Explanation: The kernel's entry point. The global keyword in CUDA C++ corresponds to a .visible .entry function in PTX. Each argument passed to the kernel is explicitly declared as a .param with its corresponding type (.u64 for pointers, .u32 for integers).

Register Declarations
CUDA C++:
// C++ variable declarations
PTX Assembly:
.reg .pred %cond;
.reg .b32 %B, %T, ...;
.reg .b64 %att_bth_ptr, ...;
.reg .f32 %q_val, ...;
Explanation: In PTX, all registers used within a function must be declared at the beginning. This section corresponds to all the local variable declarations in C++ (e.g., int h, float maxval, etc.). Registers are typed, such as .pred for predicates, .b32 for 32-bit integers, .b64 for 64-bit addresses, and .f32 for single-precision floats.

Load Kernel Parameters
CUDA C++:
// Loading kernel parameters into registers
PTX Assembly:
ld.param.u64 %out_ptr, [out_param];
ld.param.u64 %preattn_ptr, [preattn_param];
ld.param.u64 %attn_ptr, [attn_param];
ld.param.u64 %inp_ptr, [inp_param];
ld.param.u32 %B, [B_param];
ld.param.u32 %T, [T_param];
ld.param.u32 %C, [C_param];
ld.param.u32 %NH, [NH_param];
Explanation: The first step inside the kernel is to move the parameters from the special parameter memory space into general-purpose registers. The ld.param (load parameter) instruction performs this for each argument.

Convert Pointers to Global Address Space
CUDA C++:
// Converting pointers to global address space
PTX Assembly:
cvta.to.global.u64 %out_ptr, %out_ptr;
cvta.to.global.u64 %preattn_ptr, %preattn_ptr;
cvta.to.global.u64 %attn_ptr, %attn_ptr;
cvta.to.global.u64 %inp_ptr, %inp_ptr;
Explanation: The cvta.to.global.u64 (convert address) instruction translates the loaded pointer from a parameter-specific address to a generic global memory address that can be used by load and store instructions like ld.global.

Get Head Index
CUDA C++:
  int h = blockIdx.x;
PTX Assembly:
mov.u32 %h, %ctaid.x;
Explanation: The head index h is assigned by reading the block's X-dimension ID from the special register %ctaid.x and moving it into the %h register.

Get Batch Index
CUDA C++:
  int b = blockIdx.y;
PTX Assembly:
mov.u32 %b, %ctaid.y;
Explanation: The batch index b is assigned by reading the block's Y-dimension ID from the special register %ctaid.y into the %b register.

Get Token Index
CUDA C++:
  int t = threadIdx.x;
PTX Assembly:
mov.u32 %t, %tid.x;
Explanation: The token index t (representing the current query token) is assigned by reading the thread's X-dimension ID from the special register %tid.x into the %t register.

Bounds Checking
CUDA C++:
  if (b >= B || h >= NH || t >= T)
    return;
PTX Assembly:
setp.ge.u32 %cond, %t, %T;
@%cond bra $exit;
setp.ge.u32 %cond, %h, %NH;
@%cond bra $exit;
setp.ge.u32 %cond, %b, %B;
@%cond bra $exit;
Explanation: These are guard clauses to prevent out-of-bounds execution. Each check uses setp.ge.u32 (set predicate if greater or equal) to compare an index with its bound. If the condition is true, the @%cond bra $exit instruction performs a branch, immediately exiting the kernel for that thread.

Calculate Combined QKV Size
CUDA C++:
  int C3 = C * 3;
PTX Assembly:
mul.lo.u32 %C3, %C, 3;
Explanation: Calculates C*3, representing the combined size of a Query, Key, and Value vector, storing it in the %C3 register.

Calculate Head Size
CUDA C++:
  int hs = C / NH; // head size
PTX Assembly:
div.u32 %hs, %C, %NH;
Explanation: Calculates the dimension of each attention head (hs) by dividing the total channels %C by the number of heads %NH.

Compute Attention Scale Factor
CUDA C++:
  float scale = 1.0f / sqrtf((float)hs);
PTX Assembly:
cvt.rn.f32.u32 %hs_f32, %hs;
sqrt.rn.f32 %hs_sqrt, %hs_f32;
rcp.rn.f32 %scale, %hs_sqrt;
Explanation: Computes the attention scaling factor 1/sqrt(hs). This is implemented in three steps: 1) cvt converts the integer %hs to a float. 2) sqrt.rn.f32 computes the square root. 3) rcp.rn.f32 computes the reciprocal, which is a fast way to perform division.

Calculate Batch Input Pointer
CUDA C++:
  const float *inp_b = inp + b * T * C3;
PTX Assembly:
mul.lo.u32 %C3_x4, %C3, 4;
mul.lo.u32 %bT, %b, %T;
mad.wide.u32 %inp_b_ptr, %bT, %C3_x4, %inp_ptr;
Explanation: Calculates the base pointer for the current batch. It multiplies C3 by 4 to get a byte stride, computes the element offset bT, and then uses mad.wide.u32 to calculate the final address: (bT)(C34) + inp_ptr.

Calculate Query Vector Pointer
CUDA C++:
  const float *query_t = inp_b + t * C3 + h * hs;
PTX Assembly:
mul.lo.u32 %hs_x4, %hs, 4;
mad.wide.u32 %query_t_ptr, %t, %C3_x4, %inp_b_ptr;
mad.wide.u32 %query_t_ptr, %h, %hs_x4, %query_t_ptr;
Explanation: Calculates the pointer to this thread's query vector. First, it adds the token offset (t * C3 * 4) to the batch pointer. Then, it adds the head offset (h * hs * 4) to that intermediate result.

Calculate Pre-Attention Score Pointer
CUDA C++:
  float *preatt_bth = preatt + (b * NH * T * T) + (h * T * T) + (t * T);
PTX Assembly:
mad.lo.u32 %b_NH_h, %b, %NH, %h;
mul.lo.u32 %TT, %T, %T;
mul.lo.u32 %b_NH_TT, %b_NH_h, %TT;
mad.lo.u32 %bth_offset, %t, %T, %b_NH_TT;
mad.wide.u32 %preatt_bth_ptr, %bth_offset, 4, %preattn_ptr;
Explanation: Calculates the pointer to the start of the current thread's row in the preatt score matrix. The complex element offset is built up in stages for clarity and then converted to a byte offset (* 4) and added to the base pointer.

Calculate Attention Score Pointer
CUDA C++:
  float *att_bth = att + (b * NH * T * T) + (h * T * T) + (t * T);
PTX Assembly:
mad.wide.u32 %att_bth_ptr, %bth_offset, 4, %attn_ptr;
Explanation: Calculates the pointer for the final att matrix row, reusing the %bth_offset calculated previously and adding it to the base attn_ptr.

Initialize Maximum Value
CUDA C++:
  float maxval = -10000.0f;
PTX Assembly:
mov.f32 %maxval, 0fc61c4000;
Explanation: Initializes a register to hold the maximum attention score found so far, using the hexadecimal representation of -10000.0f.

Pass 1 Loop - Initialize
CUDA C++:
  for (int t2 = 0; ...
PTX Assembly:
mov.u32 %t2, 0;
$pass1_loop:
Explanation: This is the initialization part of the for loop, setting the loop counter %t2 to 0. This PTX label marks the beginning of the loop for Pass 1.

Pass 1 Loop - Calculate Key Pointer
CUDA C++:
    const float *key_t2 = inp_b + t2 * C3 + h * hs + C;
PTX Assembly:
mad.lo.u32 %hhsC, %h, %hs, %C;
mad.lo.u32 %offset, %t2, %C3, %hhsC;
mad.wide.u32 %key_t2_ptr, %offset, 4, %inp_b_ptr;
Explanation: Calculates the pointer to the key vector for token t2. The + C offset (to select the Key part of the QKV projection) is handled in the first mad instruction.

Pass 1 Loop - Initialize Dot Product Accumulator
CUDA C++:
    float val = 0.0f;
PTX Assembly:
mov.f32 %val, 0f00000000;
Explanation: Initializes an accumulator register for the dot product to zero.

Pass 1 Loop - Dot Product Inner Loop
CUDA C++:
    for (int i = 0; i < hs; i++) {
      val += query_t[i] * key_t2[i];
    }
PTX Assembly:
mov.u32 %i, 0;
$dot_product_loop: ...

mad.wide.u32 %query_ti_ptr, %i, 4, %query_t_ptr;
mad.wide.u32 %key_t2i_ptr, %i, 4, %key_t2_ptr;
ld.global.f32 %q_val, [%query_ti_ptr];
ld.global.f32 %k_val, [%key_t2i_ptr];
fma.rn.f32 %val, %q_val, %k_val, %val;

add.u32 %i, %i, 1;
setp.lt.u32 %cond, %i, %hs;
@%cond bra $dot_product_loop;
Explanation: This inner dot product loop is implemented with explicit PTX instructions for initialization (mov), the loop label ($dot_product_loop), and a conditional branch at the end. It is not unrolled because its bound, hs, is a variable. The core of the dot product. It calculates addresses for the i-th element of the query and key vectors, loads them from global memory, and uses fma.rn.f32 (fused multiply-add) to multiply them and add to the running sum in %val. These instructions handle the dot product loop's control flow: incrementing i, checking if i < hs, and branching back if true.

Pass 1 Loop - Apply Scale
CUDA C++:
    val *= scale;
PTX Assembly:
mul.f32 %val, %val, %scale;
Explanation: Applies the scaling factor to the completed dot product score.

Pass 1 Loop - Update Maximum
CUDA C++:
    if (val > maxval) { maxval = val; }
PTX Assembly:
setp.gt.f32 %cond, %val, %maxval;
@%cond mov.f32 %maxval, %val;
Explanation: Updates the maximum value. setp.gt.f32 sets a predicate if val > maxval, and the predicated mov.f32 executes the update only if true.

Pass 1 Loop - Store Pre-Attention Score
CUDA C++:
    preatt_bth[t2] = val;
PTX Assembly:
mad.wide.u32 %preatt_bthi_ptr, %t2, 4, %preatt_bth_ptr;
st.global.f32 [%preatt_bthi_ptr], %val;
Explanation: Stores the raw, scaled attention score into the preatt matrix at column t2.

Pass 1 Loop - Control Flow
CUDA C++:
  // End of Pass 1 loop
PTX Assembly:
add.u32 %t2, %t2, 1;
setp.le.u32 %cond, %t2, %t;
@%cond bra $pass1_loop;
Explanation: The control flow for the outer loop of Pass 1: increments t2, checks if t2 <= t, and branches back to $pass1_loop if true.

Pass 2 - Initialize Exponential Sum
CUDA C++:
  float expsum = 0.0f;
PTX Assembly:
mov.f32 %expsum, 0f00000000;
Explanation: Initializes the accumulator for the softmax denominator (expsum) to zero.

Pass 2 Loop - Initialize
CUDA C++:
  // Start of Pass 2 loop
PTX Assembly:
mov.u32 %t2, 0;
$pass2_loop:
Explanation: Resets the %t2 counter to 0 and defines the starting label for the Pass 2 loop.

Pass 2 Loop - Compute Exponential
CUDA C++:
    float expv = expf(preatt_bth[t2] - maxval);
PTX Assembly:
mad.wide.u32 %preatt_bthi_ptr, %t2, 4, %preatt_bth_ptr;
ld.global.f32 %preatt_val, [%preatt_bthi_ptr];
sub.f32 %preatt_val, %preatt_val, %maxval;
mul.f32 %preatt_val, %preatt_val, 0f3fb8aa3b;
ex2.approx.ftz.f32 %expv, %preatt_val;
Explanation: Calculates exp(score - maxval). It loads the score from preatt, subtracts maxval for stability, multiplies by log2(e) (0f3fb8aa3b), and finally computes the base-2 exponent with ex2.approx.ftz.f32.

Pass 2 Loop - Accumulate Exponential Sum
CUDA C++:
    expsum += expv;
PTX Assembly:
add.f32 %expsum, %expsum, %expv;
Explanation: Adds the newly calculated exponential value to the running sum in %expsum.

Pass 2 Loop - Store Unnormalized Value
CUDA C++:
    att_bth[t2] = expv;
PTX Assembly:
mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
st.global.f32 [%att_bthi_ptr], %expv;
Explanation: Stores the temporary, un-normalized numerator value into the final att matrix.

Pass 2 Loop - Control Flow
CUDA C++:
  // End of Pass 2 loop
PTX Assembly:
add.u32 %t2, %t2, 1;
setp.le.u32 %cond, %t2, %t;
@%cond bra $pass2_loop;
Explanation: The control flow for the Pass 2 loop, which increments and checks the t2 counter.

Compute Inverse of Exponential Sum
CUDA C++:
  float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
PTX Assembly:
mov.f32 %expsum_inv, 0f00000000;
setp.eq.f32 %cond, %expsum, 0f00000000;
@!%cond rcp.rn.f32 %expsum_inv, %expsum;
Explanation: Safely calculates 1.0/expsum. It initializes the result to 0.0. A predicate checks if expsum is zero. If it is not zero (@!%cond), the rcp.rn.f32 instruction computes the reciprocal.

Pass 3 Loop - Initialize Normalization
CUDA C++:
  for (int t2 = 0; t2 <= t; t2++) {
PTX Assembly:
mov.u32 %t2, 0;
$pass3_loop:
Explanation: Initializes the Pass 3 loop, which will normalize the attention scores.

Pass 3 Loop - Normalize Attention Scores
CUDA C++:
    att_bth[t2] *= expsum_inv;
PTX Assembly:
mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
ld.global.f32 %att_val, [%att_bthi_ptr];
mul.f32 %att_val, %att_val, %expsum_inv;
st.global.f32 [%att_bthi_ptr], %att_val;
Explanation: This is a read-modify-write operation. It loads the numerator from att, multiplies it by expsum_inv, and stores the final normalized score back.

Pass 3 Loop - Control Flow
CUDA C++:
  // End of Pass 3 normalization loop
PTX Assembly:
add.u32 %t2, %t2, 1;
setp.le.u32 %cond, %t2, %t;
@%cond bra $pass3_loop;
Explanation: Control flow for the normalization loop.

Zero Out Future Tokens - Initialize
CUDA C++:
  for (int t2 = t + 1; t2 < T; t2++) {
PTX Assembly:
add.u32 %t2, %t, 1;
bra $zero_out_check;
$zero_out_loop:
Explanation: Initializes the loop to zero out future tokens for causality. It starts t2 at t+1.

Zero Out Future Tokens - Store Zero
CUDA C++:
    att_bth[t2] = 0.0f;
PTX Assembly:
mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
st.global.f32 [%att_bthi_ptr], 0f00000000;
Explanation: Stores the value 0.0f into the att matrix for a future token position.

Zero Out Future Tokens - Control Flow
CUDA C++:
  // End of zeroing loop
PTX Assembly:
add.u32 %t2, %t2, 1;
$zero_out_check:
setp.lt.u32 %cond, %t2, %T;
@%cond bra $zero_out_loop;
Explanation: Control flow for the zeroing loop, which continues as long as t2 < T.

Calculate Output Pointer
CUDA C++:
  float *out_bth = out + (b * T * C) + (t * C) + (h * hs);
PTX Assembly:
mul.lo.u32 %tC, %t, %C;
mad.lo.u32 %bth_offset, %bT, %C, %tC;
mad.lo.u32 %bth_offset, %h, %hs, %bth_offset;
mad.wide.u32 %out_bth_ptr, %bth_offset, 4, %out_ptr;
Explanation: Calculates the final output pointer for this thread's vector slot.

Initialize Output Vector to Zero
CUDA C++:
  for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
PTX Assembly:
mov.u32 %i, 0;
$init_zero_loop: ... st.global.f32 [%out_bthi_ptr], 0f00000000; ...
Explanation: This loop initializes the output vector in global memory to all zeros before starting accumulation.

Accumulation Loop - Initialize
CUDA C++:
  for (int t2 = 0; t2 <= t; t2++) {
PTX Assembly:
mov.u32 %t2, 0;
$accumulate_loop:
Explanation: Initializes the final accumulation loop, which aggregates the weighted value vectors.

Accumulation Loop - Calculate Value Pointer
CUDA C++:
    const float *value_t2 = inp_b + t2 * C3 + h * hs + C * 2;
PTX Assembly:
shl.b32 %C2, %C, 1;
mad.lo.u32 %offset, %h, %hs, %C2;
mad.lo.u32 %value_t2_offset, %t2, %C3, %offset;
mad.wide.u32 %value_t2_ptr, %value_t2_offset, 4, %inp_b_ptr;
Explanation: Calculates the pointer to the value vector for token t2. The + C*2 offset is implemented with shl.b32 (shift left by 1), which is a fast way to multiply by 2.

Accumulation Loop - Load Attention Score
CUDA C++:
    float att_score = att_bth[t2];
PTX Assembly:
mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
ld.global.f32 %att_val_f32, [%att_bthi_ptr];
Explanation: Loads the final, normalized attention score for the current t2 token.

Accumulation Loop - Inner Loop Initialize
CUDA C++:
    for (int i = 0; i < hs; i++) {
PTX Assembly:
mov.u32 %i, 0;
$accumulate_inner_loop:
Explanation: Initializes the inner loop for aggregating the value vector components.

Accumulation Loop - Weighted Sum
CUDA C++:
      out_bth[i] += att_score * value_t2[i];
PTX Assembly:
ld.global.f32 %value_f32, [%value_t2i_ptr];
ld.global.f32 %out_val_f32, [%out_bthi_ptr];
fma.rn.f32 %out_val_f32, %att_val_f32, %value_f32, %out_val_f32;
st.global.f32 [%out_bthi_ptr], %out_val_f32;
Explanation: The final accumulation step. It loads the value component, loads the current out component, performs the weighted sum att_score * value + out using fma, and stores the new result back.

Accumulation Loop - Inner Loop Control Flow
CUDA C++:
    // End of inner accumulation loop
PTX Assembly:
add.u32 %i, %i, 1;
setp.lt.u32 %cond, %i, %hs;
@%cond bra $accumulate_inner_loop;
Explanation: Control flow for the inner accumulation loop.

Accumulation Loop - Outer Loop Control Flow
CUDA C++:
  // End of outer accumulation loop
PTX Assembly:
add.u32 %t2, %t2, 1;
setp.le.u32 %cond, %t2, %t;
@%cond bra $accumulate_loop;
Explanation: Control flow for the outer accumulation loop.

Kernel Exit
CUDA C++:
}
PTX Assembly:
$exit:
ret;
Explanation: The common exit point for the kernel. The ret instruction ends execution for the thread.

**Final PTX Kernel:**

.version 8.7
.target sm_80
.address_size 64

.visible .entry attention_fwd_kernel(
    .param .u64 out_param,
    .param .u64 preattn_param,
    .param .u64 attn_param,
    .param .u64 inp_param,
    .param .u32 B_param,
    .param .u32 T_param,
    .param .u32 C_param,
    .param .u32 NH_param
)
{
  // Register Declarations
  .reg .pred %cond;
  .reg .b32 %B, %T, %bT, %TT, %tC, %C, %NH, %h, %b, %t, %hs, %hhsC, %C3, %C2,
            %hs_x4, %C3_x4, %t2, %i, %bth_offset, %b_NH_h, %b_NH_TT, %bth_t2_offset, 
            %offset, %value_t2_offset;
  .reg .b64 %att_bth_ptr, %preatt_bth_ptr, %out_bth_ptr,
            %out_ptr, %preattn_ptr, %attn_ptr, %inp_ptr, %inp_b_ptr,
            %query_t_ptr, %key_t2_ptr, %query_ti_ptr,
            %key_t2i_ptr, %preatt_bthi_ptr, %att_bthi_ptr,
            %out_bthi_ptr, %value_t2i_ptr, %value_t2_ptr, %C_x4;
  .reg .f32 %q_val, %k_val, %expsum_inv_f32, %hs_f32,
            %scale_f32, %hs_sqrt_f32, %preatt_val_f32, %att_val_f32,
            %expv_f32, %val_f32, %out_val_f32, %hs_sqrt, %scalee, %expsum_inv,
            %expsum, %maxval, %value, %value_f32, %out_val, %att_val, %preatt_val,
            %expv, %val, %scale;

  // Load parameters
  ld.param.u64 %out_ptr, [out_param];
  ld.param.u64 %preattn_ptr, [preattn_param];
  ld.param.u64 %attn_ptr, [attn_param];
  ld.param.u64 %inp_ptr, [inp_param];
  ld.param.u32 %B, [B_param];
  ld.param.u32 %T, [T_param];
  ld.param.u32 %C, [C_param];
  ld.param.u32 %NH, [NH_param];
  cvta.to.global.u64 %out_ptr, %out_ptr;
  cvta.to.global.u64 %preattn_ptr, %preattn_ptr;
  cvta.to.global.u64 %attn_ptr, %attn_ptr;
  cvta.to.global.u64 %inp_ptr, %inp_ptr;

  // Index and Guarding
  mov.u32 %h, %ctaid.x;
  mov.u32 %b, %ctaid.y;
  mov.u32 %t, %tid.x;
  
  // Guarding is automatically handled by the grid size and block size
  setp.ge.u32  %cond, %t,  %T;
  @%cond bra   $exit;
  setp.ge.u32  %cond, %h,  %NH;
  @%cond bra   $exit;
  setp.ge.u32  %cond, %b,  %B;
  @%cond bra   $exit;

  // calculate scale
  mul.lo.u32 %C3, %C, 3;
  div.u32 %hs, %C, %NH;
  cvt.rn.f32.u32 %hs_f32, %hs;
  sqrt.rn.f32 %hs_sqrt, %hs_f32;
  rcp.rn.f32 %scale, %hs_sqrt;

  // load pointers
  mul.lo.u32 %C3_x4, %C3, 4;
  mul.lo.u32 %hs_x4, %hs, 4;
  mul.lo.u32 %bT, %b, %T;
  mad.wide.u32 %inp_b_ptr, %bT, %C3_x4, %inp_ptr;
  mad.wide.u32 %query_t_ptr, %t, %C3_x4, %inp_b_ptr;
  mad.wide.u32 %query_t_ptr, %h, %hs_x4, %query_t_ptr;
  mad.lo.u32 %b_NH_h, %b, %NH, %h;
  mul.lo.u32 %TT, %T, %T;
  mul.lo.u32 %b_NH_TT, %b_NH_h, %TT;
  mad.lo.u32 %bth_offset, %t, %T, %b_NH_TT;
  mad.wide.u32 %preatt_bth_ptr, %bth_offset, 4, %preattn_ptr;
  mad.wide.u32 %att_bth_ptr, %bth_offset, 4, %attn_ptr;

  // Pass 1: Calculate Q.K^T and find maxval (causal attention) 
  // Each thread finds its own maxval, no block reduction needed
  mov.f32 %maxval, 0fc61c4000; // -10000.0f
  mov.u32 %t2, 0;
  $pass1_loop:
    mad.lo.u32 %hhsC, %h, %hs, %C;
    mad.lo.u32 %offset, %t2, %C3, %hhsC;
    mad.wide.u32 %key_t2_ptr, %offset, 4, %inp_b_ptr;

    // Dot product
    mov.f32 %val, 0f00000000; // 0.0f
    mov.u32 %i, 0;
    $dot_product_loop:
      mad.wide.u32 %query_ti_ptr, %i, 4, %query_t_ptr;
      mad.wide.u32 %key_t2i_ptr, %i, 4, %key_t2_ptr;
      ld.global.f32 %q_val, [%query_ti_ptr];
      ld.global.f32 %k_val, [%key_t2i_ptr];
      fma.rn.f32 %val, %q_val, %k_val, %val;
      add.u32 %i, %i, 1;
      setp.lt.u32 %cond, %i, %hs;
      @%cond bra $dot_product_loop;

    mul.f32 %val, %val, %scale;
    setp.gt.f32 %cond, %val, %maxval;
    @%cond mov.f32 %maxval, %val;
    mad.wide.u32 %preatt_bthi_ptr, %t2, 4, %preatt_bth_ptr;
    st.global.f32 [%preatt_bthi_ptr], %val;
    add.u32 %t2, %t2, 1;
    setp.le.u32 %cond, %t2, %t;
    @%cond bra $pass1_loop;

  // Pass 2: Calculate exponentials and sum for the softmax denominator
  // Each thread calculates its own sum, no block reduction.
  mov.f32 %expsum, 0f00000000; // 0.0f
  mov.u32 %t2, 0;
  $pass2_loop:
    mad.wide.u32 %preatt_bthi_ptr, %t2, 4, %preatt_bth_ptr;
    ld.global.f32 %preatt_val, [%preatt_bthi_ptr];
    sub.f32 %preatt_val, %preatt_val, %maxval; 
    mul.f32 %preatt_val, %preatt_val, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %expv, %preatt_val; 
    add.f32 %expsum, %expsum, %expv;
    mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
    st.global.f32 [%att_bthi_ptr], %expv;
    add.u32 %t2, %t2, 1;
    setp.le.u32 %cond, %t2, %t;
    @%cond bra $pass2_loop;

  // calculatte expsum_inv
  mov.f32 %expsum_inv, 0f00000000; // 0.0f
  setp.eq.f32 %cond, %expsum, 0f00000000; // avoid division by zero
  @!%cond rcp.rn.f32 %expsum_inv, %expsum;

  // Pass 3: Normalize to get final softmax scores
  mov.u32 %t2, 0;
  $pass3_loop:
    mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
    ld.global.f32 %att_val, [%att_bthi_ptr];
    mul.f32 %att_val, %att_val, %expsum_inv;
    st.global.f32 [%att_bthi_ptr], %att_val; 
    add.u32 %t2, %t2, 1;
    setp.le.u32 %cond, %t2, %t;
    @%cond bra $pass3_loop;

    // Explicitly zero out future tokens
    add.u32 %t2, %t, 1;
      bra $zero_out_check;
    $zero_out_loop:
      mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
      st.global.f32 [%att_bthi_ptr], 0f00000000; // 0.0f
      add.u32 %t2, %t2, 1;
    $zero_out_check:
      setp.lt.u32 %cond, %t2, %T;
      @%cond bra $zero_out_loop;

  // Pass 4: Accumulate weighted values into the output
  mul.lo.u32 %tC, %t, %C;
  mad.lo.u32 %bth_offset, %bT, %C, %tC;
  mad.lo.u32 %bth_offset, %h, %hs, %bth_offset;
  mad.wide.u32 %out_bth_ptr, %bth_offset, 4, %out_ptr;

  // Initialize the output vector to zeros
  mov.u32 %i, 0;
  $init_zero_loop:
    mad.wide.u32 %out_bthi_ptr, %i, 4, %out_bth_ptr;
    st.global.f32 [%out_bthi_ptr], 0f00000000; // 0.0f
    add.u32 %i, %i, 1;
    setp.lt.u32 %cond, %i, %hs;
    @%cond bra $init_zero_loop;

  mov.u32 %t2, 0;
  shl.b32 %C2, %C, 1;
  mad.lo.u32 %offset, %h, %hs, %C2;

  $accumulate_loop:
    mad.lo.u32 %value_t2_offset, %t2, %C3, %offset;
    mad.wide.u32 %value_t2_ptr, %value_t2_offset, 4, %inp_b_ptr;
    mad.wide.u32 %att_bthi_ptr, %t2, 4, %att_bth_ptr;
    ld.global.f32 %att_val_f32, [%att_bthi_ptr];
    mov.u32 %i, 0;
  $accumulate_inner_loop:
    mad.wide.u32 %value_t2i_ptr, %i, 4, %value_t2_ptr;
    mad.wide.u32 %out_bthi_ptr, %i, 4, %out_bth_ptr;
    ld.global.f32 %value_f32, [%value_t2i_ptr];
    ld.global.f32 %out_val_f32, [%out_bthi_ptr];
    fma.rn.f32 %out_val_f32, %att_val_f32, %value_f32, %out_val_f32;
    st.global.f32 [%out_bthi_ptr], %out_val_f32;

    // Inner loop increment and branch
    add.u32 %i, %i, 1;
    setp.lt.u32 %cond, %i, %hs;
    @%cond bra $accumulate_inner_loop;

  // Outer loop increment and branch
  add.u32 %t2, %t2, 1;
  setp.le.u32 %cond, %t2, %t;
  @%cond bra $accumulate_loop;

  $exit:
  ret;
}
"""