PLAN_2_12 = """

Objective: Create `ModelNew(nn.Module)` with a TileLang kernel for `fused_linear_mul_leaky_relu` (replaces `nn.Linear` -> scalar multiply -> `LeakyReLU`). Output MUST numerically match PyTorch `nn.Linear` + ops.

**1. `ModelNew` PyTorch Wrapper Essentials:**
    *   **`__init__`:**
        *   Store `in_features, out_features, multiplier, negative_slope`.
        *   Create `nn.Parameter` for `self.weight (out, in)` and `self.bias (out)`.
        *   **CRITICAL for Correctness:** Initialize `self.weight` using `torch.nn.init.kaiming_uniform_(..., a=5**0.5)` and `self.bias` via fan-in based `torch.nn.init.uniform_`. These match `nn.Linear` defaults.
        *   Cache compiled kernels by `batch_size`.
    *   **`forward(self, x)`:**
        *   Convert `x`, `self.weight`, `self.bias` to `cuda` and kernel `dtype` (e.g., `float16`).
        *   Get/compile kernel for current `batch_size`, passing `multiplier`, `negative_slope`.
        *   Call kernel: `result = kernel(x, self.weight, self.bias)`.
        *   Return `result.to(torch.float32)`.

**2. TileLang Kernel (`fused_linear_mul_leaky_relu`) Essentials:**
    *   **Signature:** Kernel name `fused_linear_mul_leaky_relu`.
        *   Args: `x (batch,in,dtype)`, `weight (out,in,dtype)`, `bias (out,dtype)`. Output: `out (batch,out,dtype)`.
        *   Compile-time args: `batch_size, in_features, out_features, multiplier_val, negative_slope_val`.
        *   Internal: `block_M,N,K`; `dtype="float16"`, `accum_dtype="float32"`.
        *   Use `@tilelang.jit(out_idx=-1)` and `@T.prim_func`.
    *   **Logic (`T.Kernel` grid `ceil(out/N_blk), ceil(batch/M_blk)`):**
        *   Shared mem: `x_s (M,K)`, `w_s (N,K)`. Frag: `out_loc (M,N,accum_dtype)`.
        *   `T.clear(out_loc)`.
        *   **GEMM (`for ko` over `in_feat/K_blk`):**
            *   `T.copy` global `x` tile to `x_s`.
            *   `T.copy` global `weight` tile to `w_s`. (Loads `W` tile directly)
            *   `T.gemm(x_s, w_s, out_loc, transpose_B=True)`. (Crucial: `transpose_B=True` to compute `x_s @ w_s.T`).
        *   **Post-GEMM Fused Ops (`T.Parallel` over `M_blk, N_blk`):**
            *   Boundary check global indices.
            *   Compute: `val = out_loc[i,j] + bias[global_col_idx]`, then `val *= multiplier_val`, then `val = T.max(val, negative_slope_val * val)` (for LeakyReLU).
            *   Store `val` to global `output`.

Key for correctness: 1) `nn.Linear`-identical weight/bias init in `ModelNew`. 2) `transpose_B=True` in kernel `T.gemm` due to direct `W` tile loading.


"""

PLAN_3_9 = """

CUDA IMPLEMENTATION OF RESNET:

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace py = pybind11;

// Warp-level reduction using __shfl_down_sync for float values
__inline__ __device__ float warpReduceSum(float val) {
// Full mask
for (int offset = warpSize/2; offset > 0; offset /= 2) {
val += __shfl_down_sync(0xffffffff, val, offset);
}
return val;
}

//======================================================================
// Adaptive Average Pooling Kernel for 1x1 output using warp-level reduction
// Input: tensor of shape [N, C, H, W] in NCHW format
// Output: tensor of shape [N, C] where each element is the average over H*W
// Each block computes the output for one (n, c) pair
//======================================================================
__global__ void adaptive_avg_pool_1x1_kernel(const float* __restrict__ input,
float* __restrict__ output,
int N, int C, int H, int W) {
// Each block corresponds to one (n, c) pair
int nc = blockIdx.x; // block index ranges over N * C
int n = nc / C;
int c = nc % C;

// Pointer to the beginning of the (n, c) feature map
const float* in_ptr = input + n * C * H * W + c * H * W;
float sum = 0.0f;
int HW = H * W;

// Each thread sums over a strided portion of the H*W elements
for (int i = threadIdx.x; i < HW; i += blockDim.x) {
sum += in_ptr[i];
}

// Perform warp-level reduction using __shfl_down_sync
sum = warpReduceSum(sum);

// The first thread in the warp writes the average
if (threadIdx.x == 0) {
output[nc] = sum / static_cast<float>(HW);
}
}

// Host wrapper for adaptive average pooling kernel
torch::Tensor adaptive_avg_pool_1x1(torch::Tensor x) {
TORCH_CHECK(x.dim() == 4, "Input to adaptive_avg_pool_1x1 must be a 4D tensor (N, C, H, W)");
int N = x.size(0);
int C = x.size(1);
int H = x.size(2);
int W = x.size(3);

auto output = torch::empty({N, C}, x.options());
int total = N * C;
int threads = 32; // One warp per block
const dim3 blocks(total);

adaptive_avg_pool_1x1_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
x.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);

return output;
}

//======================================================================
// Fused Residual Addition and ReLU Kernel
// Performs out = max(x + identity, 0) element-wise
//======================================================================
__global__ void fused_add_relu_kernel(const float* __restrict__ x,
const float* __restrict__ identity,
float* __restrict__ out,
int numel) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < numel) {
float sum = x[idx] + identity[idx];
out[idx] = (sum > 0.f) ? sum : 0.f;
}
}

// Host wrapper for fused addition + ReLU
torch::Tensor fused_add_relu(torch::Tensor x, torch::Tensor identity) {
auto out = torch::empty_like(x);
int numel = x.numel();
int threads = 256;
int blocks = (numel + threads - 1) / threads;
fused_add_relu_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
x.data_ptr<float>(), identity.data_ptr<float>(), out.data_ptr<float>(), numel);
return out;
}

//======================================================================
// Basic Residual Block for ResNet18
// Consists of two conv->batch_norm layers and uses fused residual addition and ReLU
//======================================================================
torch::Tensor basic_block_fn(
torch::Tensor x,
torch::Tensor conv1_w,
torch::Tensor bn1_w,
torch::Tensor bn1_b,
torch::Tensor bn1_rm,
torch::Tensor bn1_rv,
torch::Tensor conv2_w,
torch::Tensor bn2_w,
torch::Tensor bn2_b,
torch::Tensor bn2_rm,
torch::Tensor bn2_rv,
torch::Tensor downsample_conv_w,
torch::Tensor downsample_bn_w,
torch::Tensor downsample_bn_b,
torch::Tensor downsample_bn_rm,
torch::Tensor downsample_bn_rv,
int64_t stride,
bool is_training
) {
torch::Tensor identity = x;

// First convolution, batch norm and ReLU
x = torch::conv2d(x, conv1_w, /*bias=*/{}, {stride, stride}, {1, 1});
x = torch::batch_norm(x, bn1_w, bn1_b, bn1_rm, bn1_rv, is_training, 0.0, 1e-5, true);
x = torch::relu(x);

// Second convolution and batch norm
x = torch::conv2d(x, conv2_w, /*bias=*/{}, {1, 1}, {1, 1});
x = torch::batch_norm(x, bn2_w, bn2_b, bn2_rm, bn2_rv, is_training, 0.0, 1e-5, true);

// Downsample identity if necessary
if (downsample_conv_w.defined()) {
identity = torch::conv2d(identity, downsample_conv_w, /*bias=*/{}, {stride, stride});
identity = torch::batch_norm(identity, downsample_bn_w, downsample_bn_b,
downsample_bn_rm, downsample_bn_rv,
is_training, 0.0, 1e-5, true);
}

// Fused addition of residual and ReLU activation
x = fused_add_relu(x, identity);
return x;
}

//======================================================================
// ResNet18 Forward Function using custom warp-level adaptive average pooling
//======================================================================
torch::Tensor module_fn(torch::Tensor x, py::object params_py, bool is_training) {
auto get_param = [&](const std::string & key) -> torch::Tensor {
return params_py.attr("__getitem__")(key.c_str()).cast<torch::Tensor>();
};

// Initial layers: Conv -> BatchNorm -> ReLU -> MaxPool
auto conv1_weight = get_param("conv1_weight");
auto bn1_weight = get_param("bn1_weight");
auto bn1_bias = get_param("bn1_bias");
auto bn1_running_mean = get_param("bn1_running_mean");
auto bn1_running_var = get_param("bn1_running_var");

x = torch::conv2d(x, conv1_weight, /*bias=*/{}, {2, 2}, {3, 3});
x = torch::batch_norm(x, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
is_training, 0.0, 1e-5, true);
x = torch::relu(x);
x = torch::max_pool2d(x, {3, 3}, {2, 2}, {1, 1});

// Process each of the 4 layers with 2 blocks each
for (int i = 1; i <= 4; ++i) {
std::string layer_name = "layer" + std::to_string(i);
for (int j = 0; j < 2; ++j) {
std::string block_name = layer_name + "_" + std::to_string(j);
int64_t stride = (i > 1 && j == 0) ? 2 : 1;

auto conv1_w = get_param(block_name + "_conv1_weight");
auto bn1_w = get_param(block_name + "_bn1_weight");
auto bn1_b = get_param(block_name + "_bn1_bias");
auto bn1_rm = get_param(block_name + "_bn1_running_mean");
auto bn1_rv = get_param(block_name + "_bn1_running_var");

auto conv2_w = get_param(block_name + "_conv2_weight");
auto bn2_w = get_param(block_name + "_bn2_weight");
auto bn2_b = get_param(block_name + "_bn2_bias");
auto bn2_rm = get_param(block_name + "_bn2_running_mean");
auto bn2_rv = get_param(block_name + "_bn2_running_var");

std::string downsample_conv_key = block_name + "_downsample_0_weight";
bool has_downsample = PyMapping_HasKeyString(params_py.ptr(), downsample_conv_key.c_str()) == 1;
torch::Tensor downsample_conv_w, downsample_bn_w, downsample_bn_b,
downsample_bn_rm, downsample_bn_rv;
if (has_downsample) {
downsample_conv_w = get_param(block_name + "_downsample_0_weight");
downsample_bn_w = get_param(block_name + "_downsample_1_weight");
downsample_bn_b = get_param(block_name + "_downsample_1_bias");
downsample_bn_rm = get_param(block_name + "_downsample_1_running_mean");
downsample_bn_rv = get_param(block_name + "_downsample_1_running_var");
}

x = basic_block_fn(x,
conv1_w,
bn1_w,
bn1_b,
bn1_rm,
bn1_rv,
conv2_w,
bn2_w,
bn2_b,
bn2_rm,
bn2_rv,
downsample_conv_w,
downsample_bn_w,
downsample_bn_b,
downsample_bn_rm,
downsample_bn_rv,
stride, is_training);
}
}

// Replace torch::adaptive_avg_pool2d with custom adaptive average pooling using warp-level primitives
x = adaptive_avg_pool_1x1(x);

// Flatten and apply the fully-connected layer
x = x.view({x.size(0), -1});
auto fc_weight = get_param("fc_weight");
auto fc_bias = get_param("fc_bias");
x = torch::linear(x, fc_weight, fc_bias);
return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &module_fn, "ResNet18 forward function with warp-level adaptive avg pooling (CUDA)");
}

"""

PLAN_3_2 = """

CUDA IMPLEMENTATION OF SHALLOW MLP THAT SOLVES THIS TASK:

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants for block sizes
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define RELU_BLOCK_SIZE 256

// Combined MLP forward kernel
// Uses warp-level primitives for efficient dot-product reduction
// and optimized block dimensions for better occupancy

template <typename scalar_t>
__global__ void efficient_mlp_forward_kernel(
const scalar_t* __restrict__ input,
const scalar_t* __restrict__ weight,
const scalar_t* __restrict__ bias,
scalar_t* __restrict__ output,
const int batch_size,
const int in_features,
const int out_features) {

// Each warp calculates one output neuron
int lane = threadIdx.x; // lane index within a warp [0, 31]
int warpId = threadIdx.y; // warp index within the block

int row = blockIdx.x; // one block per batch row
int col = blockIdx.y * BLOCK_DIM_Y + warpId; // each warp calculates one output neuron

if (row >= batch_size || col >= out_features) return;

scalar_t sum = 0;
// Each thread computes a partial sum over in_features with a stride equal to warpSize (32)
for (int i = lane; i < in_features; i += 32) {
sum += input[row * in_features + i] * weight[col * in_features + i];
}

// Use warp-level reduction via __shfl_down_sync to sum the partial results
for (int offset = 16; offset > 0; offset /= 2) {
sum += __shfl_down_sync(0xffffffff, sum, offset);
}

// The first lane writes the result along with bias
if (lane == 0) {
output[row * out_features + col] = sum + bias[col];
}
}

// ReLU kernel remains unchanged

template <typename scalar_t>
__global__ void relu_kernel(
scalar_t* __restrict__ data,
const int size) {
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
data[idx] = data[idx] > 0 ? data[idx] : 0;
}
}

// Main function to execute the MLP forward pass layer by layer

torch::Tensor mlp_cuda_forward(
torch::Tensor input,
std::vector<torch::Tensor> weights,
std::vector<torch::Tensor> biases) {

auto device = input.device();
auto num_layers = weights.size();
auto current = input;

for (size_t i = 0; i < num_layers; i++) {
const auto batch_size = current.size(0);
const auto in_features = current.size(1);
const auto out_features = weights[i].size(0);

auto output = torch::empty({batch_size, out_features},
torch::dtype(current.dtype()).device(device));

// Calculate grid and block dimensions using optimized block sizes
const dim3 block(32, BLOCK_DIM_Y);
const dim3 grid(batch_size, (out_features + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

AT_DISPATCH_FLOATING_TYPES(current.scalar_type(), "efficient_mlp_forward_kernel", ([&] {
efficient_mlp_forward_kernel<scalar_t><<<grid, block>>>(
current.data_ptr<scalar_t>(),
weights[i].data_ptr<scalar_t>(),
biases[i].data_ptr<scalar_t>(),
output.data_ptr<scalar_t>(),
batch_size,
in_features,
out_features
);
}));

// Apply ReLU for all layers except the last
if (i < num_layers - 1) {
const int size = batch_size * out_features;
const int num_blocks = (size + RELU_BLOCK_SIZE - 1) / RELU_BLOCK_SIZE;

AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "relu_kernel", ([&] {
relu_kernel<scalar_t><<<num_blocks, RELU_BLOCK_SIZE>>>(
output.data_ptr<scalar_t>(),
size
);
}));
}

current = output;
}

return current;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &mlp_cuda_forward, "Efficient MLP forward (CUDA)");
}

"""