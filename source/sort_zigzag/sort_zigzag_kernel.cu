
#include "../../include/utils.h"

template <typename scalar_t>
__global__ void cudaSortCoefficientsByZigzagKernel(const uint totalThreads,
                                                   const scalar_t* __restrict__ input,
                                                   const uint points, scalar_t* __restrict__ output,
                                                   const uint hDim, const uint wDim,
                                                   const uint* zigzag) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  if (idx < totalThreads) {
    const uint hwDim = hDim * wDim;
    const uint n = int(idx / hwDim);
    const uint h = int((idx % hwDim) / wDim);
    const uint w = idx % wDim;
    const uint p2 = points * points;

    for (uint k = 0; k < points; k++) {
      uint hk = h * points + k;
      for (uint v = 0; v < points; v++) {
        uint wv = w * points + v;

        uint inputIndex = n * hwDim * p2 + hk * wDim * points + wv;
        uint outputIndex = n * hwDim * p2 + zigzag[k * points + v] * hwDim + h * wDim + w;
        output[outputIndex] = input[inputIndex];
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaRecoverCoefficientsByZigzagKernel(const uint totalThreads,
                                                      const scalar_t* __restrict__ input,
                                                      const uint hDim, const uint wDim,
                                                      scalar_t* __restrict__ output,
                                                      const uint points, const uint* zigzag) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  if (idx < totalThreads) {
    const uint hwDim = hDim * wDim;
    const uint n = int(idx / hwDim);
    const uint h = int((idx % hwDim) / wDim);
    const uint w = idx % wDim;
    const uint p2 = points * points;

    for (uint k = 0; k < points; k++) {
      uint hk = h * points + k;
      for (uint v = 0; v < points; v++) {
        uint wv = w * points + v;

        uint outputIndex = n * hwDim * p2 + hk * wDim * points + wv;
        uint inputIndex = n * hwDim * p2 + zigzag[k * points + v] * hwDim + h * wDim + w;
        output[outputIndex] = input[inputIndex];
      }
    }
  }

  __syncthreads();
}

at::Tensor cudaSortCoefficientsByZigzag(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

  uint zigzag[p2] = {0};
  calculateZigzag(zigzag, points);

  uint* zigzagGPU;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&zigzagGPU, p2 * sizeof(uint)));
  CHECK_CUDA_ERROR(cudaMemcpy(zigzagGPU, zigzag, p2 * sizeof(uint), cudaMemcpyHostToDevice));

  std::vector<int64_t> outputSize(inputSize.begin(), inputSize.end() - 2);
  outputSize.push_back(p2);
  outputSize.push_back(height / points);
  outputSize.push_back(width / points);
  at::Tensor output = at::zeros(at::IntArrayRef(outputSize), input.options());

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint totalThreads = input.numel() / p2;
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "cudaSortCoefficientsByZigzag", ([&] {
        cudaSortCoefficientsByZigzagKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            totalThreads, input.data_ptr<scalar_t>(), points, output.data_ptr<scalar_t>(),
            height / points, width / points, zigzagGPU);
      }));

  CHECK_CUDA_ERROR(cudaFree(zigzagGPU));

  return output;
}

at::Tensor cudaRecoverCoefficientsByZigzag(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

  uint zigzag[p2] = {0};
  calculateZigzag(zigzag, points);

  uint* zigzagGPU;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&zigzagGPU, p2 * sizeof(uint)));
  CHECK_CUDA_ERROR(cudaMemcpy(zigzagGPU, zigzag, p2 * sizeof(uint), cudaMemcpyHostToDevice));

  std::vector<int64_t> outputSize(inputSize.begin(), inputSize.end() - 3);
  outputSize.push_back(height * points);
  outputSize.push_back(width * points);
  at::Tensor output = at::zeros(at::IntArrayRef(outputSize), input.options());

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint totalThreads = input.numel() / p2;
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaRecoverCoefficientsByZigzag", ([&] {
                               cudaRecoverCoefficientsByZigzagKernel<scalar_t>
                                   <<<numBlocks, threadsPerBlock>>>(
                                       totalThreads, input.data_ptr<scalar_t>(), height, width,
                                       output.data_ptr<scalar_t>(), points, zigzagGPU);
                             }));

  CHECK_CUDA_ERROR(cudaFree(zigzagGPU));

  return output;
}
