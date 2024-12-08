
#include "../include/sort_zigzag_kernel.cuh"

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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaRecoverCoefficientsByZigzag", ([&] {
        cudaRecoverCoefficientsByZigzagKernel<scalar_t>
            <<<numBlocks, threadsPerBlock>>>(totalThreads, input.data_ptr<scalar_t>(), height,
                                             width, output.data_ptr<scalar_t>(), points, zigzagGPU);
      }));

  CHECK_CUDA_ERROR(cudaFree(zigzagGPU));

  return output;
}
