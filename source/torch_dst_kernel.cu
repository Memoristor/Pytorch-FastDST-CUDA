
#include "../include/torch_dst_kernel.cuh"

at::Tensor cudaNaiveDST2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

  uint zigzag[p2] = {0};
  uint* zigzagGPU;
  calculateZigzag(zigzag, points);
  CHECK_CUDA_ERROR(cudaMalloc((void**)&zigzagGPU, p2 * sizeof(uint)));
  CHECK_CUDA_ERROR(cudaMemcpy(zigzagGPU, zigzag, p2 * sizeof(uint), cudaMemcpyHostToDevice));

  std::vector<int64_t> outputSize;
  if (sortbyZigzag) {
    outputSize.assign(inputSize.begin(), inputSize.end() - 2);
    outputSize.push_back(p2);
    outputSize.push_back(height / points);
    outputSize.push_back(width / points);
  } else {
    outputSize.assign(inputSize.begin(), inputSize.end());
  }
  at::Tensor output = at::zeros(at::IntArrayRef(outputSize), input.options());

  dim3 numBlocks;
  dim3 threadsPerBlock;
  uint totalThreads = input.numel() / (points * points);
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveDST2D", ([&] {
        cudaNaiveDST2DAndSortCoefficientsByZigzagKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            totalThreads, input.data_ptr<scalar_t>(), points, output.data_ptr<scalar_t>(),
            height / points, width / points, sortbyZigzag, zigzagGPU);
      }));

  return output;
}

at::Tensor cudaNaiveIDST2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

  uint zigzag[p2] = {0};
  uint* zigzagGPU;
  calculateZigzag(zigzag, points);
  CHECK_CUDA_ERROR(cudaMalloc((void**)&zigzagGPU, p2 * sizeof(uint)));
  CHECK_CUDA_ERROR(cudaMemcpy(zigzagGPU, zigzag, p2 * sizeof(uint), cudaMemcpyHostToDevice));

  std::vector<int64_t> outputSize;
  if (recoverbyZigzag) {
    height = height * points;
    width = width * points;
    outputSize.assign(inputSize.begin(), inputSize.end() - 3);
    outputSize.push_back(height);
    outputSize.push_back(width);
  } else {
    outputSize.assign(inputSize.begin(), inputSize.end());
  }
  at::Tensor output = at::zeros(at::IntArrayRef(outputSize), input.options());

  dim3 numBlocks;
  dim3 threadsPerBlock;
  uint totalThreads = input.numel() / p2;
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveIDST2D", ([&] {
        cudaNaiveIDST2DAndRecoverCoefficientsByZigzagKernel<scalar_t>
            <<<numBlocks, threadsPerBlock>>>(totalThreads, input.data_ptr<scalar_t>(), points,
                                             output.data_ptr<scalar_t>(), height / points,
                                             width / points, recoverbyZigzag, zigzagGPU);
      }));

  return output;
}
