
#include "../include/sort_zigzag_kernel.cuh"
#include "../include/utils.h"

at::Tensor cudaSortCoefficientsByZigzag(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

  std::vector<int64_t> outputSize(inputSize.begin(), inputSize.end() - 2);
  outputSize.push_back(p2);
  outputSize.push_back(height / points);
  outputSize.push_back(width / points);
  at::Tensor output = at::zeros(at::IntArrayRef(outputSize), input.options());

  dim3 numBlocks;
  dim3 threadsPerBlock;
  uint totalThreads = input.numel();
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "cudaSortCoefficientsByZigzag", ([&] {
                                        cudaSortCoefficientsByZigzagKernel<scalar_t>
                                            <<<numBlocks, threadsPerBlock, p2 * sizeof(uint)>>>(
                                                totalThreads, input.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(), height, width, points);
                                      }));

  return output;
}

at::Tensor cudaRecoverCoefficientsByZigzag(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

  std::vector<int64_t> outputSize(inputSize.begin(), inputSize.end() - 3);
  height = height * points;
  width = width * points;
  outputSize.push_back(height);
  outputSize.push_back(width);
  at::Tensor output = at::zeros(at::IntArrayRef(outputSize), input.options());

  dim3 numBlocks;
  dim3 threadsPerBlock;
  uint totalThreads = input.numel();
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "cudaRecoverCoefficientsByZigzag", ([&] {
                                        cudaRecoverCoefficientsByZigzagKernel<scalar_t>
                                            <<<numBlocks, threadsPerBlock, p2 * sizeof(uint)>>>(
                                                totalThreads, input.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(), height, width, points);
                                      }));

  return output;
}
