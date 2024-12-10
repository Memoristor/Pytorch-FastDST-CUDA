
#include "../include/torch_dct_kernel.cuh"
#include "../include/utils.h"

at::Tensor cudaNaiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

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
  uint totalThreads = input.numel();
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "cudaNaiveDCT2D", ([&] {
                                        cudaNaiveDCT2DAndSortCoefficientsByZigzagKernel<scalar_t>
                                            <<<numBlocks, threadsPerBlock, p2 * sizeof(uint)>>>(
                                                totalThreads, input.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(), height, width, points,
                                                sortbyZigzag);
                                      }));
  return output;
}

at::Tensor cudaNaiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];
  int p2 = points * points;

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
  uint totalThreads = input.numel();
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveIDCT2D", ([&] {
        cudaNaiveIDCT2DAndRecoverCoefficientsByZigzagKernel<scalar_t>
            <<<numBlocks, threadsPerBlock, p2 * sizeof(uint)>>>(
                totalThreads, input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), height,
                width, points, recoverbyZigzag);
      }));

  return output;
}
