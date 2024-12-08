
#include "../include/torch_dst_kernel.cuh"

at::Tensor cudaNaiveDST2D(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];

  at::Tensor output = at::zeros_like(input);

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint totalThreads = input.numel() / (points * points);
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveDST2D", ([&] {
        cudaNaiveDST2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            totalThreads, input.data_ptr<scalar_t>(), points, output.data_ptr<scalar_t>(),
            height / points, width / points);
      }));

  return output;
}

at::Tensor cudaNaiveIDST2D(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];

  at::Tensor output = at::zeros_like(input);

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint totalThreads = input.numel() / (points * points);
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveIDST2D", ([&] {
        cudaNaiveIDST2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            totalThreads, input.data_ptr<scalar_t>(), points, output.data_ptr<scalar_t>(),
            height / points, width / points);
      }));

  return output;
}
