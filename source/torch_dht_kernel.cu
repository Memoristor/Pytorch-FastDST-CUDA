
#include "../include/torch_dht_kernel.cuh"

at::Tensor cudaNaiveDHT2D(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];

  at::Tensor output = at::zeros_like(input);

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint totalThreads = input.numel() / (points * points);
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveDHT2D", ([&] {
        cudaNaiveDHT2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            totalThreads, input.data_ptr<scalar_t>(), points, output.data_ptr<scalar_t>(),
            height / points, width / points);
      }));

  return output;
}

at::Tensor cudaNaiveIDHT2D(const at::Tensor input, const uint points) {
  at::IntList inputSize = input.sizes();
  int height = inputSize[inputSize.size() - 2];
  int width = inputSize[inputSize.size() - 1];

  at::Tensor output = at::zeros_like(input);

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint totalThreads = input.numel() / (points * points);
  optimalCUDABlocksAndThreadsPerBlock(totalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "cudaNaiveIDHT2D", ([&] {
        cudaNaiveIDHT2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            totalThreads, input.data_ptr<scalar_t>(), points, output.data_ptr<scalar_t>(),
            height / points, width / points);
      }));

  return output;
}
