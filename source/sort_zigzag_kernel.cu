
#include <stdio.h>

#include "../include/utils.h"

template <typename scalar_t>
__global__ void cudaSortCoefficientsByZigzagKernel(const uint numTotalThreads, const uint batchSizeDim, const uint channelDim, const uint heightDim, const uint widthDim, const scalar_t* __restrict__ input, const uint numPoints, const uint* zigzag, const uint priority, scalar_t* __restrict__ output) {
    
    const uint idx = threadIdx.x + blockIdx.x * blockDim.x
                   + (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x
                   + (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    if (idx < numTotalThreads) {
        
        const uint chwDim = channelDim * heightDim * widthDim;
        const uint hwDim = heightDim * widthDim;
        const uint n = int(idx / chwDim);
        const uint c = int((idx % chwDim) / hwDim);
        const uint h = int((idx % hwDim) / widthDim);
        const uint w = idx % widthDim;
        const uint numPointsPow2 = numPoints * numPoints;

        for (uint k = 0; k < numPoints; k++) {
            uint hk = h * numPoints + k;
            for (uint v = 0; v < numPoints; v++) {
                uint wv = w * numPoints + v;
                
                uint inputSpectralIndex = k * numPoints + v;
                uint outputSpectralIndex = zigzag[inputSpectralIndex];
                uint outputChannelIndex = priority == SORT_BY_FREQUENCIES ? outputSpectralIndex * channelDim + c : c * numPointsPow2 + outputSpectralIndex;
                uint inputIndex = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv; 
                uint outputIndex = n * chwDim * numPointsPow2 + outputChannelIndex * hwDim + h * widthDim + w;
                output[outputIndex] = input[inputIndex];
            }
        }
    }

    __syncthreads();
}

template <typename scalar_t>
__global__ void cudaRecoverCoefficientsByZigzagKernel(const uint numTotalThreads, const uint batchSizeDim, const uint channelDim, const uint heightDim, const uint widthDim, const scalar_t* __restrict__ input, const uint numPoints, const uint* zigzag, const uint priority, scalar_t* __restrict__ output) {
    
    const uint idx = threadIdx.x + blockIdx.x * blockDim.x
                   + (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x
                   + (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    if (idx < numTotalThreads) {
        
        const uint chwDim = channelDim * heightDim * widthDim;
        const uint hwDim = heightDim * widthDim;
        const uint n = int(idx / chwDim);
        const uint c = int((idx % chwDim) / hwDim);
        const uint h = int((idx % hwDim) / widthDim);
        const uint w = idx % widthDim;
        const uint numPointsPow2 = numPoints * numPoints;

        for (uint k = 0; k < numPoints; k++) {
            uint hk = h * numPoints + k;
            for (uint v = 0; v < numPoints; v++) {
                uint wv = w * numPoints + v;

                uint outputSpectralIndex = k * numPoints + v;
                uint inputSpectralIndex = zigzag[outputSpectralIndex];
                uint inputChannelIndex = priority == SORT_BY_FREQUENCIES ? inputSpectralIndex * channelDim + c : c * numPointsPow2 + inputSpectralIndex;
                uint outputIndex = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv; 
                uint inputIndex = n * chwDim * numPointsPow2 + inputChannelIndex * hwDim + h * widthDim + w;
                output[outputIndex] = input[inputIndex];
            }
        }
    }

    __syncthreads();
}

at::Tensor cudaSortCoefficientsByZigzag(const at::Tensor input, const uint numPoints, const uint priority) {
    at::IntList inputSize = input.sizes();
    int batchSize = inputSize[0];
    int channel = inputSize[1];
    int height = inputSize[2];
    int width = inputSize[3];

    uint zigzag[numPoints * numPoints] = {0};
    calculateZigzag(zigzag, numPoints);

    uint *zigzagGPU;
    CHECK_CUDA_ERROR(cudaMalloc((void **) &zigzagGPU, numPoints * numPoints * sizeof(uint)));
    CHECK_CUDA_ERROR(cudaMemcpy(zigzagGPU, zigzag, numPoints * numPoints * sizeof(uint), cudaMemcpyHostToDevice));

    at::Tensor output = at::zeros({batchSize, channel * numPoints * numPoints, height / numPoints, width / numPoints}, input.options());

    dim3 numBlocks;
    dim3 threadsPerBlock;

    uint numTotalThreads = batchSize * channel * height * width / (numPoints * numPoints);
    optimalCUDABlocksAndThreadsPerBlock(numTotalThreads, numBlocks, threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaSortCoefficientsByZigzag", ([&] {
                cudaSortCoefficientsByZigzagKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                    numTotalThreads, batchSize, channel, height / numPoints, width / numPoints, input.data_ptr<scalar_t>(), numPoints, zigzagGPU, priority, output.data_ptr<scalar_t>()
                );
            }
        )
    );

    CHECK_CUDA_ERROR(cudaFree(zigzagGPU));

    return output;
}


at::Tensor cudaRecoverCoefficientsByZigzag(const at::Tensor input, const uint numPoints, const uint priority) {
    at::IntList inputSize = input.sizes();
    int batchSize = inputSize[0];
    int channel = inputSize[1];
    int height = inputSize[2];
    int width = inputSize[3];

    uint zigzag[numPoints * numPoints] = {0};
    calculateZigzag(zigzag, numPoints);

    uint *zigzagGPU;
    CHECK_CUDA_ERROR(cudaMalloc((void **) &zigzagGPU, numPoints * numPoints * sizeof(uint)));
    CHECK_CUDA_ERROR(cudaMemcpy(zigzagGPU, zigzag, numPoints * numPoints * sizeof(uint), cudaMemcpyHostToDevice));

    at::Tensor output = at::zeros({batchSize, channel / (numPoints * numPoints), height * numPoints, width * numPoints}, input.options());

    dim3 numBlocks;
    dim3 threadsPerBlock;

    uint numTotalThreads = batchSize * channel * height * width / (numPoints * numPoints);
    optimalCUDABlocksAndThreadsPerBlock(numTotalThreads, numBlocks, threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaRecoverCoefficientsByZigzag", ([&] {
                cudaRecoverCoefficientsByZigzagKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                    numTotalThreads, batchSize, channel / (numPoints * numPoints), height, width, input.data_ptr<scalar_t>(), numPoints, zigzagGPU, priority, output.data_ptr<scalar_t>()
                );
            }
        )
    );

    CHECK_CUDA_ERROR(cudaFree(zigzagGPU));

    return output;
}
