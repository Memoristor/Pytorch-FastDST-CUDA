
#include <stdio.h>

#include "../include/utils.h"

template <typename scalar_t>
__global__ void cudaNativeDST2DKernel(const uint numTotalThreads, const uint batchSizeDim, const uint channelDim, const uint heightDim, const uint widthDim, const scalar_t* __restrict__ input, const uint numPoints, scalar_t* __restrict__ output) {
    
    const uint idx = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numTotalThreads) {
        
        const uint chwDim = channelDim * heightDim * widthDim;
        const uint hwDim = heightDim * widthDim;
        const uint n = int(idx / chwDim);
        const uint c = int((idx - n * chwDim) / hwDim);
        const uint h = int((idx - n * chwDim - c * hwDim) / widthDim);
        const uint w = idx - n * chwDim - c * hwDim - h * widthDim;
        const uint numPointsPow2 = numPoints * numPoints;

        for (uint k = 0; k < numPoints; k++) {
            uint hk = h * numPoints + k;

            for (uint v = 0; v < numPoints; v++) {
                uint wv = w * numPoints + v;

                uint spectralIdx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv; 

                for (uint i = 0; i < numPoints; i++) {
                    uint hi = h * numPoints + i;
                    float sin_i_k = sinf((i + 1.0f) * (k + 1.0f) * M_PI / (numPoints + 1));

                    for (uint j = 0; j < numPoints; j++) {
                        uint wj = w * numPoints + j;
                        float sin_j_v = sinf((j + 1.0f) * (v + 1.0f) * M_PI / (numPoints + 1));
                        
                        uint specialIdx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hi * widthDim * numPoints + wj; 
                        output[spectralIdx] += input[specialIdx] * (2.0f / (numPoints + 1)) * sin_i_k * sin_j_v;                    
                    }
                }
            }
        }
    }

    __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNativeIDST2DKernel(const uint numTotalThreads, const uint batchSizeDim, const uint channelDim, const uint heightDim, const uint widthDim, const scalar_t* __restrict__ input, const uint numPoints, scalar_t* __restrict__ output) {
    
    const uint idx = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numTotalThreads) {
        
        const uint chwDim = channelDim * heightDim * widthDim;
        const uint hwDim = heightDim * widthDim;
        const uint n = int(idx / chwDim);
        const uint c = int((idx - n * chwDim) / hwDim);
        const uint h = int((idx - n * chwDim - c * hwDim) / widthDim);
        const uint w = idx - n * chwDim - c * hwDim - h * widthDim;
        const uint numPointsPow2 = numPoints * numPoints;

        for (uint i = 0; i < numPoints; i++) {
            uint hi = h * numPoints + i;

            for (uint j = 0; j < numPoints; j++) {
                uint wj = w * numPoints + j;
                
                uint specialIdx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hi * widthDim * numPoints + wj; 

                for (uint k = 0; k < numPoints; k++) {
                    uint hk = h * numPoints + k;
                    float sin_i_k = sinf((i + 1.0f) * (k + 1.0f) * M_PI / (numPoints + 1));

                    for (uint v = 0; v < numPoints; v++) {
                        uint wv = w * numPoints + v;
                        float sin_j_v = sinf((j + 1.0f) * (v + 1.0f) * M_PI / (numPoints + 1));

                        uint spectralIdx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv; 
                        output[specialIdx] += input[spectralIdx] * (2.0f / (numPoints + 1)) * sin_i_k * sin_j_v;
                    }
                }
            }
        }
    }

    __syncthreads();
}

at::Tensor cudaNativeDST2D(const at::Tensor input, const uint numPoints) {
    at::IntList inputSize = input.sizes();
    int batchSize = inputSize[0];
    int channel = inputSize[1];
    int height = inputSize[2];
    int width = inputSize[3];

    at::Tensor output = at::zeros_like(input);

    dim3 numBlocks;
    dim3 threadsPerBlock;

    uint numTotalThreads = batchSize * channel * height * width / (numPoints * numPoints);
    optimalCUDABlocksAndThreadsPerBlock(numTotalThreads, numBlocks, threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNativeDST2D", ([&] {
                cudaNativeDST2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                    numTotalThreads, batchSize, channel, height / numPoints, width / numPoints, input.data_ptr<scalar_t>(), numPoints, output.data_ptr<scalar_t>()
                );
            }
        )
    );

    return output;
}

at::Tensor cudaNativeIDST2D(const at::Tensor input, const uint numPoints) {
    at::IntList inputSize = input.sizes();
    int batchSize = inputSize[0];
    int channel = inputSize[1];
    int height = inputSize[2];
    int width = inputSize[3];

    at::Tensor output = at::zeros_like(input);

    dim3 numBlocks;
    dim3 threadsPerBlock;

    uint numTotalThreads = batchSize * channel * height * width / (numPoints * numPoints);
    optimalCUDABlocksAndThreadsPerBlock(numTotalThreads, numBlocks, threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNativeIDST2D", ([&] {
                cudaNativeIDST2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                    numTotalThreads, batchSize, channel, height / numPoints, width / numPoints, input.data_ptr<scalar_t>(), numPoints, output.data_ptr<scalar_t>()
                );
            }
        )
    );

    return output;
}
