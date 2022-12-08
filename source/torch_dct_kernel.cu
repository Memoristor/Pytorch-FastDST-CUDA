
#include <stdio.h>

#include "../include/utils.h"

template <typename scalar_t>
__global__ void cudaNativeDCTII2DKernel(const uint numTotalThreads, const uint batchSizeDim, const uint channelDim, const uint heightDim, const uint widthDim, const scalar_t* __restrict__ input, const uint numPoints, scalar_t* __restrict__ output) {
    
    const uint idx = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numTotalThreads) {
        
        const uint chwDim = channelDim * heightDim * widthDim;
        const uint hwDim = heightDim * widthDim;
        const uint n = int(idx / chwDim);
        const uint c = int((idx - n * chwDim) / hwDim);
        const uint h = int((idx - n * chwDim - c * hwDim) / widthDim);
        const uint w = idx - n * chwDim - c * hwDim - h * widthDim;
        const uint numPointsPow2 = numPoints * numPoints;

        const float sqrt_1_2 = sqrtf(1.0f / 2);
        for (uint k = 0; k < numPoints; k++) {
            uint hk = h * numPoints + k;
            float lambda_k = k == 0 ? sqrt_1_2 : 1.0f;

            for (uint v = 0; v < numPoints; v++) {
                uint wv = w * numPoints + v;
                float lambda_v = v == 0 ? sqrt_1_2 : 1.0f;

                uint spectral_idx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv; 

                for (uint i = 0; i < numPoints; i++) {
                    uint hi = h * numPoints + i;
                    float cos_i_k = cosf((2.0f * i + 1.0f) * k * M_PI / (2.0f * numPoints));

                    for (uint j = 0; j < numPoints; j++) {
                        uint wj = w * numPoints + j;
                        float cos_j_v = cosf((2.0f * j + 1.0f) * v * M_PI / (2.0f * numPoints));
                        
                        uint special_idx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hi * widthDim * numPoints + wj; 
                        output[spectral_idx] += input[special_idx] * (2.0f / numPoints) * lambda_k * lambda_v * cos_i_k * cos_j_v;                    
                    }
                }
            }
        }
    }

    __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNativeIDCTII2DKernel(const uint numTotalThreads, const uint batchSizeDim, const uint channelDim, const uint heightDim, const uint widthDim, const scalar_t* __restrict__ input, const uint numPoints, scalar_t* __restrict__ output) {
    
    const uint idx = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numTotalThreads) {
        
        const uint chwDim = channelDim * heightDim * widthDim;
        const uint hwDim = heightDim * widthDim;
        const uint n = int(idx / chwDim);
        const uint c = int((idx - n * chwDim) / hwDim);
        const uint h = int((idx - n * chwDim - c * hwDim) / widthDim);
        const uint w = idx - n * chwDim - c * hwDim - h * widthDim;
        const uint numPointsPow2 = numPoints * numPoints;

        const float sqrt_1_2 = sqrtf(1.0f / 2);
        for (uint i = 0; i < numPoints; i++) {
            uint hi = h * numPoints + i;

            for (uint j = 0; j < numPoints; j++) {
                uint wj = w * numPoints + j;
                
                uint special_idx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hi * widthDim * numPoints + wj; 

                for (uint k = 0; k < numPoints; k++) {
                    uint hk = h * numPoints + k;
                    float lambda_k = k == 0 ? sqrt_1_2 : 1.0f;    
                    float cos_i_k = cosf((2.0f * i + 1.0f) * k * M_PI / (2.0f * numPoints));

                    for (uint v = 0; v < numPoints; v++) {
                        uint wv = w * numPoints + v;
                        float lambda_v = v == 0 ? sqrt_1_2 : 1.0f;    
                        float cos_j_v = cosf((2.0f * j + 1.0f) * v * M_PI / (2.0f * numPoints));

                        uint spectral_idx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv; 
                        output[special_idx] += input[spectral_idx] * (2.0f / numPoints) * lambda_k * lambda_v * cos_i_k * cos_j_v;
                    }
                }
            }
        }
    }

    __syncthreads();
}

at::Tensor cudaNativeDCTII2D(const at::Tensor input, const uint numPoints) {
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

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNativeDCTII2D", ([&] {
                cudaNativeDCTII2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                    numTotalThreads, batchSize, channel, height / numPoints, width / numPoints, input.data<scalar_t>(), numPoints, output.data<scalar_t>()
                );
            }
        )
    );

    return output;
}

at::Tensor cudaNativeIDCTII2D(const at::Tensor input, const uint numPoints) {
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

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNativeIDCTII2D", ([&] {
                cudaNativeIDCTII2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                    numTotalThreads, batchSize, channel, height / numPoints, width / numPoints, input.data<scalar_t>(), numPoints, output.data<scalar_t>()
                );
            }
        )
    );

    return output;
}
