
#include <stdio.h>

#include "../include/utils.h"

/* CUDA <<<numBlocks, threadsPerBlock>> template:
 *   const dim3 numBlocks(N, C);
 *   const dim3 threadsPerBlock(int(H / numPoints), int(W / numPoints));
 */
template <typename scalar_t>
__global__ void cudaNativeDCTII2DKernel(const scalar_t* __restrict__ input, const uint numPoints, scalar_t* __restrict__ output) {
    const float sqrt_1_2 = sqrtf(1.0f / 2);

    const uint n = blockIdx.x;
    const uint c = blockIdx.y;
    const uint C = gridDim.y;
    const uint H = blockDim.x * numPoints;
    const uint W = blockDim.y * numPoints;

    for (uint k1 = 0; k1 < numPoints; k1++) {
        uint hk = threadIdx.x * numPoints + k1;
        float lambda_k1 = k1 == 0 ? sqrt_1_2 : 1.0f;

        for (uint k2 = 0; k2 < numPoints; k2++) {
            uint wk = threadIdx.y * numPoints + k2;
            float lambda_k2 = k2 == 0 ? sqrt_1_2 : 1.0f;

            uint spectral_idx = n * C * H * W + c * H * W + hk * W + wk; 

            for (uint n1 = 0; n1 < numPoints; n1++) {
                uint hn = threadIdx.x * numPoints + n1;
                float cos_n1_k1 = cosf((2.0f * n1 + 1.0f) * k1 * M_PI / (2.0f * numPoints));

                for (uint n2 = 0; n2 < numPoints; n2++) {
                    uint wn = threadIdx.y * numPoints + n2;
                    float cos_n2_k2 = cosf((2.0f * n2 + 1.0f) * k2 * M_PI / (2.0f * numPoints));
                    
                    uint special_idx = n * C * H * W + c * H * W + hn * W + wn; 
                    output[spectral_idx] += input[special_idx] * (2.0f / numPoints) * lambda_k1 * lambda_k2 * cos_n1_k1 * cos_n2_k2;                    
                }
            }
        }
    }

    __syncthreads();
}

/* CUDA <<<numBlocks, threadsPerBlock>> template:
 *   const dim3 numBlocks(N, C);
 *   const dim3 threadsPerBlock(int(H / numPoints), int(W / numPoints));
 */
template <typename scalar_t>
__global__ void cudaNativeIDCTII2DKernel(const scalar_t* __restrict__ input, const uint numPoints, scalar_t* __restrict__ output) {
    const float sqrt_1_2 = sqrtf(1.0f / 2);

    const uint n = blockIdx.x;
    const uint c = blockIdx.y;
    const uint C = gridDim.y;
    const uint H = blockDim.x * numPoints;
    const uint W = blockDim.y * numPoints;

    for (uint n1 = 0; n1 < numPoints; n1++) {
        uint hn = threadIdx.x * numPoints + n1;

        for (uint n2 = 0; n2 < numPoints; n2++) {
            uint wn = threadIdx.y * numPoints + n2;
            
            uint special_idx = n * C * H * W + c * H * W + hn * W + wn; 

            for (uint k1 = 0; k1 < numPoints; k1++) {
                uint hk = threadIdx.x * numPoints + k1;
                float lambda_k1 = k1 == 0 ? sqrt_1_2 : 1.0f;    
                float cos_n1_k1 = cosf((2.0f * n1 + 1.0f) * k1 * M_PI / (2.0f * numPoints));

                for (uint k2 = 0; k2 < numPoints; k2++) {
                    uint wk = threadIdx.y * numPoints + k2;
                    float lambda_k2 = k2 == 0 ? sqrt_1_2 : 1.0f;    
                    float cos_n2_k2 = cosf((2.0f * n2 + 1.0f) * k2 * M_PI / (2.0f * numPoints));

                    uint spectral_idx = n * C * H * W + c * H * W + hk * W + wk; 
                    output[special_idx] += input[spectral_idx] * (2.0f / numPoints) * lambda_k1 * lambda_k2 * cos_n1_k1 * cos_n2_k2;
                }
            }
        }
    }

    __syncthreads();
}

at::Tensor cudaNativeDCTII2D(const at::Tensor input, const uint numPoints) {
    at::IntList inputSize = input.sizes();
    int N = inputSize[0];
    int C = inputSize[1];
    int H = inputSize[2];
    int W = inputSize[3];

    at::Tensor output = at::zeros_like(input);

    const dim3 numBlocks(N, C);
    const dim3 threadsPerBlock(int(H / numPoints), int(W / numPoints));
    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNativeDCTII2D", ([&] {
                cudaNativeDCTII2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(input.data<scalar_t>(), numPoints, output.data<scalar_t>());
            }
        )
    );

    return output;
}

at::Tensor cudaNativeIDCTII2D(const at::Tensor input, const uint numPoints) {
    at::IntList inputSize = input.sizes();
    int N = inputSize[0];
    int C = inputSize[1];
    int H = inputSize[2];
    int W = inputSize[3];

    at::Tensor output = at::zeros_like(input);

    const dim3 numBlocks(N, C);
    const dim3 threadsPerBlock(int(H / numPoints), int(W / numPoints));
    AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNativeIDCTII2D", ([&] {
                cudaNativeIDCTII2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(input.data<scalar_t>(), numPoints, output.data<scalar_t>());
            }
        )
    );

    return output;
}
