
#include <stdio.h>
#include "utils.h"

/* CUDA <<<grid, block>> template:
 *   const dim3 grid(int(h / psize), int(w / psize));
 *   const dim3 block(n, c);
 */
template <typename scalar_t>
__global__ void cudaNativeDCTII2DKernel(const scalar_t* __restrict__ input, const uint32_t psize, scalar_t* __restrict__ output) {
    const scalar_t sqrt_2_N = sqrt(2 / psize);
    const scalar_t sqrt_1_N = sqrt(1 / psize);

    const uint32_t blockDim_h = blockDim.x * psize;
    const uint32_t blockDim_w = blockDim.y * psize;

    const uint32_t n = blockIdx.x;
    const uint32_t c = blockIdx.y;
    for (uint32_t i = 0; i < psize; i++) {
        uint32_t h = threadIdx.x * psize + i;
        for (uint32_t j = 0; j < psize; j++) {
            uint32_t w = threadIdx.y * psize + j;

            uint32_t idx = n * gridDim.y * blockDim_h * blockDim_w + c * blockDim_h * blockDim_w + h * blockDim_w + w;
            output[idx] = input[idx] * 10;
        }
    }
}

at::Tensor cudaNativeDCTII2DForward(const at::Tensor input, const uint32_t psize) {
    at::IntList inpsize = input.sizes();
    int n = inpsize[0];
    int c = inpsize[1];
    int h = inpsize[2];
    int w = inpsize[3];

    at::Tensor output = at::zeros_like(input);

    const dim3 grid(int(h / psize), int(w / psize));
    const dim3 block(n, c);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "native_dctii_2d_cuda_forward", ([&] {
                cudaNativeDCTII2DKernel<scalar_t><<<grid, block>>>(input.data<scalar_t>(), psize, output.data<scalar_t>());
            }
        )
    );

    return output;
}
