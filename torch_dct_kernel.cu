
#include "utils.h"

#include <iostream>


template <typename scalar_t>
__global__ void native_dctii_2d_kernel(const scalar_t* __restrict__ input, const uint32_t size, const bool sort_freq, scalar_t* __restrict__ output) {
    // const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // c[idx] = a[idx] + b[idx]; 
}

at::Tensor native_dctii_2d_cuda_forward(const at::Tensor input, const uint32_t size, const bool sort_freq = true) {
    CHECK_INPUT(input);

    std::cout << "??" << std::endl;

    return at::zeros_like(input);
}

// at::Tensor fast_dct_cuda_forward(
//     at::Tensor a, 
//     at::Tensor b
// ) {
//     auto c = at::zeros_like(a);

//     const dim3 grid(32, 1);
//     const dim3 block(3, 1);

//     AT_DISPATCH_FLOATING_TYPES(a.type(), "fast_dct_cuda_forward", ([&] {
//                 testKernel<scalar_t><<<grid, block>>>(
//                     a.data<scalar_t>(),
//                     b.data<scalar_t>(),
//                     c.data<scalar_t>()
//                 );
//             }
//         )
//     );

//     return c;
// }
