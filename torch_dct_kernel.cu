
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void testKernel(
    scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    scalar_t* __restrict__ c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx]; 
}



at::Tensor fast_dct_cuda_forward(
    at::Tensor a, 
    at::Tensor b
) {
    auto c = at::zeros_like(a);

    const dim3 grid(32, 1);
    const dim3 block(3, 1);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "fast_dct_cuda_forward", ([&] {
                testKernel<scalar_t><<<grid, block>>>(
                    a.data<scalar_t>(),
                    b.data<scalar_t>(),
                    c.data<scalar_t>()
                );
            }
        )
    );

    return c;
}
