#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "include/utils.h"


// __global__ void assign(int* a_d, int value){
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     a_d[idx] = value;
// }


int main(int argc, char* argv[]) { 
    std::cout << argv[0] << " starting..." << std::endl;

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    
    std::cout << tensor.scalar_type() << std::endl;

    // CHECK_INPUT(tensor);

    // int num_elem = 6;

    // dim3 block(3);
    // dim3 grid((num_elem + block.x - 1) / block.x);
    // inc_gpu<<<grid, dim>>>(a, num_elem);

    return 0;
}
