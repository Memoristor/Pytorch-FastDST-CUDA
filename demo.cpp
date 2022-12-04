#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "include/utils.h"


int main(int argc, char* argv[]) { 
    std::cout << argv[0] << " starting" << std::endl;

    torch::Device device = torch::kCPU;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }

    torch::Tensor tensor = torch::rand({2, 30, 30, 3});

    // tensor.cuda();

    // std::cout << tensor << std::endl;
    // std::cout << torch::cuda::is_available() << std::endl;
    // std::cout << torch::cuda::device_count() << std::endl;
    
    // std::cout << tensor.scalar_type() << std::endl;
    // std::cout << tensor.sizes()[0] << std::endl;
    // std::cout << tensor.sizes()[0] << std::endl;
    // std::cout << tensor.device() << std::endl;



    return 0;
}
