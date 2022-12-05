#include <iostream>

#include "torch_dct.cuh"


int main(int argc, char* argv[]) { 
    std::cout << argv[0] << " starting" << std::endl;

    torch::Device device = torch::kCPU;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Demo runs on GPU." << std::endl;
        device = torch::kCUDA;
    }

    torch::Tensor a = torch::rand({32 * 3});
    torch::Tensor b = torch::rand({32 * 3});
    torch::Tensor c = torch::zeros({32 * 3});

    a = a.cuda();
    b = b.cuda();
    c = c.cuda();
    test_sumGPU(a, b, c);

    return 0;
}
