#include <iostream>
#include <torch/torch.h>


int main(int argc, char* argv[]) { 
    std::cout << argv[0] << " starting..." << std::endl;

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    

    return 0;
}
