
#include "utils.h"

torch::Tensor fast_dct_cuda_forward(   
    torch::Tensor a, 
    torch::Tensor b
);


torch::Tensor fast_dct_forward(
    torch::Tensor a, 
    torch::Tensor b
) {
    return fast_dct_cuda_forward(a, b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fast_dct_forward, "LLTM forward (CUDA)");
}
