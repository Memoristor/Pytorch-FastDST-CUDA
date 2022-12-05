
#include "utils.h"

at::Tensor native_dctii_2d_cuda_forward(const at::Tensor input, const uint32_t size, const bool sort_freq = true);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("native_dctii_2d", &native_dctii_2d_cuda_forward, "Native DCT-II forward (CUDA)");
}
