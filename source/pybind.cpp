
#include "../include/utils.h"

at::Tensor naiveDCT2D(const at::Tensor input, const uint numPoints);
at::Tensor naiveIDCT2D(const at::Tensor input, const uint numPoints);

at::Tensor naiveDST2D(const at::Tensor input, const uint numPoints);
at::Tensor naiveIDST2D(const at::Tensor input, const uint numPoints);

at::Tensor naiveDHT2D(const at::Tensor input, const uint numPoints);
at::Tensor naiveIDHT2D(const at::Tensor input, const uint numPoints);

at::Tensor sortCoefficients(const at::Tensor input, const uint numPoints, const uint priority);
at::Tensor recoverCoefficients(const at::Tensor input, const uint numPoints, const uint priority);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naiveDCT2D", &naiveDCT2D, "Naive 2D-DCT (CUDA)");
  m.def("naiveIDCT2D", &naiveIDCT2D, "Naive 2D-IDCT (CUDA)");

  m.def("naiveDST2D", &naiveDST2D, "Naive 2D-DST (CUDA)");
  m.def("naiveIDST2D", &naiveIDST2D, "Naive 2D-IDST (CUDA)");

  m.def("naiveDHT2D", &naiveDHT2D, "Naive 2D-DHT (CUDA)");
  m.def("naiveIDHT2D", &naiveIDHT2D, "Naive 2D-IDHT (CUDA)");

  m.def("sortCoefficients", &sortCoefficients, "Sort Coefficients by Zigzag (CUDA)");
  m.def("recoverCoefficients", &recoverCoefficients, "Recover Coefficients by Zigzag (CUDA)");
}
