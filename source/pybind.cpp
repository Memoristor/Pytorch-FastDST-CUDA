
#include "../include/utils.h"

at::Tensor nativeDCT2D(const at::Tensor input, const uint numPoints, const bool sortCoff);
at::Tensor nativeIDCT2D(const at::Tensor input, const uint numPoints, const bool sortCoff);

at::Tensor nativeDST2D(const at::Tensor input, const uint numPoints, const bool sortCoff);
at::Tensor nativeIDST2D(const at::Tensor input, const uint numPoints, const bool sortCoff);

at::Tensor nativeDHT2D(const at::Tensor input, const uint numPoints, const bool sortCoff);
at::Tensor nativeIDHT2D(const at::Tensor input, const uint numPoints, const bool sortCoff);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nativeDCT2D", &nativeDCT2D, "Native 2D-DCT (CUDA)");
  m.def("nativeIDCT2D", &nativeIDCT2D, "Native 2D-IDCT (CUDA)");

  m.def("nativeDST2D", &nativeDST2D, "Native 2D-DST (CUDA)");
  m.def("nativeIDST2D", &nativeIDST2D, "Native 2D-IDST (CUDA)");

  m.def("nativeDHT2D", &nativeDHT2D, "Native 2D-DHT (CUDA)");
  m.def("nativeIDHT2D", &nativeIDHT2D, "Native 2D-IDHT (CUDA)");
}
