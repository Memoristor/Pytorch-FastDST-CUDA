
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNativeDCTII2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNativeIDCTII2D(const at::Tensor input, const uint numPoints);

at::Tensor nativeDCTII2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitDCTPointSize(input, numPoints);
    return cudaNativeDCTII2D(padInput, numPoints);
}

at::Tensor nativeIDCTII2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitDCTPointSize(input, numPoints);
    return cudaNativeIDCTII2D(padInput, numPoints);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nativeDCTII2D", &nativeDCTII2D, "Native 2D DCT-II (CUDA)");
  m.def("nativeIDCTII2D", &nativeIDCTII2D, "Native 2D IDCT-II (CUDA)");
}
