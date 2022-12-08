
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNativeDST2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNativeIDST2D(const at::Tensor input, const uint numPoints);

at::Tensor nativeDST2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNativeDST2D(padInput, numPoints);
}

at::Tensor nativeIDST2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNativeIDST2D(padInput, numPoints);
}
