
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNativeDCT2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNativeIDCT2D(const at::Tensor input, const uint numPoints);

at::Tensor nativeDCT2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNativeDCT2D(padInput, numPoints);
}

at::Tensor nativeIDCT2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNativeIDCT2D(padInput, numPoints);
}
