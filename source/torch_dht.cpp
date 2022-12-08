
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNativeDHT2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNativeIDHT2D(const at::Tensor input, const uint numPoints);

at::Tensor nativeDHT2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNativeDHT2D(padInput, numPoints);
}

at::Tensor nativeIDHT2D(const at::Tensor input, const uint numPoints, const bool sortCoff) {
    CHECK_4DTENSOR(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNativeIDHT2D(padInput, numPoints);
}
