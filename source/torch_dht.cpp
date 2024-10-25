
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNaiveDHT2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNaiveIDHT2D(const at::Tensor input, const uint numPoints);

at::Tensor naiveDHT2D(const at::Tensor input, const uint numPoints) {
    CHECK_INPUT(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNaiveDHT2D(padInput, numPoints);
}

at::Tensor naiveIDHT2D(const at::Tensor input, const uint numPoints) {
    CHECK_INPUT(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNaiveIDHT2D(padInput, numPoints);
}
