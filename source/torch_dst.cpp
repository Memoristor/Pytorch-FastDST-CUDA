
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNaiveDST2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNaiveIDST2D(const at::Tensor input, const uint numPoints);

at::Tensor naiveDST2D(const at::Tensor input, const uint numPoints) {
    CHECK_INPUT(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNaiveDST2D(padInput, numPoints);
}

at::Tensor naiveIDST2D(const at::Tensor input, const uint numPoints) {
    CHECK_INPUT(input);
    at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
    return cudaNaiveIDST2D(padInput, numPoints);
}
