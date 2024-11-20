
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaNaiveDCT2D(const at::Tensor input, const uint numPoints);
at::Tensor cudaNaiveIDCT2D(const at::Tensor input, const uint numPoints);

at::Tensor naiveDCT2D(const at::Tensor input, const uint numPoints) {
  CHECK_INPUT(input);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
  return cudaNaiveDCT2D(padInput, numPoints);
}

at::Tensor naiveIDCT2D(const at::Tensor input, const uint numPoints) {
  CHECK_INPUT(input);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, numPoints);
  return cudaNaiveIDCT2D(padInput, numPoints);
}
