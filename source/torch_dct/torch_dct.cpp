
#include "../../include/utils.h"

at::Tensor cudaNaiveDCT2D(const at::Tensor input, const uint points);
at::Tensor cudaNaiveIDCT2D(const at::Tensor input, const uint points);

at::Tensor naiveDCT2D(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveDCT2D(padInput, points);
}

at::Tensor naiveIDCT2D(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveIDCT2D(padInput, points);
}
