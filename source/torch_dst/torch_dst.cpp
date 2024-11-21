
#include "../../include/utils.h"

at::Tensor cudaNaiveDST2D(const at::Tensor input, const uint points);
at::Tensor cudaNaiveIDST2D(const at::Tensor input, const uint points);

at::Tensor naiveDST2D(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveDST2D(padInput, points);
}

at::Tensor naiveIDST2D(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveIDST2D(padInput, points);
}
