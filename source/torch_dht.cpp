
#include "../include/utils.h"

at::Tensor cudaNaiveDHT2D(const at::Tensor input, const uint points);
at::Tensor cudaNaiveIDHT2D(const at::Tensor input, const uint points);

at::Tensor naiveDHT2D(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveDHT2D(padInput, points);
}

at::Tensor naiveIDHT2D(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveIDHT2D(padInput, points);
}
