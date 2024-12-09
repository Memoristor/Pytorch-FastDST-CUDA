
#include "../include/utils.h"

at::Tensor cudaNaiveDHT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor cudaNaiveIDHT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDHT2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveDHT2D(padInput, points, sortbyZigzag);
}

at::Tensor naiveIDHT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  CHECK_INPUT(input);
  if (recoverbyZigzag) {
    CHECK_TENSORDIM(input, 3);
  } else {
    CHECK_TENSORDIM(input, 2);
  }
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveIDHT2D(padInput, points, recoverbyZigzag);
}