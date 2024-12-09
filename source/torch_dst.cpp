
#include "../include/utils.h"

at::Tensor cudaNaiveDST2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor cudaNaiveIDST2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDST2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveDST2D(padInput, points, sortbyZigzag);
}

at::Tensor naiveIDST2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  CHECK_INPUT(input);
  if (recoverbyZigzag) {
    CHECK_TENSORDIM(input, 3);
  } else {
    CHECK_TENSORDIM(input, 2);
  }
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveIDST2D(padInput, points, recoverbyZigzag);
}