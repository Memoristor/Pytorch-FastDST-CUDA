
#include "../include/utils.h"

at::Tensor cudaNaiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor cudaNaiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveDCT2D(padInput, points, sortbyZigzag);
}

at::Tensor naiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  CHECK_INPUT(input);
  if (recoverbyZigzag) {
    CHECK_TENSORDIM(input, 3);
  } else {
    CHECK_TENSORDIM(input, 2);
  }
  at::Tensor padInput = zeroPadInputTensorToFitPointSize(input, points);
  return cudaNaiveIDCT2D(padInput, points, recoverbyZigzag);
}