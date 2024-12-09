
#include "../include/utils.h"

at::Tensor cudaNaiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor cudaNaiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_TENSORDIM(input, 2);
  CHECK_EVENDIV(input, -1, points);
  CHECK_EVENDIV(input, -2, points);
  return cudaNaiveDCT2D(input, points, sortbyZigzag);
}

at::Tensor naiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_EVENDIV(input, -1, points);
  CHECK_EVENDIV(input, -2, points);
  if (recoverbyZigzag) {
    CHECK_TENSORDIM(input, 3);
    CHECK_DIMEQUAL(input, -3, points * points);
  } else {
    CHECK_TENSORDIM(input, 2);
  }
  return cudaNaiveIDCT2D(input, points, recoverbyZigzag);
}