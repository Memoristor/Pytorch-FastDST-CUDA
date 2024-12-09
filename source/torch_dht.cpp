
#include "../include/utils.h"

at::Tensor cudaNaiveDHT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor cudaNaiveIDHT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDHT2D(const at::Tensor input, const uint points, const bool sortbyZigzag) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_TENSORDIM(input, 2);
  CHECK_EVENDIV(input, -1, points);
  CHECK_EVENDIV(input, -2, points);
  return cudaNaiveDHT2D(input, points, sortbyZigzag);
}

at::Tensor naiveIDHT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  if (recoverbyZigzag) {
    CHECK_TENSORDIM(input, 3);
    CHECK_DIMEQUAL(input, -3, points * points);
  } else {
    CHECK_TENSORDIM(input, 2);
    CHECK_EVENDIV(input, -1, points);
    CHECK_EVENDIV(input, -2, points);
  }
  return cudaNaiveIDHT2D(input, points, recoverbyZigzag);
}