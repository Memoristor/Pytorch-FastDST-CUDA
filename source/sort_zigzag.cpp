
#include "../include/utils.h"

at::Tensor cudaSortCoefficientsByZigzag(const at::Tensor input, const uint points);
at::Tensor cudaRecoverCoefficientsByZigzag(const at::Tensor input, const uint points);

at::Tensor sortCoefficients(const at::Tensor input, const uint points) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_TENSORDIM(input, 2);
  CHECK_EVENDIV(input, -1, points);
  CHECK_EVENDIV(input, -2, points);
  return cudaSortCoefficientsByZigzag(input, points);
}

at::Tensor recoverCoefficients(const at::Tensor input, const uint points) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_TENSORDIM(input, 3);
  CHECK_DIMEQUAL(input, -3, points * points);
  return cudaRecoverCoefficientsByZigzag(input, points);
}
