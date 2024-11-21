
#include "../../include/utils.h"

at::Tensor cudaSortCoefficientsByZigzag(const at::Tensor input, const uint points);
at::Tensor cudaRecoverCoefficientsByZigzag(const at::Tensor input, const uint points);

at::Tensor sortCoefficients(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 2);
  return cudaSortCoefficientsByZigzag(input, points);
}

at::Tensor recoverCoefficients(const at::Tensor input, const uint points) {
  CHECK_INPUT(input);
  CHECK_TENSORDIM(input, 3);
  return cudaRecoverCoefficientsByZigzag(input, points);
}
