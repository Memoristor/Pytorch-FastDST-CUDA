
#include <iostream>

#include "../include/utils.h"

at::Tensor cudaSortCoefficientsByZigzag(const at::Tensor input, const uint numPoints, const uint priority);
at::Tensor cudaRecoverCoefficientsByZigzag(const at::Tensor input, const uint numPoints, const uint priority);

at::Tensor sortCoefficients(const at::Tensor input, const uint numPoints, const uint priority) {
    CHECK_INPUT(input);
    return cudaSortCoefficientsByZigzag(input, numPoints, priority);
}

at::Tensor recoverCoefficients(const at::Tensor input, const uint numPoints, const uint priority) {
    CHECK_INPUT(input);
    return cudaRecoverCoefficientsByZigzag(input, numPoints, priority);
}
