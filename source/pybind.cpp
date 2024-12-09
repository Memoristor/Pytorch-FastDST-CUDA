
#include <pybind11/pybind11.h>

#include "../include/utils.h"

namespace py = pybind11;

at::Tensor naiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor naiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDST2D(const at::Tensor input, const uint points);
at::Tensor naiveIDST2D(const at::Tensor input, const uint points);

at::Tensor naiveDHT2D(const at::Tensor input, const uint points);
at::Tensor naiveIDHT2D(const at::Tensor input, const uint points);

at::Tensor sortCoefficients(const at::Tensor input, const uint points);
at::Tensor recoverCoefficients(const at::Tensor input, const uint points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Fast Discrete Signal Transform (FastDST) library";

  m.def("DCT2d", &naiveDCT2D,
        R"doc(
          Calculate 2D block-wise DCT

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than two.
              points (float): The number of points for the DCT block.
              sortbyZigzag (bool): Whether to integrate the frequency coefficients from low 
                to high frequency into a new dimension and place it as the third-to-last 
                dimension. Default value is true.

          Returns:
              torch.Tensor: The calculated DCT coefficients.

          Examples:
              >> x = torch.randn(3, 1024, 2048).cuda()
              >> dct = fadst.DCT2d(x, 4, True)  # shape: [3, 16, 256, 512]
              >>
              >> dct = fadst.DCT2d(x, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("sortbyZigzag") = true);

  m.def("IDCT2d", &naiveIDCT2D,
        R"doc(
          Calculate 2D block-wise IDCT

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than three if
                `recoverbyZigzag` is true, and a tensor with dimensions greater than two otherwise.
              points (float): The number of points for the DCT block.
              recoverbyZigzag (bool): Whether the input DCT coefficients have already been sorted 
                from low frequency to high frequency. Default value is true.

          Returns:
              torch.Tensor: The calculated IDCT result.

          Examples:
              >> dct = torch.randn(3, 16, 256, 512).cuda()
              >> y = fadst.IDCT2d(x, 4, True)  # shape: [3, 1024, 2048]
              >>
              >> dct = torch.randn(3, 1024, 2048).cuda()
              >> y = fadst.IDCT2d(x, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("recoverbyZigzag") = true);

  m.def("naiveDST2D", &naiveDST2D, "Naive 2D-DST");
  m.def("naiveIDST2D", &naiveIDST2D, "Naive 2D-IDST");

  m.def("naiveDHT2D", &naiveDHT2D, "Naive 2D-DHT");
  m.def("naiveIDHT2D", &naiveIDHT2D, "Naive 2D-IDHT");

  m.def("sortCoefficients", &sortCoefficients, "Sort Coeff");
  m.def("recoverCoefficients", &recoverCoefficients, "Recover Coeff");
}
