
#include <pybind11/pybind11.h>

#include "../include/utils.h"

namespace py = pybind11;

at::Tensor naiveDCT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor naiveIDCT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDST2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor naiveIDST2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

at::Tensor naiveDHT2D(const at::Tensor input, const uint points, const bool sortbyZigzag);
at::Tensor naiveIDHT2D(const at::Tensor input, const uint points, const bool recoverbyZigzag);

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
                `recoverbyZigzag` is true, and a tensor with dimensions greater than two
                otherwise.
              points (float): The number of points for the DCT block.
              recoverbyZigzag (bool): Whether the input DCT coefficients have already been sorted
                from low frequency to high frequency. Default value is true.

          Returns:
              torch.Tensor: The calculated IDCT result.

          Examples:
              >> dct = torch.randn(3, 16, 256, 512).cuda()
              >> y = fadst.IDCT2d(dct, 4, True)  # shape: [3, 1024, 2048]
              >>
              >> dct = torch.randn(3, 1024, 2048).cuda()
              >> y = fadst.IDCT2d(dct, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("recoverbyZigzag") = true);

  m.def("DHT2d", &naiveDHT2D,
        R"doc(
          Calculate 2D block-wise DHT

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than two.
              points (float): The number of points for the DHT block.
              sortbyZigzag (bool): Whether to integrate the frequency coefficients from low
                to high frequency into a new dimension and place it as the third-to-last
                dimension. Default value is true.

          Returns:
              torch.Tensor: The calculated DHT coefficients.

          Examples:
              >> x = torch.randn(3, 1024, 2048).cuda()
              >> dht = fadst.DHT2d(x, 4, True)  # shape: [3, 16, 256, 512]
              >>
              >> dht = fadst.DHT2d(x, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("sortbyZigzag") = true);

  m.def("IDHT2d", &naiveIDHT2D,
        R"doc(
          Calculate 2D block-wise IDHT

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than three if
                `recoverbyZigzag` is true, and a tensor with dimensions greater than two
                otherwise.
              points (float): The number of points for the DHT block.
              recoverbyZigzag (bool): Whether the input DHT coefficients have already been sorted
                from low frequency to high frequency. Default value is true.

          Returns:
              torch.Tensor: The calculated IDHT result.

          Examples:
              >> dht = torch.randn(3, 16, 256, 512).cuda()
              >> y = fadst.IDHT2d(dht, 4, True)  # shape: [3, 1024, 2048]
              >>
              >> dht = torch.randn(3, 1024, 2048).cuda()
              >> y = fadst.IDHT2d(dht, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("recoverbyZigzag") = true);

  m.def("DST2d", &naiveDST2D,
        R"doc(
          Calculate 2D block-wise DST

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than two.
              points (float): The number of points for the DST block.
              sortbyZigzag (bool): Whether to integrate the frequency coefficients from low
                to high frequency into a new dimension and place it as the third-to-last
                dimension. Default value is true.

          Returns:
              torch.Tensor: The calculated DST coefficients.

          Examples:
              >> x = torch.randn(3, 1024, 2048).cuda()
              >> dst = fadst.DST2d(x, 4, True)  # shape: [3, 16, 256, 512]
              >>
              >> dst = fadst.DST2d(x, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("sortbyZigzag") = true);

  m.def("IDST2d", &naiveIDST2D,
        R"doc(
          Calculate 2D block-wise IDST

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than three if
                `recoverbyZigzag` is true, and a tensor with dimensions greater than two
                otherwise.
              points (float): The number of points for the DST block.
              recoverbyZigzag (bool): Whether the input DST coefficients have already been sorted
                from low frequency to high frequency. Default value is true.

          Returns:
              torch.Tensor: The calculated IDST result.

          Examples:
              >> dst = torch.randn(3, 16, 256, 512).cuda()
              >> y = fadst.IDST2d(dst, 4, True)  # shape: [3, 1024, 2048]
              >>
              >> dst = torch.randn(3, 1024, 2048).cuda()
              >> y = fadst.IDST2d(dst, 4, False)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"), py::arg("recoverbyZigzag") = true);

  m.def("sort2d", &sortCoefficients,
        R"doc(
          Rearrange the tensor of frequency coefficients from low frequency to high frequency.

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than two.
              points (float): The number of points for the frequency block.

          Returns:
              torch.Tensor: A tensor whose third-to-last dimension is arranged from low to high
              frequency.

          Examples:
              >> x = torch.randn(3, 1024, 2048).cuda()
              >> dct = fadst.DCT2d(x, 4, False)  # shape: [3, 1024, 2048]
              >> y = fadst.sort2d(dct, 4)  # shape: [3, 16, 256, 512]
              >>
          )doc",
        py::arg("input"), py::arg("points"));

  m.def("recover2d", &recoverCoefficients,
        R"doc(
          Recover a tensor whose third-to-last dimension is arranged from low frequency to high
          frequency back to its original unsorted state.

          Parameters:
              input (torch.Tensor): An input tensor with dimensions greater than three.
              points (float): The number of points for the frequency block.

          Returns:
              torch.Tensor: A tensor in its unsorted state without any frequency-based sorting.

          Examples:
              >> x = torch.randn(3, 1024, 2048).cuda()
              >> dct = fadst.DCT2d(x, 4, True)  # shape: [3, 16, 256, 512]
              >> y = fadst.recover2d(dct, 4)  # shape: [3, 1024, 2048]
              >>
          )doc",
        py::arg("input"), py::arg("points"));
}
