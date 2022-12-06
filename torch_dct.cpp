
#include <iostream>
#include "utils.h"

namespace F = torch::nn::functional;

at::Tensor cudaNativeDCTII2DForward(const at::Tensor input, const uint32_t psize);


at::Tensor nativeDCTII2DForward(const at::Tensor input, const uint32_t psize, const bool sort_freq) {
    CHECK_4DTENSOR(input)

    at::IntList inpsize = input.sizes();

    int padh = int((inpsize[2] + psize - 1) / psize) * psize - inpsize[2];
    int padw = int((inpsize[3] + psize - 1) / psize) * psize - inpsize[3];
    int padtop = int(padh / 2);
    int padbtm = padh - padtop;
    int padlft = int(padw / 2);
    int padrgt = padw - padlft;

    at::Tensor padout = F::pad(input, F::PadFuncOptions({padlft, padrgt, padtop, padbtm}).mode(torch::kConstant));

    return cudaNativeDCTII2DForward(padout, psize);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nativeDCTII2DForward", &nativeDCTII2DForward, "Native 2D DCT-II forward (CUDA)");
}
