
#include "../include/utils.h"

namespace F = torch::nn::functional;

at::Tensor zeroPadInputTensorToFitDCTPointSize(const at::Tensor input, const uint numPoints) {
    auto inputSize = input.sizes();

    int padh = int((inputSize[2] + numPoints - 1) / numPoints) * numPoints - inputSize[2];
    int padw = int((inputSize[3] + numPoints - 1) / numPoints) * numPoints - inputSize[3];
    int padtop = int(padh / 2);
    int padbtm = padh - padtop;
    int padlft = int(padw / 2);
    int padrgt = padw - padlft;

    return F::pad(input, F::PadFuncOptions({padlft, padrgt, padtop, padbtm}).mode(torch::kConstant));
}

void optimalCUDABlocksAndThreads(const uint numTotalThreads, dim3 &numBlocks, dim3 &threadsPerBlock) {
    if (numTotalThreads <= 1024) {
        numBlocks.x = 1;
        numBlocks.y = 1;
        numBlocks.z = 1;

        threadsPerBlock.x = numTotalThreads;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;
    } else {

    }
}