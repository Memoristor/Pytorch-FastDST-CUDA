
#include "../include/utils.h"

namespace F = torch::nn::functional;

at::Tensor zeroPadInputTensorToFitPointSize(const at::Tensor input, const uint numPoints) {
    auto inputSize = input.sizes();

    int padh = int((inputSize[2] + numPoints - 1) / numPoints) * numPoints - inputSize[2];
    int padw = int((inputSize[3] + numPoints - 1) / numPoints) * numPoints - inputSize[3];
    int padtop = int(padh / 2);
    int padbtm = padh - padtop;
    int padlft = int(padw / 2);
    int padrgt = padw - padlft;

    return F::pad(input, F::PadFuncOptions({padlft, padrgt, padtop, padbtm}).mode(torch::kConstant));
}

void optimalCUDABlocksAndThreadsPerBlock(const uint numTotalThreads, dim3 &numBlocks, dim3 &threadsPerBlock) {
    if (numTotalThreads <= 1024) {
        threadsPerBlock.x = numTotalThreads;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;

        numBlocks.x = 1;
        numBlocks.y = 1;
        numBlocks.z = 1;
    } else {
        threadsPerBlock.x = 1024;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;

        numBlocks.x = int((numTotalThreads + 1024 - 1) / 1024);
        numBlocks.y = 1;
        numBlocks.z = 1;

        if (numBlocks.x > 1024) {
            numBlocks.y = int((numBlocks.x + 1024 - 1) / 1024);
            numBlocks.x = 1024;

            if (numBlocks.y > 1024) {
                numBlocks.z = int((numBlocks.y + 1024 - 1) / 1024);
                numBlocks.y = 1024;
            }
        }
    }
}

void calculateZigzag(uint *zigzag, uint numPoints)
{
	uint isX = 0;
	uint t = 0, i = 0, d = numPoints * numPoints, xy[2] = {0, 0};
	uint mid = numPoints * numPoints / 2 + numPoints % 2;
 
	while (d != mid)
	{
		if (xy[isX] < i)
			xy[isX]++;

		if (xy[!isX] > 0)
			xy[!isX]--;
 
		if (xy[isX] == i)
		{
			i++;
			isX = !isX;
		}
 
		zigzag[xy[0] * numPoints + xy[1]] = t++;
		zigzag[(numPoints - 1 - xy[0]) * numPoints + (numPoints - 1 - xy[1])] = --d;
	}
}