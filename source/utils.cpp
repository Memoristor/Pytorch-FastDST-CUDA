
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

void calculateZigzag(uint32_t *zigzag, uint32_t numPoints)
{
	int row = 0, col = 0;
    int goingUp = 1; // 1: going up, 0: going down

    for (int i = 0; i < numPoints * numPoints; ++i) {
        zigzag[i] = row * numPoints + col;

        if (goingUp) {
            if (col == numPoints - 1) { 
                // Reach the right-most boundary and turn downward
				row++;
                goingUp = 0;
            } else if (row == 0) { 
				// Reach the top-most boundary and turn downward
                col++;
                goingUp = 0;
            } else { 
				// Continue going up
                row--;
                col++;
            }
        } else {
            if (row == numPoints - 1) {
				// Reach the down-most boundary and turn upward
                col++;
                goingUp = 1;
            } else if (col == 0) {
				// Reach the left-most boundary and turn upward
                row++;
                goingUp = 1;
            } else { 
				//Continue going down
                row++;
                col--;
            }
        }
    }
}