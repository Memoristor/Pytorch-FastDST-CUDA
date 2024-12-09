#ifndef __DEVICE_KERNEL_CUH
#define __DEVICE_KERNEL_CUH

#include "utils.h"

__device__ __forceinline__ void initZigzag(uint* zigzag, uint points) {
  uint row = 0, col = 0;
  uint goingUp = 1;  // 1: going up, 0: going down

  for (uint i = 0; i < points * points; ++i) {
    zigzag[i] = row * points + col;

    if (goingUp) {
      if (col == points - 1) {
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
      if (row == points - 1) {
        // Reach the down-most boundary and turn upward
        col++;
        goingUp = 1;
      } else if (col == 0) {
        // Reach the left-most boundary and turn upward
        row++;
        goingUp = 1;
      } else {
        // Continue going down
        row++;
        col--;
      }
    }
  }
}

#endif /* __DEVICE_KERNEL_CUH */