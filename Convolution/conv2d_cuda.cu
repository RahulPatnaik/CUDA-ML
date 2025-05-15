// conv2d_cuda.cu
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_DIM       16
#define FILTER_RADIUS  1

// constant‐memory filter
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void conv2d_kernel(const float* __restrict__ N,
                              float* __restrict__ P,
                              int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int row = by*TILE_DIM + ty, col = bx*TILE_DIM + tx;

    __shared__ float N_s[TILE_DIM+2*FILTER_RADIUS][TILE_DIM+2*FILTER_RADIUS+1];
    int sm_r = ty + FILTER_RADIUS, sm_c = tx + FILTER_RADIUS;

    // center
    if(row<height && col<width)
      N_s[sm_r][sm_c] = __ldg(&N[row*width + col]);
    else
      N_s[sm_r][sm_c] = 0.f;

    // halos (top/bottom/left/right + corners)
    if(ty < FILTER_RADIUS) {
        int r=row-FILTER_RADIUS, c=col;
        N_s[sm_r-FILTER_RADIUS][sm_c] =
          (r>=0 && c<width)? __ldg(&N[r*width + c]) : 0.f;
    }
    if(ty>=TILE_DIM-FILTER_RADIUS) {
        int r=row+FILTER_RADIUS, c=col;
        N_s[sm_r+FILTER_RADIUS][sm_c] =
          (r<height && c<width)? __ldg(&N[r*width + c]) : 0.f;
    }
    if(tx < FILTER_RADIUS) {
        int r=row, c=col-FILTER_RADIUS;
        N_s[sm_r][sm_c-FILTER_RADIUS] =
          (r<height && c>=0)? __ldg(&N[r*width + c]) : 0.f;
    }
    if(tx>=TILE_DIM-FILTER_RADIUS) {
        int r=row, c=col+FILTER_RADIUS;
        N_s[sm_r][sm_c+FILTER_RADIUS] =
          (r<height && c<width)? __ldg(&N[r*width + c]) : 0.f;
    }
    // corners
    if(tx<FILTER_RADIUS && ty<FILTER_RADIUS){
        int r=row-FILTER_RADIUS, c=col-FILTER_RADIUS;
        N_s[sm_r-FILTER_RADIUS][sm_c-FILTER_RADIUS] =
          (r>=0&&c>=0)? __ldg(&N[r*width + c]) : 0.f;
    }
    if(tx>=TILE_DIM-FILTER_RADIUS && ty<FILTER_RADIUS){
        int r=row-FILTER_RADIUS, c=col+FILTER_RADIUS;
        N_s[sm_r-FILTER_RADIUS][sm_c+FILTER_RADIUS] =
          (r>=0&&c<width)? __ldg(&N[r*width + c]) : 0.f;
    }
    if(tx<FILTER_RADIUS && ty>=TILE_DIM-FILTER_RADIUS){
        int r=row+FILTER_RADIUS, c=col-FILTER_RADIUS;
        N_s[sm_r+FILTER_RADIUS][sm_c-FILTER_RADIUS] =
          (r<height&&c>=0)? __ldg(&N[r*width + c]) : 0.f;
    }
    if(tx>=TILE_DIM-FILTER_RADIUS && ty>=TILE_DIM-FILTER_RADIUS){
        int r=row+FILTER_RADIUS, c=col+FILTER_RADIUS;
        N_s[sm_r+FILTER_RADIUS][sm_c+FILTER_RADIUS] =
          (r<height&&c<width)? __ldg(&N[r*width + c]) : 0.f;
    }

    __syncthreads();

    // convolution
    if(row<height && col<width) {
      float acc = 0.f;
      #pragma unroll
      for(int fy=-FILTER_RADIUS; fy<=FILTER_RADIUS; ++fy){
        #pragma unroll
        for(int fx=-FILTER_RADIUS; fx<=FILTER_RADIUS; ++fx){
          acc += F_c[FILTER_RADIUS+fy][FILTER_RADIUS+fx] *
                 N_s[sm_r+fy][sm_c+fx];
        }
      }
      P[row*width + col] = acc;
    }
}

torch::Tensor conv2d_forward(torch::Tensor input,
                             torch::Tensor filter) {
  TORCH_CHECK(input.dim()==2,  "input must be H×W");
  TORCH_CHECK(filter.dim()==2 && 
              filter.size(0)==2*FILTER_RADIUS+1 &&
              filter.size(1)==2*FILTER_RADIUS+1,
              "filter must be (2R+1)x(2R+1)");

  int H = input.size(0), W = input.size(1);
  auto output = torch::empty({H,W}, input.options());

  // copy filter into constant memory
  cudaMemcpyToSymbol(F_c,
                     filter.data_ptr<float>(),
                     sizeof(float)*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1));

  // launch kernel
  dim3 block(TILE_DIM, TILE_DIM),
       grid((W+TILE_DIM-1)/TILE_DIM,
            (H+TILE_DIM-1)/TILE_DIM);
  conv2d_kernel<<<grid,block>>>(
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    W, H
  );
  cudaDeviceSynchronize();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d_forward", &conv2d_forward, "CUDA 2D convolution");
}
