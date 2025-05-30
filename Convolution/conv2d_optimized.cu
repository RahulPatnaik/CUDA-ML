#include <iostream>
#include <cuda.h>
#include <cstdlib>

#define TILE_DIM 16
#define FILTER_RADIUS 1
#define IMG_WIDTH 64
#define IMG_HEIGHT 64

// Kernel declaration 
__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void conv2D_optimized(const float * __restrict__ N, float *P, int width, int height){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Shared memory with padding to avoid bank conflicts
    __shared__ float N_s[TILE_DIM + 2*FILTER_RADIUS][TILE_DIM + 2*FILTER_RADIUS + 1];

    // Position in shared memory
    int smem_row = ty + FILTER_RADIUS;
    int smem_col = tx + FILTER_RADIUS;

    // Load central tile with __ldg for read-only cache
    if(row < height && col < width){
        N_s[smem_row][smem_col] = __ldg(&N[row * width + col]);
    } else {
        N_s[smem_row][smem_col] = 0.0f;
    }

    // Load halos - use all threads to load halos more efficiently

    // Top halo
    if (ty < FILTER_RADIUS) {
        int halo_row = row - FILTER_RADIUS;
        int halo_col = col;
        if(halo_row >= 0 && halo_col < width)
            N_s[smem_row - FILTER_RADIUS][smem_col] = __ldg(&N[halo_row * width + halo_col]);
        else
            N_s[smem_row - FILTER_RADIUS][smem_col] = 0.0f;
    }

    // Bottom halo
    if (ty >= TILE_DIM - FILTER_RADIUS) {
        int halo_row = row + FILTER_RADIUS;
        int halo_col = col;
        if(halo_row < height && halo_col < width)
            N_s[smem_row + FILTER_RADIUS][smem_col] = __ldg(&N[halo_row * width + halo_col]);
        else
            N_s[smem_row + FILTER_RADIUS][smem_col] = 0.0f;
    }

    // Left halo
    if (tx < FILTER_RADIUS) {
        int halo_row = row;
        int halo_col = col - FILTER_RADIUS;
        if(halo_row < height && halo_col >= 0)
            N_s[smem_row][smem_col - FILTER_RADIUS] = __ldg(&N[halo_row * width + halo_col]);
        else
            N_s[smem_row][smem_col - FILTER_RADIUS] = 0.0f;
    }

    // Right halo
    if (tx >= TILE_DIM - FILTER_RADIUS) {
        int halo_row = row;
        int halo_col = col + FILTER_RADIUS;
        if(halo_row < height && halo_col < width)
            N_s[smem_row][smem_col + FILTER_RADIUS] = __ldg(&N[halo_row * width + halo_col]);
        else
            N_s[smem_row][smem_col + FILTER_RADIUS] = 0.0f;
    }

    // Load corners
    if(tx < FILTER_RADIUS && ty < FILTER_RADIUS){
        int r = row - FILTER_RADIUS;
        int c = col - FILTER_RADIUS;
        if(r >= 0 && c >= 0)
            N_s[smem_row - FILTER_RADIUS][smem_col - FILTER_RADIUS] = __ldg(&N[r * width + c]);
        else
            N_s[smem_row - FILTER_RADIUS][smem_col - FILTER_RADIUS] = 0.0f;
    }
    if(tx >= TILE_DIM - FILTER_RADIUS && ty < FILTER_RADIUS){
        int r = row - FILTER_RADIUS;
        int c = col + FILTER_RADIUS;
        if(r >= 0 && c < width)
            N_s[smem_row - FILTER_RADIUS][smem_col + FILTER_RADIUS] = __ldg(&N[r * width + c]);
        else
            N_s[smem_row - FILTER_RADIUS][smem_col + FILTER_RADIUS] = 0.0f;
    }
    if(tx < FILTER_RADIUS && ty >= TILE_DIM - FILTER_RADIUS){
        int r = row + FILTER_RADIUS;
        int c = col - FILTER_RADIUS;
        if(r < height && c >= 0)
            N_s[smem_row + FILTER_RADIUS][smem_col - FILTER_RADIUS] = __ldg(&N[r * width + c]);
        else
            N_s[smem_row + FILTER_RADIUS][smem_col - FILTER_RADIUS] = 0.0f;
    }
    if(tx >= TILE_DIM - FILTER_RADIUS && ty >= TILE_DIM - FILTER_RADIUS){
        int r = row + FILTER_RADIUS;
        int c = col + FILTER_RADIUS;
        if(r < height && c < width)
            N_s[smem_row + FILTER_RADIUS][smem_col + FILTER_RADIUS] = __ldg(&N[r * width + c]);
        else
            N_s[smem_row + FILTER_RADIUS][smem_col + FILTER_RADIUS] = 0.0f;
    }

    __syncthreads(); // wait for shared memory loading

    // Perform convolution
    float Pvalue = 0.0f;

    if(row < height && col < width){
        #pragma unroll
        for(int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++){
            #pragma unroll
            for(int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++){
                Pvalue += F_c[FILTER_RADIUS + fRow][FILTER_RADIUS + fCol] * 
                          N_s[smem_row + fRow][smem_col + fCol];
            }
        }
        P[row * width + col] = Pvalue;
    }
}

// Host code
int main() {
    const int img_size = IMG_WIDTH * IMG_HEIGHT;
    const int filter_size = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);

    // Allocate host memory
    float *h_input = new float[img_size];
    float *h_output = new float[img_size];
    float h_filter[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

    // Initialize input with grayscale ramp
    for (int i = 0; i < img_size; i++) {
        h_input[i] = static_cast<float>(i % 256);
    }

    // Initialize filter with all ones (simple box blur)
    for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            h_filter[i + FILTER_RADIUS][j + FILTER_RADIUS] = 1.0f;
        }
    }

    // Normalize filter
    for (int i = 0; i < 2 * FILTER_RADIUS + 1; i++) {
        for (int j = 0; j < 2 * FILTER_RADIUS + 1; j++) {
            h_filter[i][j] /= filter_size;
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, img_size * sizeof(float));
    cudaMalloc((void**)&d_output, img_size * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, img_size * sizeof(float), cudaMemcpyHostToDevice);

    // Copy filter to constant memory
    cudaMemcpyToSymbol(F_c, h_filter, sizeof(h_filter));

    // Grid and block setup
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((IMG_WIDTH + TILE_DIM - 1) / TILE_DIM, (IMG_HEIGHT + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    conv2D_optimized<<<gridDim, blockDim>>>(d_input, d_output, IMG_WIDTH, IMG_HEIGHT);

    // Copy result back
    cudaMemcpy(h_output, d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output sample
    std::cout << "Output sample (5x5):" << std::endl;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            std::cout << h_output[i * IMG_WIDTH + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
