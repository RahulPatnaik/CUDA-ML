#define TILE_DIM 32
#define FILTER_RADIUS 1
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void conv2D(float *N, float *P, int width, int height){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Shared memory with halo padding
    __shared__ float N_s[TILE_DIM + 2*FILTER_RADIUS][TILE_DIM + 2*FILTER_RADIUS];

    // Position in shared memory
    int smem_row = ty + FILTER_RADIUS;
    int smem_col = tx + FILTER_RADIUS;

    // Load central tile
    if(row < height && col < width){
        N_s[smem_row][smem_col] = N[row * width + col];
    } else {
        N_s[smem_row][smem_col] = 0.0f;
    }

    // Load top halo
    if(ty < FILTER_RADIUS) {
        int halo_row = row - FILTER_RADIUS;
        int halo_col = col;
        N_s[smem_row - FILTER_RADIUS][smem_col] =
            (halo_row >= 0 && halo_col < width) ? N[halo_row * width + halo_col] : 0.0f;
    }

    // Load bottom halo
    if(ty >= TILE_DIM - FILTER_RADIUS) {
        int halo_row = row + FILTER_RADIUS;
        int halo_col = col;
        N_s[smem_row + FILTER_RADIUS][smem_col] =
            (halo_row < height && halo_col < width) ? N[halo_row * width + halo_col] : 0.0f;
    }

    // Load left halo
    if(tx < FILTER_RADIUS) {
        int halo_row = row;
        int halo_col = col - FILTER_RADIUS;
        N_s[smem_row][smem_col - FILTER_RADIUS] =
            (halo_row < height && halo_col >= 0) ? N[halo_row * width + halo_col] : 0.0f;
    }

    // Load right halo
    if(tx >= TILE_DIM - FILTER_RADIUS) {
        int halo_row = row;
        int halo_col = col + FILTER_RADIUS;
        N_s[smem_row][smem_col + FILTER_RADIUS] =
            (halo_row < height && halo_col < width) ? N[halo_row * width + halo_col] : 0.0f;
    }

    // Load corners
    if(tx < FILTER_RADIUS && ty < FILTER_RADIUS) {
        int r = row - FILTER_RADIUS;
        int c = col - FILTER_RADIUS;
        N_s[smem_row - FILTER_RADIUS][smem_col - FILTER_RADIUS] =
            (r >= 0 && c >= 0) ? N[r * width + c] : 0.0f;
    }

    if(tx >= TILE_DIM - FILTER_RADIUS && ty < FILTER_RADIUS) {
        int r = row - FILTER_RADIUS;
        int c = col + FILTER_RADIUS;
        N_s[smem_row - FILTER_RADIUS][smem_col + FILTER_RADIUS] =
            (r >= 0 && c < width) ? N[r * width + c] : 0.0f;
    }

    if(tx < FILTER_RADIUS && ty >= TILE_DIM - FILTER_RADIUS) {
        int r = row + FILTER_RADIUS;
        int c = col - FILTER_RADIUS;
        N_s[smem_row + FILTER_RADIUS][smem_col - FILTER_RADIUS] =
            (r < height && c >= 0) ? N[r * width + c] : 0.0f;
    }

    if(tx >= TILE_DIM - FILTER_RADIUS && ty >= TILE_DIM - FILTER_RADIUS) {
        int r = row + FILTER_RADIUS;
        int c = col + FILTER_RADIUS;
        N_s[smem_row + FILTER_RADIUS][smem_col + FILTER_RADIUS] =
            (r < height && c < width) ? N[r * width + c] : 0.0f;
    }

    __syncthreads(); // ensure shared memory is fully loaded

    // Perform convolution
    float Pvalue = 0.0f;
    if(row < height && col < width){
        for(int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++){
            for(int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++){
                Pvalue += F_c[FILTER_RADIUS + fRow][FILTER_RADIUS + fCol] *
                          N_s[smem_row + fRow][smem_col + fCol];
            }
        }
        P[row * width + col] = Pvalue;
    }
}
