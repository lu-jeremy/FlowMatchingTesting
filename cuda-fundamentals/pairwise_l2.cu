#include <cuda_runtime.h>
#include <cuda/std/limits>

#include <algorithm>
#include <iostream>

__global__ void match_dist(const float * __restrict__ x, const float * __restrict__ y,
        float *output, int b, int m, int n) {
    float min_dist = cuda::std::numeric_limits<float>::min();

    //__shared__ float shm[128][3];

    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int j = threadIdx.x; j < m; j += blockDim.x) {
            //shm[j][0] = x[i * b + j * 3];
            //shm[j][1] = x[i * b + j * 3 + 1];
            //shm[j][2] = x[i * b + j * 3 + 2];
            //}

            //__syncthreads();

            float dist = 0.0;

            for (int k = threadIdx.y; k < n; k += blockDim.y) {
                float dx = y[i * b + k * 3] - x[i * b + j * 3]; 
                float dy = y[i * b + k * 3 + 1] - x[i * b + j * 3 + 1]; 
                float dz = y[i * b + k * 3 + 2] - x[i * b + j * 3 + 2];

                dist = dx * dx + dy * dy + dz * dz;
                min_dist = fminf(dist, min_dist);  // warp divergence
            }

            min_dist = fminf(dist, min_dist);
        }
    }

    *output = min_dist;
}

int main() {
    int b = 1;
    int m = 1;
    int n = 1;

    const float x[1][2][3] = {{{0, 0, 0}, {0, 0, 0}}};
    const float y[1][2][3] = {{{0, 0, 0}, {0, 0, 0}}};
    float *output;

    float *d_x;
    float *d_y;
    float *d_output;
    
    cudaMalloc(&d_x, b * m * 3 * sizeof(float));
    cudaMalloc(&d_y, b * n * 3 * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_x, x, b * m * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, b * n * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int blockDim = 128;
    int blockSize = max(b * m, b * n);

    match_dist<<<blockDim, blockSize>>>(d_x, d_y, d_output, b, m, n); 

    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_output);

    std::cout << *output << '\n';
}
