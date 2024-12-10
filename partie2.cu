#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define KERNEL_SIZE 5
#define NUM_KERNELS 6
#define POOL_SIZE 2

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            //M[i*p + j] = (float)(round((double)rand()/RAND_MAX  * 20 - 20));
            M[i*p + j] = (float)((double)rand() / RAND_MAX);
        }
    }
}

__global__ void cudaConvolution2D(float *input, float *kernel, float *output, int input_width, int input_height, int kernel_size, int output_width, int output_height, int num_kernels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        int kernel_radius = kernel_size / 2;
        for (int k = 0; k < num_kernels; k++) {
            float value = 0.0f;
            for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < input_width && iy >= 0 && iy < input_height) {
                        value += input[(iy * input_width) + ix] * kernel[(k * kernel_size * kernel_size) + ((ky + kernel_radius) * kernel_size) + (kx + kernel_radius)];
                    }
                }
            }
            output[(k * output_width * output_height) + (y * output_width) + x] = value;
        }
    }
}


__global__ void cudaPooling2D(float *input, float *output, int input_width, int input_height, int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        float sum = 0.0f;
        int count = 0;
        for (int ky = 0; ky < 2; ky++) {
            for (int kx = 0; kx < 2; kx++) {
                int ix = x * 2 + kx;
                int iy = y * 2 + ky;
                if (ix < input_width && iy < input_height) {
                    sum += input[iy * input_width + ix];
                    count++;
                }
            }
        }
        output[y * output_width + x] = sum / count;
    }
}

void initializeData(float *data, int size, bool random = true) {
    for (int i = 0; i < size; i++) {
        data[i] = random ? static_cast<float>(rand()) / RAND_MAX : 0.0f;
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%10.2f ", M[i*p + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    //int nn=atoi(argv[1]);
    //int pp=atoi(argv[1]);
    srand(time(NULL)); // Initialisation du générateur de nombres aléatoires
    const int n = 32; //nn;//10000;
    const int p = 32; //pp;//10000;
    const int maxaff = 10;

    float* A, *B, *C;
    A = (float*)malloc(n*p*sizeof(float));
    B = (float*)malloc(n*p*sizeof(float));
    C = (float*)malloc(n*p*sizeof(float));

    MatrixInit(A, 5, 5);
    MatrixInit(B, 1, 1);

    MatrixPrint(A, 5, 5);
    MatrixPrint(B, 1, 1);
    
    free(A);
    free(B);
    free(C);

    cudaDeviceSynchronize();
    return 0;
}


















// Taille de l'image et des noyaux (à adapter)




/*
int main() {
    // Allocation de la mémoire pour les matrices
    float *d_input, *d_kernel, *d_output;
    float *h_input = new float[INPUT_WIDTH * INPUT_HEIGHT];
    float *h_kernel = new float[NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE];
    float *h_output = new float[NUM_KERNELS * (INPUT_WIDTH - KERNEL_SIZE + 1) * (INPUT_HEIGHT - KERNEL_SIZE + 1)];

    // Initialiser les données
    initializeData(h_input, INPUT_WIDTH * INPUT_HEIGHT);
    initializeData(h_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE);

    // Allouer de la mémoire sur le GPU
    cudaMalloc((void **)&d_input, INPUT_WIDTH * INPUT_HEIGHT * sizeof(float));
    cudaMalloc((void **)&d_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output, NUM_KERNELS * (INPUT_WIDTH - KERNEL_SIZE + 1) * (INPUT_HEIGHT - KERNEL_SIZE + 1) * sizeof(float));

    // Copier les données vers le GPU
    cudaMemcpy(d_input, h_input, INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Dimension des blocs et de la grille
    dim3 blockDim(16, 16);
    dim3 gridDim((INPUT_WIDTH - KERNEL_SIZE + 1 + blockDim.x - 1) / blockDim.x, 
                 (INPUT_HEIGHT - KERNEL_SIZE + 1 + blockDim.y - 1) / blockDim.y);

    // Lancer la convolution 2D
    cudaConvolution2D<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, INPUT_WIDTH, INPUT_HEIGHT, KERNEL_SIZE, 
                                              INPUT_WIDTH - KERNEL_SIZE + 1, INPUT_HEIGHT - KERNEL_SIZE + 1, NUM_KERNELS);
    cudaDeviceSynchronize();

    // Lancer le sous-échantillonnage 2D
    dim3 poolGridDim((INPUT_WIDTH / POOL_SIZE + blockDim.x - 1) / blockDim.x, 
                     (INPUT_HEIGHT / POOL_SIZE + blockDim.y - 1) / blockDim.y);

    cudaPooling2D<<<poolGridDim, blockDim>>>(d_output, d_output, INPUT_WIDTH - KERNEL_SIZE + 1, INPUT_HEIGHT - KERNEL_SIZE + 1, 
                                              (INPUT_WIDTH - KERNEL_SIZE + 1) / 2, (INPUT_HEIGHT - KERNEL_SIZE + 1) / 2);
    cudaDeviceSynchronize();

    // Copier les résultats du GPU vers le CPU
    cudaMemcpy(h_output, d_output, NUM_KERNELS * (INPUT_WIDTH - KERNEL_SIZE + 1) * (INPUT_HEIGHT - KERNEL_SIZE + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Libérer la mémoire GPU
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // Afficher quelques résultats
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Libérer la mémoire CPU
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    return 0;
}

*/