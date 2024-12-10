#include <stdio.h>
#include <stdlib.h>

float* load_weights(const char* filename, int size) {
    float* weights = (float*)malloc(size * sizeof(float));
    FILE* file = fopen(filename, "r");
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &weights[i]);
    }
    fclose(file);
    return weights;
}

__global__ void cudaConvolution2D(float* input, float* weights, float* output, int input_width, int input_height, int kernel_size) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < input_width - kernel_size + 1 && y < input_height - kernel_size + 1) {
        float sum = 0.0;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += input[(y + j) * input_width + (x + i)] * weights[j * kernel_size + i];
            }
        }
        output[y * (input_width - kernel_size + 1) + x] = sum;
    }
}



int main() {
    int layer_sizes[7] = {6 * 5 * 5, 16 * 5 * 5, 120 * 400, 84 * 120, 10 * 84};
    float* d_weights[7];

    for (int i = 0; i < 7; i++) {
        char weights_file[50];
        sprintf(weights_file, "layer%d_weights.txt", i);
        float* weights = load_weights(weights_file, layer_sizes[i]);
        cudaMalloc((void**)&d_weights[i], layer_sizes[i] * sizeof(float));
        cudaMemcpy(d_weights[i], weights, layer_sizes[i] * sizeof(float), cudaMemcpyHostToDevice);
        free(weights);
        printf("Poids pour la couche %d chargés dans le GPU.\n", i);
    }
    for (int i = 0; i < 7; i++) {
        cudaFree(d_weights[i]);
    }
    return 0;
}
