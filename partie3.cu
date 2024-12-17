#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define KERNEL_SIZE 5
#define NUM_FEATURE_MAPS_C1 6
#define NUM_FEATURE_MAPS_C2 16
#define S1_OUTPUT_SIZE 14 * 14 * NUM_FEATURE_MAPS_C1
#define S2_OUTPUT_SIZE 5 * 5 * NUM_FEATURE_MAPS_C2
#define FC1_OUTPUT_SIZE 120
#define FC2_OUTPUT_SIZE 84
#define OUTPUT_CLASSES 10

float* load_weights(const char* filename, int size) {
    float* weights = (float*)malloc(size * sizeof(float));
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erreur : %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%e", &weights[i]);
    }
    fclose(file);
    return weights;
}

float* load_biases(const char* filename, int size) {
    float* biases = (float*)malloc(size * sizeof(float));
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erreur : %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%e", &biases[i]);
    }
    fclose(file);
    return biases;
}

void load_mnist_image(const char* filename, float* data, int image_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Erreur : %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 16, SEEK_SET);
    unsigned char pixel;
    for (int i = 0; i < image_size; i++) {
        fread(&pixel, sizeof(unsigned char), 1, file);
        data[i] = pixel / 255.0f;
    }
    fclose(file);
}

__device__ float activation_tanh(float x) {
    return tanhf(x);
}

__device__ float activation_relu(float x) {
    return fmaxf(0.0f, x);
}

__global__ void cudaConvolution2D(float* input, float* weights, float* biases, float* output, int input_width, int kernel_size) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int map = blockIdx.z; // Feature map index

    if (x < input_width - kernel_size + 1 && y < input_width - kernel_size + 1) {
        float sum = biases[map]; // Initialise avec le biais
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += input[(y + j) * input_width + (x + i)] * weights[map * kernel_size * kernel_size + j * kernel_size + i];
            }
        }
        output[map * (input_width - kernel_size + 1) * (input_width - kernel_size + 1) + y * (input_width - kernel_size + 1) + x] = activation_tanh(sum);
    }
}

__global__ void cudaDenseLayer(float* input, float* weights, float* biases, float* output, int input_size, int output_size, bool use_relu) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < output_size) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = use_relu ? activation_relu(sum) : activation_tanh(sum);
    }
}

int main() {
    int input_size = INPUT_WIDTH * INPUT_HEIGHT;
    float* h_input = (float*)malloc(input_size * sizeof(float));
    load_mnist_image("/users/sedaarsl63/Documents/TP_hard/train-images.idx3-ubyte", h_input, input_size);

    // GPU memory allocation
    float *d_input, *d_C1, *d_S1, *d_C2, *d_S2, *d_fc1, *d_fc2, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_C1, NUM_FEATURE_MAPS_C1 * 28 * 28 * sizeof(float));
    cudaMalloc(&d_S1, S1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_C2, NUM_FEATURE_MAPS_C2 * 10 * 10 * sizeof(float));
    cudaMalloc(&d_S2, S2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc2, FC2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_CLASSES * sizeof(float));

    // Load weights and biases for all layers
    float* h_C1_weights = load_weights("layer0_weights.txt", NUM_FEATURE_MAPS_C1 * KERNEL_SIZE * KERNEL_SIZE);
    float* h_C1_biases = load_biases("layer0_biases.txt", NUM_FEATURE_MAPS_C1);
    float* h_C2_weights = load_weights("layer2_weights.txt", NUM_FEATURE_MAPS_C2 * KERNEL_SIZE * KERNEL_SIZE);
    float* h_C2_biases = load_biases("layer2_biases.txt", NUM_FEATURE_MAPS_C2);
    float* h_fc1_weights = load_weights("layer5_weights.txt", FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE);
    float* h_fc1_biases = load_biases("layer5_biases.txt", FC1_OUTPUT_SIZE);
    float* h_fc2_weights = load_weights("layer6_weights.txt", FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE);
    float* h_fc2_biases = load_biases("layer6_biases.txt", FC2_OUTPUT_SIZE);
    float* h_output_weights = load_weights("layer7_weights.txt", OUTPUT_CLASSES * FC2_OUTPUT_SIZE);
    float* h_output_biases = load_biases("layer7_biases.txt", OUTPUT_CLASSES);

    // GPU allocation for weights and biases
    float *d_C1_weights, *d_C1_biases, *d_C2_weights, *d_C2_biases;
    float *d_fc1_weights, *d_fc1_biases, *d_fc2_weights, *d_fc2_biases, *d_output_weights, *d_output_biases;

    cudaMalloc(&d_C1_weights, NUM_FEATURE_MAPS_C1 * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C1_biases, NUM_FEATURE_MAPS_C1 * sizeof(float));
    cudaMalloc(&d_C2_weights, NUM_FEATURE_MAPS_C2 * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C2_biases, NUM_FEATURE_MAPS_C2 * sizeof(float));
    cudaMalloc(&d_fc1_weights, FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_biases, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc2_weights, FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc2_biases, FC2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_weights, OUTPUT_CLASSES * FC2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_biases, OUTPUT_CLASSES * sizeof(float));

    // Copy weights and biases to GPU
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_weights, h_C1_weights, NUM_FEATURE_MAPS_C1 * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_biases, h_C1_biases, NUM_FEATURE_MAPS_C1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_weights, h_C2_weights, NUM_FEATURE_MAPS_C2 * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_biases, h_C2_biases, NUM_FEATURE_MAPS_C2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_weights, h_fc1_weights, FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_biases, h_fc1_biases, FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weights, h_fc2_weights, FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_biases, h_fc2_biases, FC2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, h_output_weights, OUTPUT_CLASSES * FC2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_biases, h_output_biases, OUTPUT_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    // Layer 1: C1
    dim3 blockSize(16, 16);
    dim3 gridSize((28 + 15) / 16, (28 + 15) / 16, NUM_FEATURE_MAPS_C1);
    cudaConvolution2D<<<gridSize, blockSize>>>(d_input, d_C1_weights, d_C1_biases, d_C1, INPUT_WIDTH, KERNEL_SIZE);
    cudaDeviceSynchronize();

    // Layer 2: C2
    gridSize = dim3((10 + 15) / 16, (10 + 15) / 16, NUM_FEATURE_MAPS_C2);
    cudaConvolution2D<<<gridSize, blockSize>>>(d_S1, d_C2_weights, d_C2_biases, d_C2, 14, KERNEL_SIZE);
    cudaDeviceSynchronize();

    // Dense Layer 1
    cudaDenseLayer<<<(FC1_OUTPUT_SIZE + 31) / 32, 32>>>(d_S2, d_fc1_weights, d_fc1_biases, d_fc1, S2_OUTPUT_SIZE, FC1_OUTPUT_SIZE, false);
    cudaDeviceSynchronize();

    // Dense Layer 2
    cudaDenseLayer<<<(FC2_OUTPUT_SIZE + 31) / 32, 32>>>(d_fc1, d_fc2_weights, d_fc2_biases, d_fc2, FC1_OUTPUT_SIZE, FC2_OUTPUT_SIZE, false);
    cudaDeviceSynchronize();

    // Output Layer
    cudaDenseLayer<<<(OUTPUT_CLASSES + 31) / 32, 32>>>(d_fc2, d_output_weights, d_output_biases, d_output, FC2_OUTPUT_SIZE, OUTPUT_CLASSES, true);
    cudaDeviceSynchronize();

    // Retrieve results
    float h_output[OUTPUT_CLASSES];
    cudaMemcpy(h_output, d_output, OUTPUT_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Probabilités finales :\n");
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        printf("Classe %d: %.6f\n", i, h_output[i]);
    }

    return 0;
}
