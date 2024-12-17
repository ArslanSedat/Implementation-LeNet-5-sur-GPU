#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cfloat>


typedef struct {
    float* values;
    int N;  
    int P;  
    int L;  
} ThreeDArray;

ThreeDArray* init_3DArray() {
    ThreeDArray* arr = (ThreeDArray*)malloc(sizeof(ThreeDArray));
    if (arr == NULL) {
        fprintf(stderr, "Error allocating memory for ThreeDArray structure\n");
        exit(1);
    }
    arr->N = 0;
    arr->P = 0;
    arr->L = 0;
    arr->values = NULL;

    return arr;
}

ThreeDArray* create_3D_array(int N, int P, int L) {
    // Allouer de la mémoire pour la structure ThreeDArray
    ThreeDArray* arr = (ThreeDArray*)malloc(sizeof(ThreeDArray));
    if (arr == NULL) {
        fprintf(stderr, "Error allocating memory for ThreeDArray structure\n");
        exit(1);
    }

    arr->N = N;
    arr->P = P;
    arr->L = L;

    arr->values = (float*)malloc(N * P * L * sizeof(float));
    if (arr->values == NULL) {
        fprintf(stderr, "Error allocating memory for the 3D array values\n");
        exit(1);
    }

    return arr;
}

void init_zero_3D_array(ThreeDArray *array) {
    // Vérifier si le tableau est valide
    if (array == NULL || array->values == NULL) {
        printf("Array is empty or not initialized.\n");
        return;
    }
    for (int i = 0; i < array->N * array->P * array->L; i++) {
        array->values[i] = 0.0;
    }
}

int random_number(int min, int max) {
    return min + rand() % (max - min + 1);
}

void init_random_3D_array(ThreeDArray* array) {
    if (array == NULL || array->values == NULL) {
        printf("Array is empty or not initialized.\n");
        return;
    }

    srand(time(NULL));

    for (int i = 0; i < array->N * array->P * array->L; i++) {
        // Calcul d'un nombre aléatoire entre -20 et 20
        array->values[i] = (float)(rand() % 10 - 5); // -20 à 20 inclus
    }
}

void init_a_value(ThreeDArray* array, float value) {
    if (array == NULL || array->values == NULL) {
        printf("Array is empty or not initialized.\n");
        return;
    }
    for (int i = 0; i < array->N * array->P * array->L; i++) {
        array->values[i] = value; // -20 à 20 inclus
    }
}

void print_3D_array(ThreeDArray* array) {
    if (array == NULL || array->values == NULL) {
        printf("Array is empty or not initialized.\n");
        return;
    }

    // Boucles pour parcourir le tableau 3D (N x P x L)
    for (int i = 0; i < array->N; i++) {
        for (int j = 0; j < array->P; j++) {
            for (int k = 0; k < array->L; k++) {
                // Calculer l'index dans la représentation 1D
                int index = i * array->P * array->L + j * array->L + k;
                printf("%10.2f ", array->values[index]);
            }
            printf("\n");  // Nouvelle ligne après chaque ligne P
        }
        printf("\n");  // Nouvelle ligne après chaque plan N
    }
    printf("//\n");
}

void free_3DArray(ThreeDArray* arr) {
    if (arr != NULL) {
        if (arr->values != NULL) {
            free(arr->values); // Libérer les données de values si elles ont été allouées
        }
        free(arr); // Libérer la structure elle-même
    }
}

__global__ void cudaConvolution2D(float *images, float *kernel, float *output, 
                                   int AN, int AL, int AP, 
                                   int BN, int BL, int BP,
                                    int CL, int CP, int stride = 1) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxtid = blockDim.x;

    int size = max(AN*CL*CP/maxtid,1) ;
    float temp;
    int i,j;

    int outHeight = (AL - BL) / 1 + 1;
    int outWidth = (AP - BP) / 1 + 1;

    for (int kk = tid * size; kk <= (tid+1) * size; kk++) {
        if (true) { //(kk < AN * outHeight * outWidth) 
            int imageIdx = kk / (outHeight * outWidth);  // Indice de l'image
            int remaining = kk % (outHeight * outWidth);
            int outRow = remaining / outWidth;  // Ligne de la sortie
            int outCol = remaining % outWidth; // Colonne de la sortie

            int startRow = outRow * stride;
            int startCol = outCol * stride;

            float sum = 0.0f;

            for (i = 0; i < BL; i++) {
                for (j = 0; j < BP; j++) {
                    int rowIdx = startRow + i;
                    int colIdx = startCol + j;

                    if (rowIdx < AL && colIdx < AP) { //
                        int imgIndex = imageIdx * AL * AP + rowIdx * AP + colIdx;
                        int kernelIndex = i * BP + j;
                        sum += images[imgIndex] * kernel[kernelIndex]; // Produit scalaire
                    }
                }
            }
            int outputIndex = imageIdx * outHeight * outWidth + outRow * outWidth + outCol;
            output[outputIndex] = sum;
        }
    }
}

void cudaConvolution2D_GPU(int blocks, int thread, ThreeDArray* A, ThreeDArray* B, ThreeDArray* C) {
    float *cuda_A, *cuda_B, *cuda_C;

    int outHeight = (A->L - B->L) / 1 + 1;  // Hauteur de la sortie (sans padding)
    int outWidth = (A->P - B->P) / 1 + 1;   // Largeur de la sortie (sans padding)

    if (C->values != NULL) {
        free(C->values);
    }
    C->values = (float*)malloc(A->N * outHeight * outWidth * sizeof(float));

    cudaMalloc((void**)&cuda_A, A->N * A->L * A->P * sizeof(float));
    cudaMalloc((void**)&cuda_B, B->L * B->P * sizeof(float));  // Noyau
    cudaMalloc((void**)&cuda_C, A->N * outHeight * outWidth * sizeof(float));  // Résultat

    cudaMemcpy(cuda_A, A->values, A->N * A->L * A->P * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B->values, B->L * B->P * sizeof(float), cudaMemcpyHostToDevice);

    cudaConvolution2D<<<blocks, thread>>>(cuda_A, cuda_B, cuda_C, 
                                         A->N, A->L, A->P, 
                                         B->N, B->L, B->P,
                                         outHeight, outWidth, 1);

    cudaMemcpy(C->values, cuda_C, A->N * outHeight * outWidth * sizeof(float), cudaMemcpyDeviceToHost);

    C->N = A->N;
    C->L = outHeight;
    C->P = outWidth;

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);

}

__global__ void avgPooling2D(float *images, float *output, 
                              int N, int L, int P, 
                              int stride = 2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Taille de la portion de travail pour chaque thread
    int size = max(N * L * P / blockDim.x, 1);
    
    int i, j;

    // Taille de l'image de sortie après pooling
    int outHeight = L / stride;
    int outWidth = P / stride;

    // Parcourir chaque pixel de l'image de sortie
    for (int kk = tid * size; kk <= (tid + 1) * size; kk++) {
        if (kk < N * outHeight * outWidth) {
            int imageIdx = kk / (outHeight * outWidth);  // Indice de l'image
            int remaining = kk % (outHeight * outWidth);
            int outRow = remaining / outWidth;  // Ligne de la sortie
            int outCol = remaining % outWidth; // Colonne de la sortie

            float sum = 0.0f;

            // Moyenne d'un bloc 2x2
            for (i = 0; i < stride; i++) {
                for (j = 0; j < stride; j++) {
                    int rowIdx = outRow * stride + i;
                    int colIdx = outCol * stride + j;

                    // Vérifier que les indices sont dans les limites
                    if (rowIdx < L && colIdx < P) {
                        int imgIndex = imageIdx * L * P + rowIdx * P + colIdx;
                        sum += images[imgIndex];
                    }
                }
            }

            // Calculer la moyenne du bloc
            float avg = sum / (stride * stride);

            // Affecter la valeur moyenne à l'image de sortie
            int outputIndex = imageIdx * outHeight * outWidth + outRow * outWidth + outCol;
            output[outputIndex] = avg;
        }
    }
}

void AvgPooling_GPU(int blocks, int thread, ThreeDArray* A, ThreeDArray* C) {
    float *cuda_A, *cuda_B, *cuda_C;

    ThreeDArray* B = create_3D_array(1, 2, 2);
    init_a_value(B, 1);
    //init_random_3D_array(B);
    int stride = 2;
    int outHeight = (A->L - B->L) / stride + 1;  //stride == 2
    int outWidth = (A->P - B->P) / stride + 1;   // stride == 2

    if (C->values != NULL) {
        free(C->values);
    }
    C->values = (float*)malloc(A->N * outHeight * outWidth * sizeof(float));

    cudaMalloc((void**)&cuda_A, A->N * A->L * A->P * sizeof(float));
    cudaMalloc((void**)&cuda_B, B->L * B->P * sizeof(float)); 
    cudaMalloc((void**)&cuda_C, A->N * outHeight * outWidth * sizeof(float));

    cudaMemcpy(cuda_A, A->values, A->N * A->L * A->P * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B->values, B->L * B->P * sizeof(float), cudaMemcpyHostToDevice);

    avgPooling2D<<<blocks, thread>>>(cuda_A, cuda_C, 
                                         A->N, A->L, A->P, stride);

    cudaMemcpy(C->values, cuda_C, A->N * outHeight * outWidth * sizeof(float), cudaMemcpyDeviceToHost);

    C->N = A->N;
    C->L = outHeight;
    C->P = outWidth;

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    free_3DArray(B);
}

int main() {
    ThreeDArray* A = create_3D_array(1, 8, 8);
    ThreeDArray* B = create_3D_array(1, 1, 1);
    ThreeDArray* C = create_3D_array(2, 4, 4);

    init_random_3D_array(A);
    init_random_3D_array(B);

    print_3D_array(A);
    print_3D_array(B);

    cudaConvolution2D_GPU(10,10,A, B, C);
    print_3D_array(C);

    AvgPooling_GPU(10,10,A, C);
    print_3D_array(C);
    /*
    ThreeDArray* B = create_3D_array(1, 8, 2);
    ThreeDArray* C = create_3D_array(2, 1, 4);
    print_3D_array(A);
    init_random_3D_array(B);
    print_3D_array(B);
    init_zero_3D_array(C);
    print_3D_array(C);
    
    multiGPU(1, 1, A,  B, C);

    printf("matrice C : \n");
    print_3D_array(C);
    */

    return 0;
}
