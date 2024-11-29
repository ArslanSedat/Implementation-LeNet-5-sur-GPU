#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Structure pour représenter une matrice
struct Matrice {
    float* valeurs;
    int lignes;
    int colonnes;
};

// Fonction pour initialiser une matrice aléatoirement
void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i*p + j] = (float)((double)rand() / RAND_MAX * 2 - 1); // Valeurs entre -1 et 1
        }
    }
}

void MatrixInit0(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i*p + j] = 0;
        }
    }
}
// Fonction pour afficher une matrice
void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%10.4f ", M[i*p + j]);
        }
        printf("\n");
    }
}

// Fonction pour additionner deux matrices CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
        }
    }
}

// Fonction kernel pour additionner deux matrices GPU
__global__ void MatrixAddKernel(float *M1, float *M2, float *Mout, int n, int p) {
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
        }
    }
    /*
    for (int i = 0; i < n; i++) {
        //if (i%threadIdx.x == 0)
            for (int j = 0; j < p; j++) {
                //if (true or i%blockIdx.x == 0)
                    Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
            }
    }*/
    //Mout[n*blockIdx.x + threadIdx.x ] = n*blockIdx.x + threadIdx.x;
    //if (idx < n && idy < p) {
    //    Mout[idx*p + idy] = M1[idx*p + idy] + M2[idx*p + idy];
    //}
}

int main() {
    srand(time(NULL)); // Initialisation du générateur de nombres aléatoires
    const int n = 5;
    const int p = 5;
    const int maxaff = 20;

    int blocks = 1;
    int thread = 1;
    clock_t debut, fin;

    // Allocation de mémoire pour les matrices
    struct Matrice A, B, C;
    A.lignes = n;
    A.colonnes = p;
    B.lignes = n;
    B.colonnes = p;
    C.lignes = n;
    C.colonnes = p;

    float *cuda_A, *cuda_B, *cuda_C; 

    A.valeurs = (float*)malloc(n*p * sizeof(float));
    B.valeurs = (float*)malloc(n*p * sizeof(float));
    C.valeurs = (float*)malloc(n*p * sizeof(float));

    // Initialisation des matrices
    if ((n+p) < maxaff) {
        MatrixInit(A.valeurs, n, p);
        MatrixInit(B.valeurs, n, p);
        printf("Matrice A:\n");
        MatrixPrint(A.valeurs, n, p);
        printf("Matrice B:\n");
        MatrixPrint(B.valeurs, n, p);
    }
    else
    {
        printf("n,p : %i, %i \n",n,p);
    }

    // Addition des matrices résultat
    debut = clock();
    MatrixAdd(A.valeurs, B.valeurs, C.valeurs, n, p);
    fin = clock();
    printf("\n\nTemps CPU : %.2f ms\n", (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
    printf("Résultat de l'addition CPU:\n");
    if ((n+p) < maxaff) 
        MatrixPrint(C.valeurs, n, p);

    //MatrixAddGPU(A.valeurs, B.valeurs, C.valeurs, n, p);
    for (int i = 1; i <= n; i *= 2) {
        MatrixInit0(C.valeurs, n, p);
        debut = clock();
        blocks = 1;
        thread = i;

        //(float*)malloc(n*p * sizeof(float));
        cudaMalloc((void**)&cuda_A, n*p * sizeof(float));
        cudaMalloc((void**)&cuda_B, n*p * sizeof(float));
        cudaMalloc((void**)&cuda_C, n*p * sizeof(float));

        cudaMemcpy(cuda_A, A.valeurs, sizeof(float) * n*p, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_B, B.valeurs, sizeof(float) * n*p, cudaMemcpyHostToDevice);

        MatrixAddKernel<<<blocks, thread>>>(cuda_A, cuda_B, cuda_C, n, p);

        cudaMemcpy(C.valeurs, cuda_C, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
        cudaFree(cuda_A);
        cudaFree(cuda_B);
        cudaFree(cuda_C);
        
        fin = clock();
        printf("\nTemps GPU avec n = %i,p = %i : blocks, thread : %i, %i : %.2f ms\n",n,p, blocks, thread,   (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
        if ((n+p) < maxaff) {
            printf("Résultat de l'addition GPU:\n");
            MatrixPrint(C.valeurs, n, p);
        }
    }
    /*
    //MatrixAddGPU(A.valeurs, B.valeurs, C.valeurs, n, p);
    debut = clock();
    blocks = 100;
    thread = 100;
    MatrixAddKernel<<<blocks, thread>>>(A.valeurs, B.valeurs, C.valeurs, n, p);
    cudaDeviceSynchronize();
    fin = clock();
    printf("\n\nTemps GPU blocks, thread : %i, %i : %.2f ms\n",blocks, thread, (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
    printf("Résultat de l'addition GPU:\n");
    if ((n+p) < maxaff) 
        MatrixPrint(C.valeurs, n, p);
    */


    free(A.valeurs);
    free(B.valeurs);
    free(C.valeurs);

    cudaDeviceSynchronize();
    return 0;
}
