#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


struct Matrice {
    float* valeurs;
    int lignes;
    int colonnes;
};


void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i*p + j] = (float)(round((double)rand()/RAND_MAX  * 20 - 20));
            //M[i*p + j] = (float)((double)rand() / RAND_MAX * 2 - 1);
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

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%10.2f ", M[i*p + j]);
        }
        printf("\n");
    }
}


//ADD 
//ADD CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
        }
    }
}
//ADD GPU
__global__ void MatrixAddKernel(float *M1, float *M2, float *Mout, int n, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxtid = blockDim.x;

    int size = n*p/maxtid ;

    for (int i = (tid)*size; i < (tid+1)*size ; i++) { //tid * size_per_thread //((tid + 1) * size_per_thread)
        if (i < n*p)
            Mout[i] = M1[i] + M2[i]; //size; //tid * 100 + i;
    }
}
void addGPU(int blocks, int thread, float *M1, float *M2, float *Mout, int n, int p) {
    float *cuda_A, *cuda_B, *cuda_C; 
    
    cudaMalloc((void**)&cuda_A, n*p * sizeof(float));
    cudaMalloc((void**)&cuda_B, n*p * sizeof(float));
    cudaMalloc((void**)&cuda_C, n*p * sizeof(float));

    cudaMemcpy(cuda_A, M1, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, M2, sizeof(float) * n*p, cudaMemcpyHostToDevice);

    MatrixAddKernel<<<blocks, thread>>>(cuda_A, cuda_B, cuda_C, n, p);

    cudaMemcpy(Mout, cuda_C, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
}

//MULTI
//MULTI CPU
void MatrixMulti(float *M1, float *M2, float *Mout, int n, int p) {
    float temp;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < p; k++) {
                sum += M1[i*n+k] * M2[k*n+j]; 
            }
            Mout[i*p + j] = sum;
        }  
    }
}
//MULTI GPU
__global__ void MatrixMultiKernel(float *M1, float *M2, float *Mout, int n, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxtid = blockDim.x;

    int size = n*p/maxtid ;
    float temp;
    int i,j;

    for (int kk = (tid)*size; kk < (tid+1)*size ; kk++) { //tid * size_per_thread //((tid + 1) * size_per_thread)
        if (kk < n*p){
            i = kk%p;
            j = kk/p;
            float sum = 0.0f;
            for (int k = 0; k < p; k++) {
                sum += M1[i*n+k] * M2[k*n+j]; 
            }
            Mout[i*p + j] = sum;
        }   
    }
}
void multiGPU(int blocks, int thread, float *M1, float *M2, float *Mout, int n, int p) {
    float *cuda_A, *cuda_B, *cuda_C; 
    
    cudaMalloc((void**)&cuda_A, n*p * sizeof(float));
    cudaMalloc((void**)&cuda_B, n*p * sizeof(float));
    cudaMalloc((void**)&cuda_C, n*p * sizeof(float));

    cudaMemcpy(cuda_A, M1, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, M2, sizeof(float) * n*p, cudaMemcpyHostToDevice);

    MatrixMultiKernel<<<blocks, thread>>>(cuda_A, cuda_B, cuda_C, n, p);

    cudaMemcpy(Mout, cuda_C, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
}

int main(int argc, char *argv[]) {
    int nn=atoi(argv[1]);
    int pp=atoi(argv[1]);
    srand(time(NULL)); // Initialisation du générateur de nombres aléatoires
    const int n = nn;//10000;
    const int p = pp;//10000;
    const int maxaff = 10;

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

    //initialisation
    MatrixInit(B.valeurs, n, p);
    MatrixInit(A.valeurs, n, p);
    //float source_array[] = {1, 2, 3, 14, 5, 6, 7, 8, 9};
    //memcpy(A.valeurs, source_array, sizeof(source_array));

    //float source_array2[] = {10,11,12,13,14,15,16,17,18};
    //memcpy(B.valeurs, source_array2, sizeof(source_array));


    if ((n+p) < maxaff) {
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
    printf("\n Addition : \n\n");
    MatrixInit0(C.valeurs, n, p);
    debut = clock();
    MatrixAdd(A.valeurs, B.valeurs, C.valeurs, n, p);
    fin = clock();
    printf("Temps CPU : %.2f ms\n", (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
    if ((n+p) < maxaff) 
    {
        printf("Résultat de l'addition CPU:\n");
        MatrixPrint(C.valeurs, n, p);
    }
        
    //MatrixAddGPU(A.valeurs, B.valeurs, C.valeurs, n, p);
    for (int i = 1; i <= n; i *= 2) {
        MatrixInit0(C.valeurs, n, p);
        debut = clock();
        blocks = i;
        thread = i;

        addGPU(blocks, thread, A.valeurs, B.valeurs, C.valeurs, n, p);

        fin = clock();
        printf("Temps GPU avec n = %i,p = %i : blocks, thread : %i, %i : %.2f ms\n",n,p, blocks, thread,   (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
        if ((n+p) < maxaff) {
            printf("Résultat de l'addition GPU:\n");
            MatrixPrint(C.valeurs, n, p);
        }
    }

    
    // Multiplication des matrices résultat
    printf("\nMultiplication : \n\n");
    MatrixInit0(C.valeurs, n, p);
    debut = clock();
    MatrixMulti(A.valeurs, B.valeurs, C.valeurs, n, p);
    fin = clock();
    printf("Temps CPU : %.2f ms\n", (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
    if ((n+p) < maxaff) 
    {
        printf("Résultat de la multiplication CPU:\n");
        MatrixPrint(C.valeurs, n, p);
    }

    for (int i = 1; i <= n; i *= 2) {
        MatrixInit0(C.valeurs, n, p);
        debut = clock();
        blocks = i;
        thread = i;

        multiGPU(blocks, thread, A.valeurs, B.valeurs, C.valeurs, n, p);

        fin = clock();
        printf("Temps GPU avec n = %i,p = %i : blocks, thread : %i, %i : %.2f ms\n",n,p, blocks, thread,   (double)(fin - debut) / CLOCKS_PER_SEC * 1000);
        if ((n+p) < maxaff) {
            printf("Résultat de la multiplication GPU:\n");
            MatrixPrint(C.valeurs, n, p);
        }
    }

    free(A.valeurs);
    free(B.valeurs);
    free(C.valeurs);

    cudaDeviceSynchronize();
    return 0;
}
