#include <stdio.h>
#include <stdlib.h>


typedef struct {
    float*** values;
    int N;  
    int P;  
    int L;  
} ThreeDArray;

ThreeDArray init_3DArray() {
    ThreeDArray arr = {NULL, 0, 0, 0};
    return arr;
}

ThreeDArray create_3D_array(int N, int P, int L) {
    float*** tab = NULL;
    
    tab = (float***)malloc(N * sizeof(float**));
    /*if (tab == NULL) {
        printf("Erreur d'allocation pour la première dimension\n");
        return NULL;
    }*/
    for (int i = 0; i < N; i++) {
        tab[i] = (float**)malloc(P * sizeof(float*));
        /*if (tab[i] == NULL) {
            printf("Erreur d'allocation pour la deuxième dimension\n");
            while (i >= 0) {
                free(tab[i]);
                i--;
            }
            return NULL;
        }*/
        for (int j = 0; j < P; j++) {
            tab[i][j] = (float*)malloc(L * sizeof(float));
            /*if (tab[i][j] == NULL) {
                printf("Erreur d'allocation pour la troisième dimension\n");
                // Libérer les sous-tableaux déjà alloués
                for (int k = i; k >= 0; k--) {
                    for (int l = P - 1; l >= 0; l--) {
                        free(tab[k][l]);
                    }
                    free(tab[k]);
                }
                return NULL;
            }*/
        }
    }
    
    ThreeDArray myArr = init_3DArray();
    myArr.values = tab;
    myArr.N = N;
    myArr.P = P;
    myArr.L = L;
    return myArr;
}

void init_zero_3D_array(ThreeDArray array) {
    for (int i = 0; i < array.N; i++) {
        for (int j = 0; j < array.P; j++) {
            for (int k = 0; k < array.L; k++) {
                array.values[i][j][k] = 0;
            }
        }
    }
}

int random_number(int min, int max) {
    return min + rand() % (max - min + 1);
}

void init_random_3D_array(ThreeDArray array) {
    for (int i = 0; i < array.N; i++) {
        for (int j = 0; j < array.P; j++) {
            for (int k = 0; k < array.L; k++) {
                array.values[i][j][k] = (float)(round((double)rand()/RAND_MAX  * 20 - 20));
                //tableau[i][j][k] = (float) random_number(-100, 100);
                //printf("i : %i, j %i k %i : %f\n", i, j, k, tableau[i][j][k]);
            }
        }
    }
}

void print_3D_array(ThreeDArray array) {
    for (int i = 0; i < array.N; i++) {
        printf("Dimension N (%d):\n", i);
        for (int j = 0; j < array.P; j++) {
            for (int k = 0; k < array.L; k++) {
                printf("%10.2f ", (float) array.values[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void destroy_3DArray(ThreeDArray* arr) {
    if (arr && arr->values) {
        // Libérer chaque sous-tableau
        for (int i = 0; i < arr->N; i++) {
            if (arr->values[i]) {
                for (int j = 0; j < arr->P; j++) {
                    if (arr->values[i][j]) {
                        free(arr->values[i][j]);
                    }
                }
                free(arr->values[i]);
            }
        }
        
        // Libérer le tableau principal
        free(arr->values);
    }
    
    // Réinitialiser les champs
    arr->values = NULL;
    arr->N = 0;
    arr->P = 0;
    arr->L = 0;
}



 /*
__global__ void cudaConvolution2D(ThreeDArray *input, ThreeDArray *kernel, ThreeDArray *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    output->values[0][0][0] = 9;
   
    if (x < output->N && y < output->P) {
        int kernel_radius = kernel->L / 2;
        for (int k = 0; k < output->N; k++) {
            float value = 0.0f;
            for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < input->N && iy >= 0 && iy < input->P) {
                        value += 1;//input.values[k][iy][ix] * kernel.values[0][(ky + kernel_radius) * kernel.L + kx + kernel_radius];
                    }
                }
            }
            output->values[k][y][x] = 9;//value;
        }
    }
}
*/


__global__ void cudaConvolution2D(ThreeDArray* input, ThreeDArray* kernel, ThreeDArray* output) {
    // Calcul des indices de thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Assurez-vous que les indices sont dans les limites de la matrice de sortie
    if (x < output->L && y < output->P && z < output->N) {
        // Calcul de l'index linéaire dans le tableau aplati
        int index = z * output->P * output->L + y * output->L + x;

        // Initialiser la valeur à 9.0f
        output->values[0][0][0] = 9.0f;
    }
}



/*
ThreeDArray* createGPUArray(ThreeDArray A) {
    ThreeDArray* arr = (ThreeDArray*)malloc(sizeof(ThreeDArray));
    arr->N = A.N;
    arr->P = A.P;
    arr->L = A.L;

    cudaMalloc((void**)&arr->values, arr->N * sizeof(float**));
    for (int i = 0; i < arr->N; i++) {
        cudaMalloc((void**)&arr->values[i], arr->P * sizeof(float*));
        for (int j = 0; j < arr->P; j++) {
            cudaMalloc((void**)&arr->values[i][j], arr->L * sizeof(float));
        }
    }

    return arr;
}
*/


ThreeDArray* createGPUArray(ThreeDArray A) {
   ThreeDArray* arr = (ThreeDArray*)malloc(sizeof(ThreeDArray));
    arr->N = A.N;
    arr->P = A.P;
    arr->L = A.L;
    printf("test : \n");
    cudaMalloc((void**)&arr->values, arr->N * sizeof(float**));
    for (int i = 0; i < arr->N; i++) {
        cudaMalloc((void**)&arr->values[i], arr->P * sizeof(float*));
        for (int j = 0; j < arr->P; j++) {
            cudaMalloc((void**)&arr->values[i][j], arr->L * sizeof(float));
        }
    }
    printf("test : \n");
    printf("f %f\n", arr->values[0][0][0] );
    return arr;
}

/*
printf("test : \n");
printf("f %f\n", arr->values[0][0][0] );
*/

/*
void copyCPUtoGPU(ThreeDArray cpuArr, ThreeDArray* gpuArr) {
    cudaDeviceSynchronize();
    for (int i = 0; i < cpuArr.N; i++) {
        for (int j = 0; j < cpuArr.P; j++) {
            cudaMemcpy(gpuArr->values[i][j], cpuArr.values[i][j], sizeof(float) * cpuArr.L, cudaMemcpyHostToDevice);
        }
    }
}
*/

void copyCPUtoGPU(ThreeDArray cpuArr, ThreeDArray* gpuArr) {
    cudaDeviceSynchronize();

    for (int i = 0; i < cpuArr.N; i++) {
        float** gpuLevel2;
        cudaMemcpy(&gpuLevel2, &gpuArr->values[i], sizeof(float**), cudaMemcpyDeviceToHost);
        for (int j = 0; j < cpuArr.P; j++) {
            float* gpuLevel3;
            cudaMemcpy(&gpuLevel3, &gpuLevel2[j], sizeof(float*), cudaMemcpyDeviceToHost);
            cudaMemcpy(gpuLevel3, cpuArr.values[i][j], sizeof(float) * cpuArr.L, cudaMemcpyHostToDevice);
        }
    }
    cudaDeviceSynchronize();
}

/*
void copyGPUtoCPU(ThreeDArray* gpuArr, ThreeDArray cpuArr) {
    cudaDeviceSynchronize();
    
    // Copie du tableau 3D
    for (int i = 0; i < gpuArr->N; i++) {
        for (int j = 0; j < gpuArr->P; j++) {
            cudaMemcpy(cpuArr.values[i][j], gpuArr->values[i][j], sizeof(float) * gpuArr->L, cudaMemcpyDeviceToHost);
        }
    }
}
*/

/*
void copyGPUtoCPU(ThreeDArray* gpuArr, ThreeDArray cpuArr) {
    cudaDeviceSynchronize();
    for (int i = 0; i < gpuArr->N; i++) {
        float** gpuLevel2;
        cudaMemcpy(&gpuLevel2, &gpuArr->values[i], sizeof(float**), cudaMemcpyDeviceToHost);

        for (int j = 0; j < gpuArr->P; j++) {
            float* gpuLevel3;
            cudaMemcpy(&gpuLevel3, &gpuLevel2[j], sizeof(float*), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuArr.values[i][j], gpuLevel3, sizeof(float) * gpuArr->L, cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize();
}
*/
/**/
void copyGPUtoCPU(ThreeDArray* gpuArr, ThreeDArray cpuArr) {
    // Copier les dimensions de gpuArr vers cpuArr
    cpuArr.N = gpuArr->N;
    cpuArr.P = gpuArr->P;
    cpuArr.L = gpuArr->L;

    cpuArr.values = (float***)malloc(cpuArr.N * sizeof(float**));
    for (int i = 0; i < cpuArr.N; i++) {
        cpuArr.values[i] = (float**)malloc(cpuArr.P * sizeof(float*));
        for (int j = 0; j < cpuArr.P; j++) {
            cpuArr.values[i][j] = (float*)malloc(cpuArr.L * sizeof(float));
        }
    }
    
    // Copier les données depuis le GPU vers le CPU
    for (int i = 0; i < gpuArr->N; i++) {
        for (int j = 0; j < gpuArr->P; j++) {
            printf("copie i : %i, j : %i\n",i, j);
            printf("f : %f\n",cpuArr.values[i][j][0]);
            cudaMemcpy(cpuArr.values[i][j], gpuArr->values[i][j], cpuArr.L * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    //return cpuArr;  // Retourne l'objet modifié
}




/*
void freeGPUArray(ThreeDArray* arr) {
    if (arr != NULL) {
        for (int i = 0; i < arr->N; i++) {
            for (int j = 0; j < arr->P; j++) {
                cudaFree(arr->values[i][j]);
            }
            cudaFree(&arr->values[i]);
        }
        cudaFree(arr->values);
        free(arr);
    }
}
*/

void freeGPUArray(ThreeDArray* arr) {
    if (arr != NULL) {
        // Libération des niveaux 3 (float*)
        for (int i = 0; i < arr->N; i++) {
            float** gpuLevel2;
            cudaMemcpy(&gpuLevel2, &arr->values[i], sizeof(float**), cudaMemcpyDeviceToHost);

            for (int j = 0; j < arr->P; j++) {
                float* gpuLevel3;
                cudaMemcpy(&gpuLevel3, &gpuLevel2[j], sizeof(float*), cudaMemcpyDeviceToHost);
                cudaFree(gpuLevel3); // Libération des float*
            }

            // Libération des niveaux 2 (float**)
            cudaFree(gpuLevel2);
        }

        // Libération du niveau 1 (float***)
        cudaFree(arr->values);

        // Libération de la structure elle-même
        free(arr);
    }
}


void multiGPU(int blocks, int thread, ThreeDArray A, ThreeDArray B, ThreeDArray C) {
    float *cuda_A, *cuda_B, *cuda_C; 
    
    cudaMalloc((void**)&cuda_A, A.N*A.P*A.L * sizeof(float));
    cudaMalloc((void**)&cuda_B, B.N*B.P*B.L * sizeof(float));
    cudaMalloc((void**)&cuda_C, C.N*C.P*C.L * sizeof(float));

    cudaMemcpy(cuda_A, A.values, sizeof(float) * A.N*A.P*A.L, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B.values, sizeof(float) * B.N*B.P*B.L, cudaMemcpyHostToDevice);


    /*
    A_cuda = createGPUArray(A);
    B_cuda = createGPUArray(B);
    C_cuda = createGPUArray(C);

    copyCPUtoGPU(A, A_cuda);
    copyCPUtoGPU(B, B_cuda);
    copyCPUtoGPU(C, C_cuda);

    //cudaConvolution2D<<<blocks, thread>>>(A_cuda, B_cuda, C_cuda);

    printf("haaa\n");
    copyGPUtoCPU(C_cuda, C);
    //print_3D_array(C);

    freeGPUArray(A_cuda);
    freeGPUArray(B_cuda);
    freeGPUArray(C_cuda);
    */
}



int main() {

    ThreeDArray myArr = init_3DArray();

    ThreeDArray raw_data = create_3D_array(1, 32, 32);
    init_random_3D_array(raw_data);

    ThreeDArray C1_data = create_3D_array(6, 28, 28);
    init_zero_3D_array(C1_data);

    ThreeDArray S1_data = create_3D_array(6, 14, 14);
    init_zero_3D_array(S1_data);
    
    ThreeDArray C1_kernel = create_3D_array(6, 5, 5);
    init_zero_3D_array(C1_kernel);


    

    ThreeDArray C = create_3D_array(1, 7, 7);
    init_random_3D_array(C);
    print_3D_array(C);

    ThreeDArray K = create_3D_array(1, 1, 1);
    init_random_3D_array(K);
    print_3D_array(K);

    ThreeDArray P = create_3D_array(1, 7, 7);
    init_zero_3D_array(P);
    print_3D_array(P);

    //cudaConvolution2D<<<2, 2>>>(C, K, P, P.va);
    multiGPU(2, 2, C, K, P);
    print_3D_array(P);

    return 0;
}
