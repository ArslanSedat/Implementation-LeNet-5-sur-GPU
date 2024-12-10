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

void destroy_3D_array(float*** tab, int n, int p, int l) {
    if (!tab) return;

    // Libérer les sous-tableaux
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            free(tab[i][j]);
        }
    }

    // Libérer les sous-tableaux restants
    for (int i = n - 1; i >= 0; i--) {
        for (int j = p - 1; j >= 0; j--) {
            free(tab[i][j]);
        }
        for (int j = p - 1; j >= 0; j--) {
            free(tab[i][j]);
        }
    }

    // Libérer le tableau principal
    for (int i = n - 1; i >= 0; i--) {
        free(tab[i]);
    }
    free(tab);
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


    print_3D_array(C1_kernel);
    
    return 0;
}
