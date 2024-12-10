#include <stdio.h>
#include <stdlib.h>

float*** create_3D_array(int N, int P, int L) {
    float*** tab = NULL;
    
    tab = (float***)malloc(N * sizeof(float**));
    
    if (tab == NULL) {
        printf("Erreur d'allocation pour la première dimension\n");
        return NULL;
    }
    

    for (int i = 0; i < N; i++) {
        tab[i] = (float**)malloc(P * sizeof(float*));
        
        if (tab[i] == NULL) {
            printf("Erreur d'allocation pour la deuxième dimension\n");
            while (i >= 0) {
                free(tab[i]);
                i--;
            }
            return NULL;
        }
        
        for (int j = 0; j < P; j++) {
            tab[i][j] = (float*)malloc(L * sizeof(float));
            
            if (tab[i][j] == NULL) {
                printf("Erreur d'allocation pour la troisième dimension\n");
                // Libérer les sous-tableaux déjà alloués
                for (int k = i; k >= 0; k--) {
                    for (int l = P - 1; l >= 0; l--) {
                        free(tab[k][l]);
                    }
                    free(tab[k]);
                }
                return NULL;
            }
        }
    }
    
    return tab;
}

void init_zero_3D_array(float*** tab, int n, int p, int l) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < l; k++) {
                tab[i][j][k] = 0;
            }
        }
    }
}

int random_number(int min, int max) {
    return min + rand() % (max - min + 1);
}

void init_random_3D_array(float*** tableau, int n, int p, int l) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < l; k++) {
                tableau[i][j][k] = (float)(round((double)rand()/RAND_MAX  * 20 - 20));
                //tableau[i][j][k] = (float) random_number(-100, 100);
                //printf("i : %i, j %i k %i : %f\n", i, j, k, tableau[i][j][k]);
            }
        }
    }
}

void print_3D_array(float*** tab, int N, int P, int L) {
    for (int i = 0; i < N; i++) {
        printf("Dimension N (%d):\n", i);
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < L; k++) {
                printf("%10.2f ", (float)tab[i][j][k]);
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
    float*** raw_data = create_3D_array(1, 32, 32);
    init_random_3D_array(raw_data, 1, 32, 32);

    float*** C1_data = create_3D_array(6, 28, 28);
    init_zero_3D_array(C1_data, 6, 28, 28);

    float*** S1_data = create_3D_array(6, 14, 14);
    init_zero_3D_array(S1_data, 6, 14, 14);
    
    float*** C1_kernel = create_3D_array(6, 5, 5);
    init_zero_3D_array(S1_data,6, 5, 5);



    print_3D_array(C1_kernel, 6, 5, 5);
    
    return 0;
}
