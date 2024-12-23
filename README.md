# **Implémentation du CNN LeNet-5 sur GPU avec CUDA**

## **Description du projet**
Ce projet implémente l'inférence d'un réseau neuronal convolutif classique, le **LeNet-5**, en utilisant **CUDA** pour paralléliser les calculs sur GPU. Le LeNet-5 est conçu pour la reconnaissance de chiffres avec la base de données MNIST. Ce projet explore les avantages de l'utilisation des GPU pour l'accélération des calculs.

## **Objectifs**
1. Prendre en main **CUDA** pour paralléliser des calculs sur GPU.
2. Comparer les performances entre **CPU** et **GPU** pour des opérations de base.
3. Implémenter les **premières couches** d'un CNN (convolution, sous-échantillonnage, activation).
4. Intégrer les **poids pré-entraînés** et effectuer des prédictions sur le dataset MNIST.

## **Structure du projet**
- **`partie1.cu`** : Prise en main de CUDA avec les opérations de base (addition et multiplication de matrices).
- **`partie2.cu`** : Implémentation des couches principales du LeNet-5 (convolution, sous-échantillonnage, activation).
- **`partie3.cu`** : Intégration des poids et exécution de l'inférence sur le dataset MNIST.
- **`layer...`** : Contient les poids et les biais obtenus avec le fichier Python LeNet5.
- **`LeNet5.ipynb`** : Notebook Python pour générer les poids du modèle et effectuer une inférence de référence.

## **Prérequis**
- Un GPU compatible CUDA.
- **Python** avec les bibliothèques suivantes :
  - `numpy`
  - `matplotlib`
  - `torch`

## **Installation**
Clonez le dépôt :
   ```bash
   git clone [https://github.com/username/lenet5-gpu.git](https://github.com/ArslanSedat/Implementation-LeNet-5-sur-GPU.git)
   ```

## **Utilisation**
**Exemple avec le fichier partie1 :**
  1. **Compilation :**
  nvcc partie1.cu -o partie1 
  2. **Exécution :**
  ./partie1

## **Auteurs**
- **Sedat Arslan**  
- **Kilian Pomel**
