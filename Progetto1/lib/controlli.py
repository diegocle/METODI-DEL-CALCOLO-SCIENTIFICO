import numpy as np
import scipy.sparse as sp

def controlliDim(A,x):
    """
        Controlla che la matrice A sia quadrata e che le dimensioni di x siano compatibili con A
        e valori della diagonale di A diversi da 0

        INPUT:
        A: matrice (sparse o densa), deve essere quadrata.
        x: vettore colonna, deve avere lunghezza compatibile con A.

        Solvable:
        ValueError: se A non è quadrata o se le dimensioni non sono compatibili.
    """
    m, n = A.shape
    l = x.shape[0]
    diag = A.diagonal()
    if not np.all(np.abs(diag) > 1e-16):
        raise ValueError("Nella diagonale della matrice sono presenti valori uguali a 0 ")

    # Controlli sulle dimensioni e proprietà della matrice
    if m != n:
        raise ValueError("La matrice non è quadrata")
    elif l != m:
        raise ValueError("Le dimensioni della matrice non corrispondono a quelle del vettore x")

def is_positive_definite(A):
    """
        Verifica se la matrice A è definita positiva.

        INPUT:
        A: matrice sparsa quadrata.

        Solvable:
        ValueError: se A non è definita positiva.
    """
    if not np.all(sp.linalg.eigs(A)[0]) > 0:
        raise ValueError("La matrice non è definita positiva")

def is_simmetrica(A):
    """
        Verifica se la matrice A è simmetrica.

        INPUT:
        A: matrice sparsa quadrata.

        Solvable:
        ValueError: se A non è simmetrica.
    """
    if (A - A.T).nnz != 0:
        raise ValueError("La matrice non è simmetrica")

def controlloGradientePossibile(A):
    """
        Verifica se è possibile applicare il metodo del gradiente:
        controlla che la matrice sia simmetrica e definita positiva.

        INPUT:
        A: matrice sparsa quadrata.

        Solvable:
        ValueError: se la matrice non è simmetrica o non è definita positiva.
    """
    is_positive_definite(A)
    is_simmetrica(A)