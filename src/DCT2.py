import math

import numpy as np
import scipy.fftpack as fft

def calcolaD (N):
    """
    Calcola la matrice di trasformazione DCT ortonormale D di dimensione NxN.

    Args:
        N (int): Dimensione della matrice.

    Returns:
        np.ndarray: Matrice D di dimensione (N, N).
    """
    D = np.zeros((N,N))
    alpha = np.zeros(N)

    alpha[0] = 1/np.sqrt(N)
    alpha[1:N] = np.sqrt(2/N)

    for i in range(N):
        for j in range(N):
            D[i,j] = alpha[i] * np.cos(i * math.pi * (2 * j + 1) / (2 * N))
    return D

def dct(f):
    """
    Applica la DCT 1D ortonormale a un vettore.

    Args:
        f (np.ndarray): Vettore di input.

    Returns:
        np.ndarray: Coefficienti DCT.
    """
    f = f.astype('float64')
    N = len(f)
    D = calcolaD(N)
    return D @ f

def idct(c):
    """
    Applica la IDCT 1D ortonormale a un vettore.

    Args:
        c (np.ndarray): Coefficienti DCT.

    Returns:
        np.ndarray: Vettore ricostruito.
    """
    c = c.astype('float64')
    N = len(c)
    D = calcolaD(N)
    return D.T @ c

def calcolaI_DCT2(f_mat,D, n,m):
    """
    Applica la DCT o IDCT 2D (dipende dalla matrice D usata).

    Args:
        f_mat (np.ndarray): Matrice immagine.
        D (np.ndarray): Matrice DCT (o trasposta per IDCT).
        n (int): Numero righe.
        m (int): Numero colonne.

    Returns:
        np.ndarray: Matrice trasformata.
    """
    c_mat= np.copy(f_mat.astype('float64'))

    # Trasformata per colonne
    for j in range(m):
        c_mat[:,j]= D @ c_mat[:, j]
    # Trasformata per righe
    for j in range(n):
        c_mat[j,:]= (D @ (c_mat[j,:]).T).T
    return c_mat

def dct2(f_mat):
    """
    Applica la DCT2 a una matrice.

    Args:
        f_mat (np.ndarray): Matrice immagine.

    Returns:
        np.ndarray: Coefficienti DCT2.
    """
    n, m = f_mat.shape
    D = calcolaD(n)
    return calcolaI_DCT2(f_mat, D, n, m)

def idct2(c_mat):
    """
    Applica la IDCT2 a una matrice.

    Args:
        c_mat (np.ndarray): Coefficienti DCT2.

    Returns:
        np.ndarray: Matrice immagine ricostruita.
    """
    n, m = c_mat.shape
    D = calcolaD(n)
    return calcolaI_DCT2(c_mat, D.T, n, m)

def dct_lib(f_mat):
    """
    DCT 1D ortonormale usando SciPy.
    """
    return fft.dct(f_mat, norm='ortho')

def dct2_lib(f_mat):
    """
    DCT 2D ortonormale usando SciPy (applicata su righe e poi colonne).
    """
    righe = dct_lib(f_mat.T)
    return dct_lib(righe.T)

def idct_lib(c_mat):
    """
    IDCT 1D ortonormale usando SciPy.
    """
    return fft.idct(c_mat, norm='ortho')


def idct2_lib(c_mat):
    """
    IDCT 2D ortonormale usando SciPy (applicata su righe e poi colonne).
    """
    righe = idct_lib(c_mat.T)
    return idct_lib(righe.T)


