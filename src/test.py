import time

import numpy as np

from Progetto2.src.DCT2 import dct2_lib, dct2, dct, dct_lib
from Progetto2.src.plot import plot_dct_times

test_matrix = np.array([
    [231, 32, 233, 161, 24, 71, 140, 245],
    [247, 40, 248, 245, 124, 204, 36, 107],
    [234, 202, 245, 167, 9, 217, 239, 173],
    [193, 190, 100, 167, 43, 180, 8, 70],
    [11, 24, 210, 177, 81, 243, 8, 112],
    [97, 195, 203, 47, 125, 114, 165, 181],
    [193, 70, 174, 167, 41, 30, 127, 245],
    [87, 149, 57, 192, 65, 129, 178, 228]
])

expected_dct2_result = np.array([
    [1.11e+03, 4.40e+01, 7.59e+01, -1.38e+02, 3.50e+00, 1.22e+02, 1.95e+02, -1.01e+02],
    [7.71e+01, 1.14e+02, -2.18e+01, 4.13e+01, 8.77e+00, 9.90e+01, 1.38e+02, 1.09e+01],
    [4.48e+01, -6.27e+01, 1.11e+02, -7.63e+01, 1.24e+02, 9.55e+01, -3.98e+01, 5.85e+01],
    [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01, 2.66e+01, -3.68e+01, 6.61e+01, 1.25e+02],
    [-1.09e+02, -4.33e+01, -5.55e+01, 8.17e+00, 3.02e+01, -2.86e+01, 2.44e+00, -9.41e+01],
    [-5.38e+00, 5.66e+01, 1.73e+02, -3.54e+01, 3.23e+01, 3.34e+01, -5.81e+01, 1.90e+01],
    [7.88e+01, -6.45e+01, 1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02, 3.98e+01],
    [1.97e+01, -7.81e+01, 9.72e-01, -7.23e+01, -2.15e+01, 8.13e+01, 6.37e+01, 5.90e+00]
])

expected_dct_first_row = np.array([4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02, 6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01])

def test():
    """
    Esegue test di correttezza sui risultati delle DCT1 e DCT2:
    - Confronta l'implementazione personale con quella della libreria SciPy
    - Verifica la precisione rispetto ai risultati attesi
    """
    print("Testing...")
    #confrontro se valori restituiti da idct2(sia implementata che della libreria) rispettano i valori specificati
    #nella consegna
    print("Verifica correttezza di scipy_dct2: ")
    print(np.allclose(dct2_lib(test_matrix), expected_dct2_result, rtol=1e-2))
    print("Verifica correttezza di my_dct2: ")
    print(np.allclose(dct2(test_matrix), expected_dct2_result, rtol=1e-2))
    print("Verifica correttezza di scipy_dct1: ")
    print(np.allclose(dct_lib(test_matrix[0]), expected_dct_first_row, rtol=1e-2))
    print("Verifica correttezza di my_dct1: ")
    print(np.allclose(dct(test_matrix[0]), expected_dct_first_row, rtol=1e-2))


def test_N():
    """
    Confronta i tempi di esecuzione della DCT2 tra l'implementazione personale (my_dct2)
     e quella fornita dalla libreria SciPy (dct2_lib), su matrici quadrate di dimensioni crescenti.

    Dimensioni testate: da 50x50 a 1000x1000 con step di 50.

    Returns:
        tempiDCT2_lib (list): Tempi di esecuzione della DCT2 con SciPy.
        tempiDCT2_imp (list): Tempi di esecuzione della DCT2 implementata manualmente.
        matrix_dimensions (list): Dimensioni delle matrici testate.
    """
    # da 50x50 a 1000x1000, con incrementi di 50.
    matrix_dimensions = list(range(50, 2101, 50))
    print(matrix_dimensions)

    # lista vuota per memorizzare i tempi di esecuzione della DCT2 utilizzando la libreria scipy.
    tempiDCT2_lib = []

    #lista vuota per memorizzare i tempi di esecuzione della mia implementazione della DCT2.
    tempiDCT2_imp = []
    for n in matrix_dimensions:
        print("Dimension: ", n)

        # Genero una matrice NxN con valori casuali compresi tra 0 e 255.
        np.random.seed(5)
        matrix = np.random.uniform(low=0.0, high=255.0, size=(n, n))

        # Misuro il tempo di esecuzione della DCT2 utilizzando la libreria scipy.
        start = time.time()
        dct2_lib(matrix)
        stop = time.time()
        tempiDCT2_lib.append(stop - start)

        # Misuro il tempo di esecuzione della mia implementazione della DCT2.
        start = time.time()
        dct2(matrix)
        stop = time.time()
        tempiDCT2_imp.append(stop - start)

    # Restituisco i tempi di esecuzione e le dimensioni delle matrici testate.
    return tempiDCT2_lib, tempiDCT2_imp, matrix_dimensions

def main():
    """
    Funzione principale: esegue test di correttezza e test prestazionali,
    infine stampa e visualizza i risultati.
    """
    test()
    tempiDCT2_lib, tempiDCT2_imp, matrix_dimensions = test_N()
    plot_dct_times(tempiDCT2_lib, tempiDCT2_imp, matrix_dimensions)



if __name__ == "__main__":
    main()


