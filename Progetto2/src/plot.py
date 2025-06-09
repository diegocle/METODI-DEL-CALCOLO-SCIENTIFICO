from matplotlib import pyplot as plt
import numpy as np

def plot_dct_times(times_scipy_dct, times_my_dct, matrix_dimensions):
    """
    Genera un grafico (in scala logaritmica) dei tempi di esecuzione della DCT2,
    confrontando l'implementazione personale con quella della libreria SciPy.

    Args:
        times_scipy_dct (list of float): Tempi di esecuzione per DCT2 con SciPy.
        times_my_dct (list of float): Tempi di esecuzione per la DCT2 implementata manualmente.
        matrix_dimensions (list of int): Dimensioni delle matrici NxN testate.

    Salva un file PNG e mostra il grafico.
    """
    n3 = [n ** 3 / 1e+9  for n in matrix_dimensions]
    n2_logn = [n ** 2 * np.log(n) / 1e+9  for n in matrix_dimensions]
    # Creazione della figura del grafico con dimensioni 10x6 pollici
    plt.figure(figsize=(10, 6))

    # Aggiunta della curva dei tempi
    #versione FFT
    plt.semilogy(matrix_dimensions, times_scipy_dct, label='DCT2 libreria', color="tab:green")
    plt.semilogy(matrix_dimensions, n2_logn, label='n^2 * log(n)', color="tab:green", linestyle='dashed')

    #mia implementazione
    plt.semilogy(matrix_dimensions, times_my_dct, label='DCT2 implementata', color="tab:blue")
    plt.semilogy(matrix_dimensions, n3, label='n^3', color="tab:blue", linestyle='dashed')


    plt.xlabel('Dimensione N')
    plt.ylabel('Tempo di esecuzione in secondi')
    plt.title('Tempi di esecuzione della DCT2 al variare della dimensione N')

    plt.legend()
    plt.grid(True)

    # Salvataggio dell'immagine del grafico in un file PNG
    plt.savefig('grafico_cofronto_tempi_DCT2.png')

    plt.show()