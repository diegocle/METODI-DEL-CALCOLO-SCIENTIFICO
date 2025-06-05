import scipy.sparse as sp
import time
from Progetto1.lib import risolvi as ri
from Progetto1.lib.controlli import controlliDim, controlloGradientePossibile

# Numero massimo di iterazioni consentite nei metodi iterativi.
MAXITE = 200000


def metodiIterativi(A, b, x, tol, type):
    """
        Risolve il sistema lineare Ax = b usando uno tra quattro metodi iterativi:
        1 = Jacobi, 2 = Gauss-Seidel, 3 = Gradiente, 4 = Gradiente Coniugato.

        Input:
        - A: matrice dei coefficienti (sparse matrix)
        - b: vettore dei termini noti
        - x: vettore iniziale
        - tol: tolleranza sull'errore relativo del residuo
        - type: intero da 1 a 4 che identifica il metodo iterativo

        Output:
        - errRel: errore relativo finale tra soluzione esatta e approssimata
        - nIte: numero di iterazioni eseguite
        - timeIte: tempo di esecuzione in secondi
        """
    global L, D_inv
    controlliDim(A, x)
    A = A.tocsr()

    x_k, r, errR, nIte = ri.inizializza(A, x)

    if type == 1:
        D_inv = ri.InverseMatrixDiagonal(A)
    elif type == 2:
        L = sp.tril(A)
    elif type == 3 or type == 4:
        controlloGradientePossibile(A)
    else:
        raise ValueError(" Metodo non trovato ")

    d = r

    #inizio a calcolare il tempo
    start = time.time()
    #inizio iterazioni
    while (errR > tol):

        if (nIte < MAXITE):
            #controllo come aggiornare x_k
            if type == 1:
                x_k = updateJacobi(D_inv, x_k, r)
                r, errR = ri.errorRelativoResiduo(A, b, x_k)
            elif type == 2:
                x_k = updateGausSeidel(A, x_k, r)
                r, errR = ri.errorRelativoResiduo(A, b, x_k)
            elif type == 3:
                x_k = updateGradiente(A, x_k, r)
                r, errR = ri.errorRelativoResiduo(A, b, x_k)
            else:
                x_k, d, r, errR= updateGradienteConiugato(A, b, x_k, r, d)
            nIte += 1
        else:
            raise ValueError("Arrivato al massimo di iterazioni")
    stop = time.time()
    # calcolo il tempo di esecuzione
    timeIte = stop - start
    # calcolo errore relativo
    errRel = ri.errorRelativo(x, x_k)
    return errRel, nIte, timeIte


def updateJacobi(D_inv, x, r):
    """
    Aggiornamento della soluzione secondo il metodo di Jacobi.

    Input:
    - D_inv: inversa della diagonale di A
    - x: soluzione corrente
    - r: residuo corrente

    Output:
    - x_k: nuova iterazione della soluzione
    """
    # aggiorno x_k
    x_k = x + D_inv @ r
    # calcolo il residuo
    return x_k

def updateGausSeidel(L, x, r):
    """
    Aggiornamento della soluzione secondo il metodo di Gauss-Seidel.

    Input:
    - L: parte triangolare inferiore della matrice A
    - x: soluzione corrente
    - r: residuo corrente

    Output:
    - x_k: nuova iterazione della soluzione
    """
    x_k= x + ri.SolvTriangularLower(L, r)
    #x_k = x + sp.linalg.spsolve_triangular(sp.tril(L), r.toarray(), lower=True)
    return x_k


def updateGradiente(A, x, r):
    """
    Aggiornamento secondo il metodo del Gradiente (Steepest Descent).

    Input:
    - A: matrice dei coefficienti
    - x: soluzione corrente
    - r: residuo corrente

    Output:
    - x_k: nuova iterazione della soluzione
    """
    y = A @ r
    a = (r @ r) / (r @ y)
    x_k = x + a * r
    return x_k


def updateGradienteConiugato(A, b, x, r, d):
    """
    Aggiornamento secondo il metodo del Gradiente Coniugato.

    Parametri:
    - A: matrice dei coefficienti
    - b: vettore dei termini noti
    - x: soluzione corrente
    - r: residuo corrente
    - d: direzione di discesa precedente

    Output:
    - x_k: nuova iterazione della soluzione
    - d: nuova direzione di discesa
    - r: nuovo residuo
    - errR: nuovo errore relativo del residuo
    """
    denominatore = d @ (A @ r)
    a = (r @ r) / denominatore
    x_k = x + a * d

    r, errR = ri.errorRelativoResiduo(A, b, x_k)
    beta = ((d @ A) @ r) / denominatore
    d = r - beta * d
    return x_k, d, r, errR