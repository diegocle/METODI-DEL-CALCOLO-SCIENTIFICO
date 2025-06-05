import scipy.sparse as sp
import time
from Progetto1.lib import risolvi as ri
from Progetto1.lib.controlli import controlliDim, controlloGradientePossibile
from Progetto1.lib.risolvi import inizializza

MAXITE = 200000


def metodo_jacobi(A, b, x, tol):
    """
        Metodo di Jacobi per la risoluzione di sistemi lineari Ax = b.

        Input:
            A   : matrice dei coefficienti (sparsa, quadrata)
            b   : vettore dei termini noti
            x   : soluzione esatta (per calcolo errore)
            tol : tolleranza sull'errore relativo del residuo

        Output:
            err   : errore relativo ||x - x_k|| / ||x||
            nit   : numero di iterazioni effettuate
            tempo : tempo di esecuzione in secondi
    """
    controlliDim(A,x)
    x_k, r, errR, nIte = inizializza(A, b)
    start = time.time()
    D_inv = ri.InverseMatrixDiagonal(A)
    while (errR >= tol):
        if (nIte < MAXITE):
            # aggiorno x_k
            x_k = x_k + D_inv @ r
            # calcolo il residuo
            r, errR = ri.errorRelativoResiduo(A, b, x_k)

            nIte += 1
        else:
            raise ValueError("Arrivato al massimo di iterazioni")

    stop = time.time()
    # calcolo il tempo di esecuzione
    timeIte = stop - start
    # calcolo errore relativo
    errRel = ri.errorRelativo(x, x_k)
    return errRel, nIte, timeIte


def metodo_gaus_seidelMyLU(A, b, x, tol):
    """
        Metodo di Gauss-Seidel per la risoluzione di sistemi lineari Ax = b.

        Input:
            A   : matrice dei coefficienti (sparsa, quadrata)
            b   : vettore dei termini noti
            x   : soluzione esatta (per calcolo errore)
            tol : tolleranza sull'errore relativo del residuo

        Output:
            err   : errore relativo ||x - x_k|| / ||x||
            nit   : numero di iterazioni effettuate
            tempo : tempo di esecuzione in secondi
    """
    x_k, r, errR, nIte = inizializza(A, b)
    A = A.tocsr()

    # estraggo dalla matrice A la matrice triangolare inferirore
    L = sp.tril(A).tocsr()
    start = time.time()

    while (errR >= tol):
        if (nIte < MAXITE):
            # aggiorno x_k
            x_k = x_k + ri.SolvTriangularLower(L, r)

            # calcolo il residuo
            r, errR = ri.errorRelativoResiduo(A, b, x_k)
            nIte += 1
        else:
            raise ValueError("Arrivato al massimo di iterazioni")

    stop = time.time()
    # calcolo il tempo di esecuzione
    timeIte = stop - start
    # calcolo errore relativo
    errRel = ri.errorRelativo(x, x_k)
    return errRel, nIte, timeIte


def metodo_gaus_seidel(A, b, x, tol):
    """
        Metodo di Gauss-Seidel per la risoluzione di sistemi lineari Ax = b.
        Utilizando

        Input:
            A   : matrice dei coefficienti (sparsa, quadrata)
            b   : vettore dei termini noti
            x   : soluzione esatta (per calcolo errore)
            tol : tolleranza sull'errore relativo del residuo

        Output:
            err   : errore relativo ||x - x_k|| / ||x||
            nit   : numero di iterazioni effettuate
            tempo : tempo di esecuzione in secondi
    """
    x_k, r, errR, nIte = inizializza(A, b)
    A = A.tocsr()

    # estraggo dalla matrice A la matrice triangolare inferirore
    L = sp.tril(A).tocsr()
    start = time.time()

    while (errR >= tol):
        if (nIte < MAXITE):
            # aggiorno x_k
            #x_k= x_k + ri.SolvTriangularLower(L, r)
            x_k = x_k + sp.linalg.spsolve_triangular(L, r, lower=True)
            # x_k= sp.linalg.spsolve_triangular(L, b - M @ x_k)
            # calcolo il residuo
            r, errR = ri.errorRelativoResiduo(A, b, x_k)
            nIte += 1
        else:
            raise ValueError("Arrivato al massimo di iterazioni")

    stop = time.time()
    # calcolo il tempo di esecuzione
    timeIte = stop - start
    # calcolo errore relativo
    errRel = ri.errorRelativo(x, x_k)
    return errRel, nIte, timeIte


def metodo_gradiente(A, b, x, tol):
    """
     Metodo del gradiente per la risoluzione di Ax = b.
     INPUT:
         A    : matrice del sistema
         b    : termine noto
         x    : soluzione esatta
         tol  : tolleranza
     OUTPUT:
         err  : errore relativo finale
         nit  : numero di iterazioni
         tempo: tempo impiegato
     """
    x_k, r, errR, nIte = inizializza(A, b)

    #verifico se è possibile eseguire il metodo del gradiente
    controlliDim(A,x)
    controlloGradientePossibile(A)

    start = time.time()
    while (errR >= tol):
        if (nIte < MAXITE):
            # aggiorno x_k
            rt=r.T
            alpha = (rt @ r) / (rt @ A @ r)
            x_k = x_k + alpha * r
            # calcolo il residuo
            r, errR = ri.errorRelativoResiduo(A, b, x_k)
            nIte += 1
        else:
            raise ValueError("Arrivato al massimo di iterazioni")

    stop = time.time()
    # calcolo il tempo di esecuzione
    timeIte = stop - start
    # calcolo errore relativo
    errRel = ri.errorRelativo(x, x_k)
    return errRel, nIte, timeIte


def metodo_gradiente_coniugato(A, b, x, tol):
    """
        Metodo del gradiente coniugato per la risoluzione di Ax = b.

        Input:
            A   : matrice dei coefficienti (simmetrica definita positiva)
            b   : vettore dei termini noti
            x   : soluzione esatta (per calcolo errore)
            tol : tolleranza sull'errore relativo del residuo

        Output:
            err   : errore relativo ||x - x_k|| / ||x||
            nit   : numero di iterazioni effettuate
            tempo : tempo di esecuzione in secondi
    """
    x_k, r, errR, nIte = inizializza(A, b)
    A = A.tocsr()
    d = r
    # verifico se è possibile eseguire il metodo del gradiente
    controlliDim(A, x)
    controlloGradientePossibile(A)

    start = time.time()
    while (errR >= tol):
        if (nIte < MAXITE):
            denominatore = d @ (A @ d)
            alpha = (r @ r) / denominatore

            x_k = x_k + alpha * d

            # calcolo il residuo
            r, errR = ri.errorRelativoResiduo(A, b, x_k)

            beta = ((d @ A) @ r) / denominatore
            d = r - beta * d


            nIte += 1
        else:
            raise ValueError("Arrivato al massimo di iterazioni")

    stop = time.time()
    # calcolo il tempo di esecuzione
    timeIte = stop - start
    # calcolo errore relativo
    errRel = ri.errorRelativo(x, x_k)
    return errRel, nIte, timeIte


