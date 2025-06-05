import numpy as np
import scipy.sparse as sp


def SolvTriangularLower(A, b):
  """
  Risolve un sistema lineare Ax = b dove A è una matrice triangolare inferiore .

  INPUT:
  A : matrice triangolare inferiore.
  b : vettore colonna dei termini noti.

  OUTPUT:
  x : soluzione del sistema lineare
  """

  n = A.shape[0]
  x = np.zeros(n)
  D = A.diagonal()

  # Calcolo del primo elemento della soluzione
  x[0] = b[0] / D[0]

  # Sostituzione in avanti per risolvere il sistema triangolare inferiore
  for i in range(1,n):
      # Indici degli elementi della riga i-esima nel formato CSR
      row_start = A.indptr[i]
      row_end = A.indptr[i + 1]

      # Indici di colonna degli elementi non nulli
      cols = A.indices[row_start:row_end]
      #valori non nulli
      data = A.data[row_start:row_end]

      sum_ax = 0.0
      # Calcolo del prodotto scalare tra riga i-esima di A e x
      for j, col in enumerate(cols):
          if col < i:
              sum_ax += data[j] * x[col]

      # Calcolo dell'x_i
      x[i] = (b[i] - sum_ax) / D[i]
  return x



def InverseMatrixDiagonal(A):
  """
   calcolo la matrice inversa della matrice formata dalla estrazione della diagonale di A

        INPUT:
        A : matrice sparsa dalla quale si estrae la diagonale.

        OUTPUT:
        D_inv : matrice diagonale inversa
   """

  #estraggo coefficenti diagonale
  coefficentiD= A.diagonal()
  #calcolo inversi
  coefficentiD_inv = 1/coefficentiD
  return sp.diags(coefficentiD_inv, 0, format='csr')

def errorRelativoResiduo(A,b,x):
    """
        Calcola il residuo r = b - Ax e l'errore relativo del residuo.

        INPUT:
        A: matrice del sistema.
        b: termine noto.
        x: vettore approssimato della soluzione.

        OUTPUT:
        r: residuo.
        errR: errore relativo del residuo.
    """
    #calcolo il residuo
    r= b - (A @ x)
    #calcolo errore relativo (RESIDUO)
    errR= (np.linalg.norm(r))/np.linalg.norm(b)
    return r,errR

def errorRelativo(x,x_k):
  """
    Calcola l'errore relativo tra la soluzione esatta x e quella approssimata x_k.

    INPUT:
        x: soluzione esatta.
        x_k: soluzione approssimata.

    OUTPUT:
        errore relativo tra x e x_k.
  """
  return np.linalg.norm( x-x_k)/np.linalg.norm(x)

def inizializza(A,b):
  """
     Inizializza la soluzione approssimata x_k, calcola il residuo e l'errore relativo iniziale.

    INPUT:
        A: matrice del sistema.
        x: soluzione esatta (usata per il calcolo dell'errore).
        b: termine noto.

    OUTPUT:
        x_k: soluzione iniziale (tutti zeri).
        r: residuo iniziale.
        errR: errore relativo iniziale.
        nIte: contatore delle iterazioni (inizializzato a 0).
  """
  #verifico che la matrice è salvata in formato sparso
  if not sp.issparse(A):
   A = sp.csr_array(A).tocsr()

  #inizializzo x_0
  x_k = np.zeros(A.shape[0])

  #calcolo residuo ed errore relativo(RESIDUO)
  r,errR=errorRelativoResiduo(A,b,x_k)

  #inizializzo a 0 numero di iterazioni
  nIte=0
  return x_k,r,errR,nIte

