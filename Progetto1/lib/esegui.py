import Progetto1.lib.metodiIterativi as mt
from Progetto1.lib.metodiIterativiMulti import metodiIterativi


def routine(A, b, x, tol):
  """
  Esegue e confronta quattro metodi iterativi (Jacobi, Gauss-Seidel, Gradiente, Gradiente Coniugato)
  per la risoluzione del sistema lineare Ax = b con una data tolleranza.

  Args:
    A : array_like o matrice sparsa
      Matrice dei coefficienti del sistema lineare.
    b : array_like
      Vettore dei termini noti.
    x : array_like
      Vettore iniziale per l'iterazione.
    tol : float
      Tolleranza per il criterio di arresto (relativo all'errore).

  Returns:
    Jacobi, GaussSeidel, Gradiente, GradienteConiugato : list
      Lista contenente [errore_relativo, numero_iterazioni, tempo_trascorso] per ciascun metodo.
  """
  print(f"\nRoutine con tol: {tol}")

  # Metodo di Jacobi
  errR, nIte, time_elapsed = mt.metodo_jacobi(A, b, x, tol)
  Jacobi = [errR, nIte, time_elapsed]
  print("\nMETODO DEL JACOBI")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")

  # Metodo di Gauss-Seidel
  errR, nIte, time_elapsed = mt.metodo_gaus_seidel(A, b, x, tol)
  GaussSeidel = [errR, nIte, time_elapsed]
  print("\nMETODO DEL GAUSS-SEIDEL")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")

  #Esegue una variante personalizzata del metodo di Gauss-Seidel
  errR, nIte, time_elapsed = mt.metodo_gaus_seidelMyLU(A, b, x, tol)
  GaussSeidelMy = [errR, nIte, time_elapsed]
  print("\nMETODO DI GAUSS-SEIDEL CON RISOLUTORE PERSONALIZZATO")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")

  # Metodo del Gradiente
  errR, nIte, time_elapsed = mt.metodo_gradiente(A, b, x, tol)
  Gradiente = [errR, nIte, time_elapsed]
  print("\nMETODO DEL GRADIENTE")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")

  # Metodo del Gradiente Coniugato
  errR, nIte, time_elapsed = mt.metodo_gradiente_coniugato(A, b, x, tol)
  GradienteConiugato = [errR, nIte, time_elapsed]
  print("\nMETODO DEL GRADIENTE CONIUGATO")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")

  return Jacobi, GaussSeidel, GaussSeidelMy, Gradiente, GradienteConiugato


def runJacobi(A, b, x, tol):
  """
  Esegue il metodo di Jacobi per risolvere Ax = b.
  """
  print(f"\nRoutine con tol: {tol}")
  errR, nIte, time_elapsed = mt.metodo_jacobi(A, b, x, tol)
  print("\nMETODO DEL JACOBI")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")
  return [errR, nIte, time_elapsed]


def runGaussSeidel(A, b, x, tol):
  """
  Esegue il metodo di Gauss-Seidel per risolvere Ax = b.
  """
  print(f"\nRoutine con tol: {tol}")
  errR, nIte, time_elapsed = mt.metodo_gaus_seidel(A, b, x, tol)
  print("\nMETODO DEL GAUSS-SEIDEL")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")
  return [errR, nIte, time_elapsed]


def runGaussSeidelMySolve(A, b, x, tol):
  """
  Esegue una variante personalizzata del metodo di Gauss-Seidel
  """
  print(f"\nRoutine con tol: {tol}")
  errR, nIte, time_elapsed = mt.metodo_gaus_seidelMyLU(A, b, x, tol)
  print("\nMETODO DI GAUSS-SEIDEL CON RISOLUTORE PERSONALIZZATO")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")
  return [errR, nIte, time_elapsed]


def runGradiente(A, b, x, tol):
  """
  Esegue il metodo del Gradiente per risolvere Ax = b.
  """
  print(f"\nRoutine con tol: {tol}")
  errR, nIte, time_elapsed = mt.metodo_gradiente(A, b, x, tol)
  print("\nMETODO DEL GRADIENTE")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")
  return [errR, nIte, time_elapsed]


def runGradienteConiugato(A, b, x, tol):
  """
  Esegue il metodo del Gradiente Coniugato per risolvere Ax = b.
  """
  print(f"\nRoutine con tol: {tol}")
  errR, nIte, time_elapsed = mt.metodo_gradiente_coniugato(A, b, x, tol)
  print("\nMETODO DEL GRADIENTE CONIUGATO")
  print(f"Errore relativo per ogni iterazione: {errR}")
  print(f"Numero di iterazioni: {nIte}")
  print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")
  return [errR, nIte, time_elapsed]

def routineMulti(A, b, x, tol):
  """
  Esegue e confronta quattro metodi iterativi (Jacobi, Gauss-Seidel, Gradiente, Gradiente Coniugato)
  per la risoluzione del sistema lineare Ax = b con una data tolleranza.

  Args:
    A : array_like o matrice sparsa
      Matrice dei coefficienti del sistema lineare.
    b : array_like
      Vettore dei termini noti.
    x : array_like
      Vettore iniziale per l'iterazione.
    tol : float
      Tolleranza per il criterio di arresto (relativo all'errore).

  Returns:
    Jacobi, GaussSeidel, Gradiente, GradienteConiugato : list
      Lista contenente [errore_relativo, numero_iterazioni, tempo_trascorso] per ciascun metodo.
  """
  print(f"\nRoutine con tol: {tol}")
  risultati = []
  nomi_metodi = ["Jacobi", "Gauss-Seidel", "Gradiente", "Gradiente Coniugato"]
  for i in range(1,5):
    # Metodo di Jacobi
    errR, nIte, time_elapsed = metodiIterativi(A, b, x, tol,i)
    risultati.append( [errR, nIte, time_elapsed])
    print(f"\nMETODO DEL {nomi_metodi[i-1]}")
    print(f"Errore relativo per ogni iterazione: {errR}")
    print(f"Numero di iterazioni: {nIte}")
    print(f"Tempo di esecuzione: {time_elapsed:.6f} secondi")

  return risultati

