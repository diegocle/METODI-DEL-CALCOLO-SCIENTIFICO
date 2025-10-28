

<header>

<!--
  <<< Author notes: Diego Clemente >>>
-->
# Il codice per il primo progetto si trova nella cartella `Progetto1`.
# Libreria di Solutori Iterativi per Matrici Simmetriche Positive

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)


## Descrizione

Mini-libreria per risolvere sistemi lineari \(Ax = b\) con **matrici simmetriche definite positive** usando:

- Jacobi
- Gauss-Seidel
- Gradiente
- Gradiente Coniugato

Tutti i metodi partono dal vettore nullo e si arrestano quando:  

\[
\frac{\|Ax^{(k)} - b\|}{\|b\|} < \text{tol}
\]

oppure se il numero massimo di iterazioni `maxIter` è superato.

---



# Il codice per il secondo progetto è contenuto nella cartella `Progetto2`.
## Compressione di immagini in toni di grigio con DCT2

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)


### Descrizione del progetto

Lo scopo di questo progetto è **implementare la DCT2 (Discrete Cosine Transform bidimensionale)** in un ambiente open source e studiare gli effetti di un algoritmo di compressione tipo JPEG (senza usare una matrice di quantizzazione) su immagini in scala di grigi.  

Il progetto comprende:
- Implementazione del codice DCT2;
- Confronto con la DCT2 della libreria utilizzata (versione “fast” basata su FFT);
- Scrittura di una relazione sui risultati ottenuti.

---

### Prima parte: Implementazione DCT2 e confronto tempi

1. Implementare la DCT2 come spiegata a lezione.
2. Creare array quadrati \(N \times N\) con \(N\) crescente.
3. Misurare i tempi di esecuzione della DCT2 “fatta in casa” e della DCT2 veloce della libreria.
4. Rappresentare i risultati su **grafico semilogaritmico** (ordinate in scala log) al variare di \(N\).  

**Attesi:**
- DCT2 fatta in casa: complessità ~ \(O(N^3)\)
- DCT2 versione fast: complessità ~ \(O(N^2 \log N)\)

---

### Seconda parte: Software di compressione per immagini

Il software permette di:

1. Selezionare un’immagine `.bmp` in scala di grigi dal filesystem.
2. Scegliere:
   - `F`: dimensione dei blocchi quadrati \(F \times F\)
   - `d`: soglia di taglio delle frequenze (0 ≤ d ≤ 2F − 2)
3. Suddividere l’immagine in blocchi \(F \times F\), scartando eventuali avanzi.
4. Per ogni blocco:
   - Applicare DCT2 (libreria)
   - Eliminare coefficienti `ckℓ` con `k + ℓ ≥ d`
   - Applicare DCT2 inversa
   - Arrotondare valori e limitare a [0, 255]
5. Ricomporre l’immagine e visualizzare **originale e compressa affiancate**.

---


</header>



