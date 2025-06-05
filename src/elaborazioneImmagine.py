
import os
import numpy as np
from PIL import Image

from Progetto2.src.DCT2 import dct2_lib, idct2

class ElaboraImg:
    """
        Classe per elaborazione di immagini in scala di grigi tramite
        compressione con DCT (Discrete Cosine Transform) a blocchi.
        """

    @staticmethod
    def __apriImmagine(image_path):
        """
        Apre un'immagine da disco, la converte in scala di grigi e
        restituisce la matrice corrispondente come array NumPy.

        :param image_path: percorso del file immagine
        :return: array 2D NumPy contenente i pixel in scala di grigi
        """
        img = Image.open(image_path).convert('L')
        array = np.array(img)
        return array

    @staticmethod
    def __split_matrix_into_blocks(M, f):
        """
        Divide una matrice M (immagine) in blocchi quadrati f x f.

        :param M: matrice dell'immagine (array 2D)
        :param f: dimensione del blocco (es. 8 per blocchi 8x8)
        :return: lista di blocchi f x f estratti dalla matrice
        """
        rig, col = M.shape
        blocchi = []
        # Scorri righe e colonne a passi di f per estrarre i blocchi f x f
        for i in range(rig // f):
            for j in range(col // f):
                blocchi.append(M[f*i:f*(i+1), f*j:f*(j+1)])
        return blocchi

    @staticmethod
    def __assemblaBlocchi(colonne, f, ff_blocchi, righe):
        """
        Ricompone una matrice immagine a partire da una lista di blocchi.

        :param colonne: numero di colonne dell'immagine originale
        :param f: dimensione del blocco
        :param ff_blocchi: lista di blocchi ricostruiti (post-IDCT)
        :param righe: numero di righe dell'immagine originale
        :return: array 2D ricostruito unendo i blocchi
        """
        righe_assemblate = []
        blocchi_per_riga = colonne // f

        # Ricostruisce ogni riga unendo i blocchi
        for i in range(righe // f):
            # Estrai i blocchi relativi alla riga corrente
            riga_blocchi = ff_blocchi[i * blocchi_per_riga: (i + 1) * blocchi_per_riga]
            # Unisci i blocchi (ricostruisce una riga intera)
            riga = np.hstack(riga_blocchi)
            righe_assemblate.append(riga)

        # Unisci tutte le righe per ottenere l'immagine completa
        matrice_img = np.vstack(righe_assemblate).astype(np.uint8)
        return matrice_img

    @staticmethod
    def __applicaIDCT2(c_blocchi):
        """
        Applica la trasformata discreta del coseno inversa (IDCT2) su ciascun blocco.

        :param c_blocchi: lista di blocchi con coefficienti DCT compressi
        :return: lista di blocchi ricostruiti (immagine decompressa)
        """
        ff_blocchi = []
        for c in c_blocchi:
            #applico IDCT2
            ff = idct2(c)
            # Limita tra 0 e 255 e arrotonda al intero più vicino
            ff = np.clip(np.round(ff), 0, 255)
            ff_blocchi.append(ff)

        return ff_blocchi

    @staticmethod
    def __applicaDCT(blocchi, d):
        """
        Applica la DCT2 a ciascun blocco e azzera i coefficienti di alta frequenza.

        La soglia 'd' indica la diagonale oltre la quale i coefficienti vengono annullati
        (cioè mantiene solo quelli in cui i + j < d).

        :param blocchi: lista di blocchi f x f da comprimere
        :param d: soglia di compressione (frequenze alte azzerate)
        :return: lista di blocchi trasformati e compressi
        """

        c_blocchi = []
        #applico DCT2 per ogni blocco
        for b in blocchi:
            c = dct2_lib(b)
            ri, co = c.shape
            # Applica maschera a diagonale: azzera coefficienti con i + j >= d
            mask = np.fromfunction(lambda i, j: (i + j) < d, (ri, co))
            c *= mask

            c_blocchi.append(c)
        return c_blocchi

    @staticmethod
    def elaboraImg(image_path, f, d):
        """
        Metodo per elaborare un'immagine:
        - Apre l'immagine e la converte in scala di grigi.
        - Divide l'immagine in blocchi f x f.
        - Applica la DCT e comprime eliminando frequenze in eccesso.
        - Applica la IDCT per ricostruire i blocchi.
        - Ricompone l'immagine.
        - Salva l'immagine ricostruita su disco (formato BMP).

        :param image_path: percorso dell'immagine sorgente
        :param f: dimensione dei blocchi (es. 8)
        :param d: soglia di compressione (es. 10 = mantiene solo coeff. a bassa frequenza)
        :return: nome del file immagine salvato
        """

        # 1. Carica e converte immagine in matrice
        M = ElaboraImg.__apriImmagine(image_path)
        righe, colonne = M.shape

        # 2. Suddivide in blocchi f x f
        blocchi = ElaboraImg.__split_matrix_into_blocks(M, f)

        # 3. Applica DCT con compressione
        c_blocchi = ElaboraImg.__applicaDCT(blocchi, d)

        # 4. Applica IDCT per decodifica
        ff_blocchi = ElaboraImg.__applicaIDCT2(c_blocchi)

        # 5. Ricompone l'immagine dai blocchi
        matrice_img = ElaboraImg.__assemblaBlocchi(colonne, f, ff_blocchi, righe)

        # 6. Salva l'immagine ricostruita in formato BMP
        img = Image.fromarray(matrice_img.astype(np.uint8), mode='L')
        nome_senza_ext = os.path.splitext(os.path.basename(image_path))[0]
        nome_modificato = f'Modifica{nome_senza_ext}.bmp'
        img.save(os.path.join('static', nome_modificato))

        return nome_modificato


