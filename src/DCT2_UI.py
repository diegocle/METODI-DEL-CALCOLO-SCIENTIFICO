from flask import Flask, render_template, request, flash, session
from PIL import Image
import os

from werkzeug.utils import secure_filename

from Progetto2.src.elaborazioneImmagine import ElaboraImg

# Inizializzazione dell'applicazione Flask
app = Flask(__name__)
app.secret_key = "chiave_segreta"

# Configurazione della cartella dove salvare immagini
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route("/", methods=["GET", "POST"])
def index():
    """
    Gestisce la route principale dell'applicazione.
    Permette il caricamento di un'immagine BMP e l'applicazione di una trasformata DCT2
    con parametri F (dimensione del blocco) e d (frequenza da conservare).
    """

    F = None
    d = None

    if request.method == "POST":

        # Recupera il file immagine e i parametri dal form
        file = request.files.get("image")
        F_str = request.form.get("F")
        d_str = request.form.get("d")


        # Verifica che un file sia stato selezionato o se presente in sessione
        if not file or file.filename == "":
            if session['original_image'] is not None:
                filename = session['original_image']
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            else:
                flash("Nessun file selezionato")
                return render_template("index.html")

        if file or file.filename != "":
            # Verifica che il file sia un'immagine BMP
            if not file.filename.lower().endswith(".bmp"):
                flash("Il file deve essere un'immagine BMP")
                return render_template("index.html")

            # Salva immagine
            filename = secure_filename(file.filename)
            session['original_image'] = filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

        img = Image.open(image_path)
        width, height = img.size
        # Prova a convertire i parametri in interi
        try:
            F = int(F_str)
            d = int(d_str)
        except:
            flash("F e d devono essere numeri interi")
            return render_template("index.html", original_image=filename,original_dims=f"{width} × {height}", F=F,d=d)

        # Controllo sul valore minimo di F
        if F < 1:
            flash("F deve essere un intero maggiore o uguale a 1")
            return render_template("index.html", original_image=filename,original_dims=f"{width} × {height}", F=F,d=d)

        # Calcolo del massimo valore ammesso per d
        max_d = 2 * F - 2
        if max_d < 0:
            flash("Valore di F troppo piccolo per definire il range di d")
            return render_template("index.html", original_image=filename, original_dims=f"{width} × {height}",F=F,d=d)

        # Controllo che d sia compreso nell'intervallo ammesso
        if d < 0 or d > max_d:
            flash(f"d deve essere compreso tra 0 e {max_d}")
            return render_template("index.html", original_image=filename, original_dims=f"{width} × {height}",F=F,d=d)



        try:
            # Applico DCT2 e IDCT2
            img_processed = ElaboraImg.elaboraImg(image_path, F, d)
            # Ritorna la pagina con immagine originale e immagine trasformata
            return render_template("index.html",
                                    original_image=filename,
                                    original_dims=f"{width} × {height}",
                                    processed_image=img_processed,
                                    F=F,
                                    d=d,
                                    success=True)
        except:
            flash("F troppo grande")
            return render_template("index.html", original_image=filename, original_dims=f"{width} × {height}", F=F, d=d)

    return render_template('index.html', F=None, d=None, original_image=None, processed_image=None)

if __name__ == "__main__":
    app.run()