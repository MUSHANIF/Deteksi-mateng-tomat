from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = tf.keras.models.load_model('model/tomato_model.h5')

# Kelas prediksi
class_names = ['Setengah Matang', 'Mentah', 'Matang']

@app.route('/')
def index():
    return render_template('index.html', img_path=None, hist_path=None, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Pastikan folder uploads ada
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Simpan file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        # Jika file dengan nama yang sama sudah ada, tambahkan angka unik
        base, ext = os.path.splitext(file.filename)
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{counter}{ext}")
            counter += 1

        file.save(filepath)

        # Preprocess gambar
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Buat grafik RGB
        img_data = image.img_to_array(image.load_img(filepath)) / 255.0
        plt.figure(figsize=(6, 4))
        for i, color in enumerate(['red', 'green', 'blue']):
            plt.plot(img_data[:, :, i].mean(axis=0), color=color, label=f'{color.upper()}')
        plt.legend(loc='upper right')
        plt.title('RGB Color Spread')
        plt.xlabel('Pixel Width')
        plt.ylabel('Intensity')
        plt.tight_layout()
        hist_path = os.path.join(app.config['UPLOAD_FOLDER'], 'color_graph.png')
        plt.savefig(hist_path)
        plt.close()

        # Pass data ke HTML
        return render_template(
            'index.html',
            prediction=predicted_class,
            img_path=f'uploads/{os.path.basename(filepath)}',
            hist_path=f'uploads/color_graph.png'
        )

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host="0.0.0.0", port=5000, debug=True)
   
    # Ensure the server keeps running even if the upload attempts are repeated
    app.run(debug=True, use_reloader=False)