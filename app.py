import os
import base64
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from classes_tr import CLASSES_TR  


CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Modeli yükleme
model = load_model('model/cifar100_resnet_mixed_precision.keras')


@app.route("/")
def root():
    return redirect(url_for("splash"))


@app.route("/splash")
def splash():
    return render_template("splash.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/camera')
def camera_page():
    return render_template('camera.html')


@app.route('/words')
def words():
    return render_template('words.html', words=CLASSES_TR)


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASSES[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return predicted_label, confidence



@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return redirect('/upload')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)
    label_tr = CLASSES_TR.get(label, "Bilinmiyor")

    return render_template('result.html',
                           label=label,
                           label_tr=label_tr,
                           confidence=round(confidence * 100, 2),
                           image_url=filepath)



@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data_url = request.form['image_data']
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)


    filename = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)


    label, confidence = predict_image(filepath)
    label_tr = CLASSES_TR.get(label, "Bilinmiyor")

    return render_template('result.html',
                           label=label,
                           label_tr=label_tr,
                           confidence=round(confidence * 100, 2),
                           image_url=filepath)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
