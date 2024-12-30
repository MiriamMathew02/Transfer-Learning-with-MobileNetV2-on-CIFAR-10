from flask import Flask, render_template, url_for, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model/mobilenetv2_cifar10.h5')

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index2.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = Image.open(imagefile.stream)
    img = img.resize((150, 150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CIFAR_CLASSES[np.argmax(prediction)]
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

