import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Initialize Flask app
app = Flask(__name__)

# Load the model
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

# Load model weights
weights_path = os.path.join(os.path.dirname(__file__), 'vgg_unfrozen.h5')
model_03.load_weights(weights_path)

# Clear unused variables and run garbage collection
import gc
gc.collect()

# Function to get class names
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Function to process image and get prediction
def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    # Get prediction
    result = model_03.predict(input_img)
    
    # Get the predicted class (highest probability)
    classNo = np.argmax(result, axis=1)[0]
    return classNo

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Ensure uploads directory exists
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)
        
        # Get prediction
        classNo = getResult(file_path)
        result = get_className(classNo)

        # Send the result back as JSON
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
