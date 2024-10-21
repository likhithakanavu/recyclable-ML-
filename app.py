from flask import Flask, request, render_template, redirect
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from io import BytesIO
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('recycle.h5')

# Define image dimensions expected by the model
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Adjust as per your model's input shape

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Process the uploaded image without saving it
        img = Image.open(BytesIO(file.read()))

        # Convert image to RGB mode if it's in RGBA or other mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Preprocess the image for the model
        image = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match model input shape
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map prediction to the respective class
        class_labels = {0: 'Organic', 1: 'Recyclable'}  # Adjust according to your model
        result = class_labels[predicted_class]

        return render_template('result.html', prediction=result)

# Main function to run the app
if __name__=='__main__':
    app.run(debug=True)
