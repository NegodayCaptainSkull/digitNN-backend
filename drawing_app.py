from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://main--digitnn.netlify.app"}}, allow_headers=["Content-Type", "Authorization", "X-Requested-With"])

# Load the TensorFlow model
model = tf.keras.models.load_model('model/digits_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded.split(',')[1])  # Remove the prefix from the data URL
    image = Image.open(io.BytesIO(decoded))

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Create a white background image
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = background
    else:
        image = image.convert('RGB')

    image = image.resize((28, 28)).convert('L')  # Assuming a 28x28 grayscale image

    image = ImageOps.invert(image)
    
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_normalized = image_array / 255.0  # Normalize pixel values to [0, 1]
    
    # Reshape and prepare for the model
    image_reshaped = image_normalized.reshape(1, 28, 28, 1)  # Assuming your model expects this shape
    
    # Predict
    prediction = model.predict(image_reshaped)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    # Convert confidence to percentage
    confidence_percentage = round(confidence * 100, 2)
    
    response = {
        'prediction': int(prediction.argmax()),
        'confidence': confidence_percentage
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
