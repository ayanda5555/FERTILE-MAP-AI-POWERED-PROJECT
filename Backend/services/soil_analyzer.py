import tensorflow as tf
import numpy as np
from PIL import Image
import os

class SoilAnalyzer:
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'soil_classifier.h5'
        )
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['chalky', 'clay', 'loamy', 'peaty', 'sandy', 'silty']
        self.properties = {
            "loamy": {
                "pH_range": "6.0 - 7.0",
                "drainage": "Good",
                "nutrient_retention": "High",
                "workability": "Easy",
                "water_holding": "Moderate to High",
                "color": "Dark brown",
                "texture": "Smooth, partly gritty"
            },
            "sandy": {
                "pH_range": "5.5 - 7.0",
                "drainage": "Excessive",
                "nutrient_retention": "Low",
                "workability": "Very Easy",
                "water_holding": "Low",
                "color": "Light brown/tan",
                "texture": "Gritty, coarse"
            },
            "clay": {
                "pH_range": "6.0 - 8.0",
                "drainage": "Poor",
                "nutrient_retention": "Very High",
                "workability": "Difficult when wet",
                "water_holding": "Very High",
                "color": "Red/brown/grey",
                "texture": "Sticky, smooth"
            },
            "silty": {
                "pH_range": "6.0 - 7.0",
                "drainage": "Moderate",
                "nutrient_retention": "Moderate to High",
                "workability": "Moderate",
                "water_holding": "High",
                "color": "Dark brown",
                "texture": "Silky, flour-like"
            },
            "peaty": {
                "pH_range": "3.5 - 5.5",
                "drainage": "Poor (waterlogged)",
                "nutrient_retention": "High",
                "workability": "Easy when drained",
                "water_holding": "Very High",
                "color": "Very dark/black",
                "texture": "Spongy, fibrous"
            },
            "chalky": {
                "pH_range": "7.5 - 8.5",
                "drainage": "Good to Excessive",
                "nutrient_retention": "Low",
                "workability": "Moderate",
                "water_holding": "Low to Moderate",
                "color": "Pale white/grey",
                "texture": "Stony, gritty"
            }
        }

    def predict(self, image_path):
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_class = self.classes[predicted_index]
        confidence = float(predictions[0][predicted_index])

        # Get all class probabilities
        all_predictions = {
            self.classes[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(self.classes))
        }

        return {
            "soil_type": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "properties": self.properties.get(predicted_class, {}),
            "all_predictions": all_predictions
        }

# Singleton instance
analyzer = SoilAnalyzer()
