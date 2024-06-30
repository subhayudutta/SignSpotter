import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import pandas as pd

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.class_names = self.load_class_names()

    def load_class_names(self):
        # Load class names from CSV
        csv_path = os.path.join("artifacts", "data_ingestion", "Indian-Traffic Sign-Dataset", "traffic_sign.csv")
        df = pd.read_csv(csv_path)
        class_names = df.set_index('ClassId')['Name'].to_dict()
        return class_names

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        result = np.argmax(model.predict(test_image), axis=1)
        class_id = result[0]

        # Get the prediction name from class ID
        prediction = self.class_names.get(class_id, "Unknown")

        return [{"image": prediction}]

# Example usage:
# pipeline = PredictionPipeline("test.png")
# prediction = pipeline.predict()
# print(prediction)
