# 🌟 Proyek SEHATI 

Welcome to our GitHub repository! 🎉 We are a team working on Capstone Project at Bangkit and we build an AI-powered health monitoring application designed to support users.  
---

## 📝 About the Project

- **Project Title:** SEHATI  
- **Short Description:** ~

---

## 🛠️ Teknologi yang Digunakan

- **Machine Learning:** ~
- **Cloud Computing:** ~
- **Mobile Development:** ~
- **Other:** ~

---

## 🤝 Our Team

Kami terdiri dari anggota dengan keahlian di bidang yang beragam untuk menyukseskan proyek ini:

- **Machine Learning Engineers:**
  - Wahyu Azizi
  - Dewi Suciningrum
  - Ninda Kartika Putri
- **Cloud Computing Engineers:**
  - Andika Dermawan Saputra
  - Azizah Bahirotun Ni'am	
- **Mobile Developer:**
  - Nur Kholis Wakhid

---

# Machine Learning Documentation for Sehati Capstone Project

## Overview
Sehati is a health application that provides users with personalized health recommendations and symptom self-diagnosis features. This document outlines the implementation, usage, and model management for the Machine Learning (ML) component of the Sehati app.

---

## ML Model Details
- **Model Type:** Deep Neural Network (DNN)
- **Purpose:** Predict diseases based on user-input symptoms and provide health recommendations.
- **Framework:** TensorFlow/Keras
- **Model File Format:** `.keras`
- **Training Dataset:** [Fetal Health Dataset](#) (2126 records) with preprocessing for symptom-based classification.

---

## Saving the Model
Use the following code to save the trained model in `.keras` format:
```python
model.save('sehati_model.keras')
```

---

## Loading the Model
To load the saved model for inference or further training:
```python
from keras.models import load_model

# Load the model
model = load_model('sehati_model.keras')

# Example usage
predictions = model.predict(input_data)
```

---

## Making Predictions
The following snippet demonstrates how to prepare input and use the model for predictions:
```python
import numpy as np

# Example input (binary array where 1 indicates presence of a symptom)
input_data = np.array([[0, 1, 0, 1, ...]])  # Replace '...' with feature values

# Perform prediction
predictions = model.predict(input_data)

# Get the class with the highest probability
predicted_class = np.argmax(predictions)
confidence = predictions[0][predicted_class] * 100

print("Predicted Class:", predicted_class)
print("Confidence:", confidence, "%")
```

---

## Model Updates and Compatibility
- **Why `.keras` Format?**
  - `.keras` is the recommended format for TensorFlow/Keras models.
  - It supports advanced features like custom layers and configurations.
  - Easier to load and manage in future Keras/TensorFlow versions.

- **Transition from Legacy `.h5` Format:**
  If previously using `.h5` format, migrate to `.keras` with:
  ```python
  model.save('model_name.keras')
  ```

---

## Health Recommendation Feature
- **Current Implementation:**
  Recommendations are manually sent to users based on predictions due to GCP credit limitations.
- **Future Plans:**
  Integrate generative AI with a fine-tuned Gemini model to provide personalized recommendations.

---

## Troubleshooting
### Common Warning
**Warning:**
```
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
```
**Solution:**
Switch to the `.keras` format for saving models:
```python
model.save('my_model.keras')
```

---

## Contributions
- **Machine Learning Team:** Responsible for model development, training, and deployment.
- **Cloud Computing Team:** Ensures efficient deployment using Google Cloud Platform (GCP).

---

## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
