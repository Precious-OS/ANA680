from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('rbf_svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    features = [float(request.form[feature]) for feature in ['Bare_nuclei', 'Normal_nucleoli', 'Clump_thickness', 'Uniformity_of_cell_shape', 'Single_epithelial_cell_size']]
    
    # Scale the input
    scaled_features = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(scaled_features)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    
    return result

if __name__ == '__main__':
    app.run(debug=True)