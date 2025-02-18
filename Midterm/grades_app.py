import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load artifacts using pickle
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        math = float(request.form['math_score'])
        reading = float(request.form['reading_score'])
        writing = float(request.form['writing_score'])
        
        # Create feature array
        features = [math, reading, writing, math + reading + writing]
        
        # Make prediction
        pred_num = model.predict([features])[0]
        pred_group = le.inverse_transform([pred_num])[0]
        
        return f'Predicted Ethnicity Group: {pred_group}'
    
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)