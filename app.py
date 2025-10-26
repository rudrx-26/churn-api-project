from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf

# Load model and preprocessors
model = tf.keras.models.load_model('churn_ann_model.h5')
with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    # Example: input_data must have all 11 input fields and one-hot columns
    vals = np.array([[
        input_data['CreditScore'],
        le_gender.transform([input_data['Gender']])[0],
        input_data['Age'],
        input_data['Tenure'],
        input_data['Balance'],
        input_data['NumOfProducts'],
        input_data['HasCrCard'],
        input_data['IsActiveMember'],
        input_data['EstimatedSalary'],
        input_data.get('Geography_Germany', 0),
        input_data.get('Geography_Spain', 0)
    ]])
    vals[:, :8] = scaler.transform(vals[:, :8])
    prob = model.predict(vals)[0][0]
    pred = int(prob > 0.5)
    return jsonify({'churn_probability': float(prob), 'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
