from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()    # Get your posted JSON
    return jsonify({"received": input_data})  # Echo back whatever you sent

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
