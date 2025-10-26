from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Ignore any input and just return this simple response
    return jsonify({"test": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
