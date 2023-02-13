from flask import Flask, request, jsonify
import pickle
import numpy as np
# import sklearn

model=pickle.load(open("model1.pkl","rb"))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"
    


@app.route('/predict', methods=['POST'])
def predict():
    height = request.form.get('height')
    width = request.form.get('width')

    input_query = np.array([[height, width]])
    result = model.predict(input_query)[0]

    return jsonify({'Weight': str(result)})


if __name__ == '__main__':
    app.run(debug=True, port=8001)