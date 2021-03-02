import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('LogisticR_Classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    for rendering results on HTML GUI
    '''
    
    int_features = [ float(x) for x in request.form.values() ]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text = 'Based on the input data, the donor %s the blood in the future.' %("WILL DONATE" if output == 1 else "WILL NOT DONATE"))


if __name__ == "__main__":
    app.run(debug=True)
    
