import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RF_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('pro.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    if int_features[-1] >10:
        int_features[-1] = 10

    print(f'------------------------------------------------------{int_features[-1]}-------------------------------------------------')
    final_features = [np.array(int_features)]   
    prediction = model.predict(final_features)
   
    if prediction == 0:
        return render_template('pro.html', prediction_text = f"Result: No, You cannot get the Loan.")

    else:
        return render_template('pro.html', prediction_text = f'Result: Yes, You can get the Loan.')

if __name__ == '__main__':
    app.run(debug=True)