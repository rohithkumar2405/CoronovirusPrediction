from flask import Flask,render_template,url_for,request
import pickle
import numpy as np

app=Flask(__name__)

model_path = 'Trained_Model/rf_model.pickle'
model = pickle.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    
    Breathing_Problem = int(request.form['Breathing_Problem'])
    Fever = int(request.form['Fever'])
    Dry_Cough = int(request.form['Dry_Cough'])
    Sore_throat = int(request.form['Sore_throat'])
    Running_nose = int(request.form['Running_nose'])
    Asthma = float(request.form['Asthma'])
    Chronic_Lung_Disease = float(request.form['Chronic_Lung_Disease'])
    Headache = float(request.form['Headache'])
    

    query = np.array([[Breathing_Problem, Fever, Dry_Cough, Sore_throat, Running_nose, 
                        Asthma, Chronic_Lung_Disease, Headache]])

    prediction = model.predict(query)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
