from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
#intilization of flask
#reg = LinearRegression()
app = Flask(__name__)
model = pickle.load(open("lr_model.pkl","rb"))

@app.route("/")
def hello():
    return render_template("index.html")
    
@app.route("/predict",methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template("index.html",prediction_text=prediction)

if __name__=="__main__":
    app.run(debug=True)
