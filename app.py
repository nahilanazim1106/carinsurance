from flask import Flask,render_template,request
import pickle
import numpy as np
model = pickle.load(open('XGB.pkl','rb'))
regmodel = pickle.load(open('XGBR.pkl','rb'))
lep = pickle.load(open('lep.pkl','rb'))
ler = pickle.load(open('ler.pkl','rb'))
leu = pickle.load(open('leu.pkl','rb'))
lerp = pickle.load(open('lerp.pkl','rb'))
lerr = pickle.load(open('lerr.pkl','rb'))
leru = pickle.load(open('leru.pkl','rb'))
sb = pickle.load(open('sb.pkl','rb'))
scg = pickle.load(open('scg.pkl','rb'))
syoj = pickle.load(open('syoj.pkl','rb'))
rsb = pickle.load(open('rsb.pkl','rb'))
rsc = pickle.load(open('rsc.pkl','rb'))
rsyoj = pickle.load(open('rsyoj.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    KIDSDRIV =int(request.form.get('kidsdrive'))
    YOJ =float(request.form.get("yoj"))
    PARENT1 =request.form.get('parent1')
    BLUEBOOK =float(request.form.get("bluebook"))
    TIF =int(request.form.get("tif"))
    CLM_FREQ =int(request.form.get("freq"))
    REVOKED = request.form.get("revoked")
    MVR_PTS = int(request.form.get("mvr"))
    CAR_AGE = float(request.form.get("age"))
    URBANICITY = request.form.get("ur")
    # Perform classification prediction
    claim_status = model.predict(np.array([KIDSDRIV, syoj.transform([[YOJ]])[0,0], lep.transform([PARENT1])[0], sb.transform([[BLUEBOOK]])[0,0], TIF, CLM_FREQ, ler.transform([[REVOKED]])[0], MVR_PTS, scg.transform([[CAR_AGE]])[0,0], leu.transform([URBANICITY ])[0]]).reshape(1,-1))[0]
    # Use the classification input with prediction as an input to the regression model)
    #Perform regression prediction
    input_data = np.array([KIDSDRIV, syoj.transform([[YOJ]])[0,0], lep.transform([PARENT1])[0], sb.transform([[BLUEBOOK]])[0,0], TIF, CLM_FREQ, ler.transform([[REVOKED]])[0], scg.transform([[CAR_AGE]])[0,0], leu.transform([URBANICITY ])[0]]).reshape(1,-1)
    input = np.concatenate((input_data, np.array([[claim_status]])),axis=1)
    claim_amount = regmodel.predict(input)
    
    if claim_status == 1:
        claim_amount = claim_amount
    else:
        claim_amount = 0
    return render_template('result.html', predicted=claim_status, claim_amount=claim_amount)
    
    
    
if __name__ == '__main__':
    app.run(debug=True)
