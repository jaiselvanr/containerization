from flask import Flask,request
import pandas as pd
import pickle
import sklearn


my_model=pickle.load(open('full_pipeline','rb'))
#feat_num=pickle.load(open('feat_num1','rb'))
#feat_cat=pickle.load(open('feat_cat1','rb'))

app=Flask(__name__)

@app.route('/')
def hello():
    return 'Lets explore FLASK'

@app.route('/predict1')
def pred():
    Married=request.args.get('Married')
    Education=request.args.get('Education')
    ApplicantIncome=request.args.get('ApplicantIncome')
    LoanAmount=request.args.get('LoanAmount')
    Credit_History=request.args.get('Credit_History')
  
    test=pd.DataFrame([[Married,Education,ApplicantIncome,LoanAmount,Credit_History]])
    test.columns=['Married','Education','ApplicantIncome','LoanAmount','Credit_History']
    
    ypred=my_model.predict(test)
    
    return "The Loan Rejection Probability:"+str(ypred)
    



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001)

#http://127.0.0.1:5000/predict1?Married=Yes&Education=Graduate&ApplicantIncome=1500&LoanAmount=150&Credit_History=1