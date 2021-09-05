from flask import Flask, render_template, request, jsonify, Response, url_for, redirect
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, request, jsonify, Response, url_for, redirect
from flask_cors import CORS, cross_origin
import pandas as pd
from logging import FileHandler,WARNING
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport
from sklearn.linear_model import Ridge,Lasso,RidgeCV,ElasticNet,ElasticNetCV,LinearRegression,LassoCV
from sklearn.model_selection import train_test_split
import pickle
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
free_status = True
df = pd.read_csv('ai4i2020.csv')
app = Flask(__name__,template_folder="templates")
print(df.columns)

@cross_origin()
@app.route('/', methods=["GET"])
def home():
    return render_template('home.html')
def linear_regression_model(x_train, x_test, y_train, y_test):
    lr=LinearRegression()
    (lr.fit(x_train,y_train))
    print(lr.score(x_train, y_train))
    print(lr.score(x_test, y_test))
    print(lr.coef_,lr.intercept_)
    print(lr.predict([[298.1,308.6,1551,42.8,0,0,0,0,0,0]]))
    file = 'linear.sav'
    pickle.dump(lr, open(file, 'wb'))
    return lr.score(x_test, y_test),lr


def lasso_model(x_train, x_test, y_train, y_test):
    lassocv = LassoCV(alphas=None, cv=10, max_iter=200000, normalize=True)
    lassocv.fit(x_train, y_train)
    lasso = Lasso(alpha=lassocv.alpha_, )
    lasso.fit(x_train, y_train)
    lasso.score(x_test, y_test)
    file = 'lasso.sav'
    pickle.dump(lasso, open(file, 'wb'))
    return lasso.score(x_test, y_test),lasso
def ridge_model(x_train, x_test, y_train, y_test):
    ridgecv = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10, normalize=True)
    ridgecv.fit(x_train, y_train)
    ridge = Ridge(alpha=ridgecv.alpha_)
    ridge.fit(x_train, y_train)
    ridge.score(x_test, y_test)
    file = 'ridge.sav'
    pickle.dump(ridge, open(file, 'wb'))
    return  ridge.score(x_test, y_test),ridge
def elastic_net_model(x_train, x_test, y_train, y_test):
    ###ElasticNet
    elastic = ElasticNetCV(alphas=None, cv=10)
    elastic.fit(x_train, y_train)
    elastic_lr = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio)
    elastic_lr.fit(x_train, y_train)
    elastic_lr.score(x_test, y_test)
    file = 'elasticnet.sav'
    pickle.dump(elastic_lr, open(file, 'wb'))
    return elastic_lr.score(x_test, y_test),elastic_lr
@cross_origin()
@app.route('/showReport', methods=["GET"])
def generate_pandas_profiling_report():
    report = ProfileReport(df)
    report.to_file('templates/output.html')

    return render_template(r'output.html')

def training_testing():
    x,y=select_features()
    x_train, x_test, y_train, y_test = train_test_split(feature_scaling(x),y, test_size=0.25, random_state=100)
    return x_train, x_test, y_train, y_test

def feature_scaling(x):
    scaler = StandardScaler()
    arr=scaler.fit_transform(x)
    pickle.dump(scaler,open('scaler.sav','wb'))
    return arr

@cross_origin()
@app.route('/modelAccuracy', methods=["GET"])
def model_buliding():
    x_train, x_test, y_train, y_test=training_testing()
    lr,_=linear_regression_model(x_train, x_test, y_train, y_test)
    lasso,_=lasso_model(x_train, x_test, y_train, y_test)
    ridge,_=ridge_model(x_train, x_test, y_train, y_test)
    elastic,_=elastic_net_model(x_train, x_test, y_train, y_test)
    new_dict={"Linear Model's Accuracy":lr,"Lasso Model's Accuracy":lasso,
              "Ridge Model's Accuracy":ridge,"ElasticNet Model's Accuracy":elastic}
    print(new_dict)
    df=pd.DataFrame(new_dict,index=[0])

    print(df)
    return render_template('df_to_html.html',tables=[df.to_html(classes='data', header="true")])
@cross_origin()
@app.route('/showDataset', methods=["GET"])
def df_to_html():
    return render_template('df_to_html.html', tables=[df.to_html(classes='data', header="true")])

def select_features():
    x=x=df.drop(['UDI', 'Product ID', 'Type', 'Air temperature [K]'],axis=1)
    y = df['Air temperature [K]']
    return x,y


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    model_buliding()
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            process_temperature = float(request.form['process_temperature'])
            rpm = float(request.form['rpm'])
            torque = float(request.form['torque'])
            tool_wear = float(request.form['tool_wear'])
            mf = request.form['mf']
            twf = request.form['twf']
            hdf = request.form['hdf']
            pwf = request.form['pwf']
            osf = request.form['osf']
            rnf = request.form['rnf']
            if (mf == 'yes'):
                mf = 1
            else:
                mf = 0
            if (twf == 'yes'):
                twf = 1
            else:
                twf = 0
            if (hdf == 'yes'):
                hdf = 1
            else:
                hdf = 0
            if (pwf == 'yes'):
                pwf = 1
            else:
                pwf = 0
            if (osf == 'yes'):
                osf = 1
            else:
                osf = 0
            if (rnf == 'yes'):
                rnf = 1
            else:
                rnf = 0
            model=request.form['model']
            if model=='lr':

                filename = 'linear.sav'
            elif model=='lasso':
                filename = 'lasso.sav'
            elif model=='ridge':
                filename = 'ridge.sav'
            elif model=='elasticnet':
                filename = 'elastic.sav'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            # predictions using the loaded model file
            scaler=pickle.load(open('scaler.sav','rb'))
            prediction = loaded_model.predict(
                    scaler.fit_transform([[process_temperature,rpm, torque, tool_wear, mf, twf, hdf,pwf,osf,rnf]]))
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html', prediction=(round(prediction[0])))



        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')

if __name__ == "__main__":
    # app.run()  # running the app on the local machine on port 8000
    app.run(debug=True)

    # model_buliding()
    # model_html()