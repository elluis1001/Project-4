# import dependencies to run Flask app and host it on publicly accessible colab URL
from flask import *
from google.colab import output
from google.colab.output import eval_js

# initialize Flask app
app=Flask(__name__)


# Static Root API for Diabetes Prediction Model
@app.route('/')
def home():
    return 'Root API for Diabetes Prediction Model'

# Dynamic API that will take parameters of subject through Web UI and leverage ML Model above to return probability & prediction for being diabetic
@app.route('/api/v1.0/predict/<gender>/<age>/<hypertension>/<heart_disease>/<smoking_history>/<bmi>/<HbA1c_level>/<blood_glucose_level>')
def predict(gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level):
    # create a tuple from input parameters and convert them to int or float as they are treated as string when passed from Web UI
    data = [(int(gender),int(age),int(hypertension),int(heart_disease),int(smoking_history),float(bmi),float(HbA1c_level),int(blood_glucose_level))]
    columns = ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']

    # Create spark dataframe for the input parameters
    test_df = spark.createDataFrame(data,columns)

    # invoke ML Model and capture model output in results
    test_data = assembler.transform(test_df)
    results = model.transform(test_data)

    # return the results after converting it to JSON
    return results.toJSON().first()

# Flask app when run gives local IP to access API. Since this is running on Cloud (Google Colab) and not on local notebook, local IP (127.0.0.1) will not be accessible.
# Below code will ask colab to give us publicly accessible URL

print(eval_js("google.colab.kernel.proxyPort(5000)"))
output.serve_kernel_port_as_window(5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)