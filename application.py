from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get the data from the form
        data = CustomData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch= request.form.get('lunch'),
            test_preparation_course= request.form.get('test_preparation_course'),
            reading_score= int(request.form.get('reading_score')),
            writing_score= int(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df,"-----------------------------------")
        
        pred_pipeline = PredictPipeline()
        result = pred_pipeline.predict(pred_df)
        return render_template('home.html', results=result[0])
        
if __name__ == '__main__':
    app.run(host="0.0.0.0")