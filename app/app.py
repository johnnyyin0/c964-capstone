from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('../random_forest_model.joblib')
zip_code_avg_price_df = pd.read_csv('../zip_code_avg_price.csv')
zip_code_avg_price_df.columns = zip_code_avg_price_df.columns.str.strip()
valid_zip_codes = set(zip_code_avg_price_df['zip_code'].unique())

def predict_price(bed, bath, zip_code, house_size, acre_lot):
    if zip_code not in valid_zip_codes:
        return None

    user_input = pd.DataFrame({
        'bed': [bed],
        'bath': [bath],
        'zip_code': [zip_code],
        'house_size': [house_size],
        'acre_lot': [acre_lot]
    })
    
    predicted_price = model.predict(user_input)[0]
    return predicted_price

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        bed = int(request.form['bed'])
        bath = int(request.form['bath'])
        zip_code = int(request.form['zip_code'])
        house_size = int(request.form['house_size'])
        acre_lot = float(request.form['acre_lot'])
        
        predicted_price = predict_price(bed, bath, zip_code, house_size, acre_lot)
        
        if predicted_price is None:
            return render_template('index.html', prediction="Please enter a valid Zip Code.", bed=bed, bath=bath, zip_code=zip_code, house_size=house_size, acre_lot=acre_lot)
        
        return render_template('index.html', prediction=f"${predicted_price:,.2f}", bed=bed, bath=bath, zip_code=zip_code, house_size=house_size, acre_lot=acre_lot)
    
    return render_template('index.html', prediction=None, bed="", bath="", zip_code="", house_size="", acre_lot="")

if __name__ == '__main__':
    app.run(debug=True)
