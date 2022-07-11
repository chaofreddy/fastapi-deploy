# 1. Library imports
import random
import joblib
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from xgboost import XGBRegressor


# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
async def root():
    return {"message": "Hello, welcome to the workout prediction API."}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted calories
@app.post('/predict')
async def predict_calories(heart_rate_mean: float, altitude_mean: float, ascend_m: float, descend_m: float,
                           distance_total_m: float, speed_mean: float, duration_s: float):
    model = joblib.load('calorie_regressor_simple_again.joblib')
    prediction = model.predict(
        [[heart_rate_mean, altitude_mean, ascend_m, descend_m, distance_total_m, speed_mean, duration_s]])
    return {'prediction': float(prediction)}


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)