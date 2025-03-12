from django.http import JsonResponse
from django.apps import apps
import pandas as pd
import numpy as np
from datetime import datetime

SEQ_LENGTH = 12
TARGET_MONTHS = 5

def prepare_input_data(state, date, df, state_dict):
    state_num = state_dict[state]
    date = pd.to_datetime(date)
    
    # Get the last 12 months of data for the state
    state_data = df[(df['State'] == state) & (df['Date'] <= date)].sort_values('Date').tail(SEQ_LENGTH)
    
    if len(state_data) < SEQ_LENGTH:
        return None
    
    # Prepare input data
    date_input = state_data[['Year', 'Month']].values
    date_input = np.expand_dims(date_input, axis=0)  # Add batch dimension
    state_input = np.full((1, SEQ_LENGTH), state_num)
    
    return date_input, state_input

def predict(request, state, date):
    predictor_config = apps.get_app_config('predictor')
    
    input_data = prepare_input_data(state, date, predictor_config.df, predictor_config.state_dict)
    
    if input_data is None:
        return JsonResponse({'error': f'Not enough historical data for {state} as of {date}'}, status=400)
    
    date_input, state_input = input_data
    
    # Make prediction
    prediction = predictor_config.model.predict([date_input, state_input])
    
    # Reshape and inverse transform the prediction
    prediction = prediction.reshape(TARGET_MONTHS, -1)
    prediction = predictor_config.scaler.inverse_transform(prediction)
    
    # Generate dates for the predicted months
    pred_dates = pd.date_range(start=pd.to_datetime(date) + pd.DateOffset(months=1), periods=TARGET_MONTHS, freq='M')
    
    # Create DataFrame with predictions
    columns = ['Peak Demand (MW)', 'Peak Production (MW)', 'Solar Production (MW)', 'Coal Production (MW)', 'Total Production (MW)']
    df_pred = pd.DataFrame(prediction, columns=columns, index=pred_dates)
    df_pred.index = df_pred.index.strftime('%Y-%m-%d')
    
    return JsonResponse(df_pred.to_dict(orient='index'))
