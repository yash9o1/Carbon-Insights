from django.apps import AppConfig

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        # Load your model and other resources here
        import os
        from django.conf import settings
        import joblib
        from tensorflow.keras.models import load_model
        import pandas as pd

        # Specify the path to your model files
        base_dir = settings.BASE_DIR
        model_path = os.path.join(base_dir, 'predictor', 'electricity_prediction_model.keras')
        scaler_path = os.path.join(base_dir, 'predictor', 'scaler.joblib')
        state_dict_path = os.path.join(base_dir, 'predictor', 'state_dict.joblib')
        historical_data_path = os.path.join(base_dir, 'predictor', 'historical_electricity_data.csv')

        # Load the model and other resources
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.state_dict = joblib.load(state_dict_path)
        self.df = pd.read_csv(historical_data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])