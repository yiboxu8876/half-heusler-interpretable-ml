import os
import xgboost as xgb
import pandas as pd

best_params = {
    'colsample_bytree': 0.9839337327837884,
    'gamma': 0.011327298446158353,
    'learning_rate': 0.4894184238426754,
    'max_depth': 10,
    'min_child_weight': 8.679114875791846,
    'n_estimators': 220,
    'subsample': 0.5714948980507744,
    'random_state': 42,
    'objective': 'reg:squarederror'
}

train_data_path = 'data.xlsx'
train_data = pd.read_excel(train_data_path, sheet_name='Sheet1')

feature_columns = [
    'T', 'VEnumberX', 'VEnumberY', 'VEnumber',
'aveElgaY', 'aveElga',
'stanDElgaX', 'stanDElgaZ', 'stanDElga',
'aveAWX', 'aveAWY', 'aveAW',
'stanDAWX', 'aveARY', 'aveAR',
'stanDARX', 'stanDAR',
'aveFionEX', 'aveFionEY', 'aveFionEZ', 'aveFionE',
'stanDFionEX', 'stanDFionE'
]

label_column = 'property'

trainX = train_data[feature_columns]
trainY = train_data[label_column]

best_model = xgb.XGBRegressor(**best_params)
best_model.fit(trainX, trainY)

unknown_material_data_path = 'HP-Screening.xlsx'
unknown_material_features = pd.read_excel(unknown_material_data_path, sheet_name='Sheet1')

unknown_material_features_selected = unknown_material_features[feature_columns]

predictions = best_model.predict(unknown_material_features_selected)

print("Predictions for unknown materials:", predictions)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.csv')

unknown_material_features['predictions'] = predictions
unknown_material_features.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
