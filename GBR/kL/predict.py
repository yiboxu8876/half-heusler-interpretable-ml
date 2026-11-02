import os
import xgboost as xgb
import pandas as pd

best_params = {
    'colsample_bytree': 0.6248374624871103,
    'gamma': 0.9525919188068191,
    'learning_rate': 0.38019598050921644,
    'max_depth': 4,
    'min_child_weight': 4.321263027634352,
    'n_estimators': 278,
    'subsample': 0.9012745112532361,
    'random_state': 42,
    'objective': 'reg:squarederror'
}

train_data_path = 'data.xlsx'
train_data = pd.read_excel(train_data_path, sheet_name='Sheet1')

feature_columns = [
'T', 'VEnumberX', 'VEnumberZ', 'VEnumber',
'stanDElgaX',
'aveAWY', 'aveAWZ', 'aveAW',
'stanDAWX', 'stanDAWZ', 'stanDAW',
'aveAR',
'stanDARX', 'stanDAR',
'aveFionEX', 'aveFionE'
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
