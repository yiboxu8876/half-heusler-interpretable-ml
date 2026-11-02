from pathlib import Path

import pandas as pd
import xgboost as xgb

RUN_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = RUN_DIR / "data.xlsx"
VALIDATION_DATA_PATH = RUN_DIR / "validation.xlsx"
OUTPUT_PATH = RUN_DIR / "predictions.csv"

# BEGIN BEST_PARAMS
best_params = {
    "colsample_bytree": 0.9097593801713455,
    "gamma": 0.010606128287296734,
    "learning_rate": 0.1533130358471935,
    "max_depth": 7,
    "min_child_weight": 2.221030070549163,
    "n_estimators": 192,
    "subsample": 0.5642897485544515,
    "random_state": 42,
    "objective": "reg:squarederror",
}
# END BEST_PARAMS

FEATURE_COLUMNS = [
    "T",
    "VEnumberX",
    "VEnumberY",
    "VEnumber",
    "aveElgaY",
    "aveElga",
    "stanDElgaX",
    "stanDElgaZ",
    "stanDElga",
    "aveAWX",
    "aveAWY",
    "aveAW",
    "stanDAWX",
    "aveARY",
    "aveAR",
    "stanDARX",
    "stanDAR",
    "aveFionEX",
    "aveFionEY",
    "aveFionEZ",
    "aveFionE",
    "stanDFionEX",
    "stanDFionE",
]
LABEL_COLUMN = "property"

train_data = pd.read_excel(TRAIN_DATA_PATH, sheet_name="Sheet1")
validation_data = pd.read_excel(VALIDATION_DATA_PATH, sheet_name="Sheet1")

required_train_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
missing_train = [column for column in required_train_columns if column not in train_data.columns]
missing_validation = [column for column in FEATURE_COLUMNS if column not in validation_data.columns]
if missing_train:
    raise KeyError(f"Missing training columns: {missing_train}")
if missing_validation:
    raise KeyError(f"Missing validation columns: {missing_validation}")

best_model = xgb.XGBRegressor(**best_params)
best_model.fit(train_data[FEATURE_COLUMNS], train_data[LABEL_COLUMN])
validation_data["predictions"] = best_model.predict(validation_data[FEATURE_COLUMNS])
validation_data.to_csv(OUTPUT_PATH, index=False)

print(f"Predicted {len(validation_data)} validation rows -> {OUTPUT_PATH}")
