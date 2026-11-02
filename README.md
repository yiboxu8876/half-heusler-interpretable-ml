# Interpretable machine learning for half-Heusler thermoelectrics

Processed inputs and analysis scripts for modelling the thermoelectric figure of merit (zT) and lattice thermal conductivity (kL) of half-Heusler compounds.

## Contents

| Directory | Contents |
| --- | --- |
| [GBR](GBR) | XGBoost training scripts, input data, feature rankings and prediction scripts |
| [GBR/initial_37_features](GBR/initial_37_features) | Initial feature-ranking inputs and scripts |
| [SHAP](SHAP) | Ensemble SHAP analysis, model parameter records and input data |
| [SISSO](SISSO) | Variable-selection script, SISSO configuration and training input |
| [SGD](SGD) | Subgroup-discovery configuration and XARF input for zT cutoff 1.0 |

The `zT_n` and `zT_p` directories contain the n-type and p-type analyses, respectively. Their inputs are from the corrected calculation runs.

## Input files

- `all_data.xlsx`: full input table for the corresponding model; the combined zT and kL tables each contain 499 records.
- `data.xlsx`: development data used by the corresponding calculation.
- `train.xlsx` and `test.xlsx`: training and internal test subsets read by the GBR training scripts.
- `validation.xlsx`: held-out evaluation data for the corresponding run.
- `features_importance_average_FF.dat`: feature-ranking results.
- SHAP directories `0`–`19`: parameter records for the ensemble models.

Spreadsheet columns contain record identifiers, metadata, descriptors and the target column `property`. Temperature is in kelvin. Consult the manuscript and Supplementary Information for descriptor definitions and experimental sources.

## Running the analyses

Run each script from its own working directory. Python dependencies include NumPy, pandas, scikit-learn, XGBoost, Hyperopt, Matplotlib, SHAP, Pillow, xlrd and openpyxl. The original scripts use legacy library interfaces; compatibility with current package versions has not been validated.

For GBR, `GBR-CV.py` reads the supplied training and test files. Prediction scripts may require additional model outputs or an `HP-Screening.xlsx` table supplied by the user.

For SHAP, replace `<SHAP_DIR>` with the analysis directory containing the numbered parameter folders and `<DATA_DIR>` with the directory containing `data.xlsx`, then run `SHAP_Analysis_Ensemble.py`.

SISSO and subgroup discovery require their respective external software. SISSO inputs are provided alongside `VarSelect_SISSO.py`; the SGD configuration reads `train.xarf` from its working directory.

The scripts and processed inputs are provided as research materials. The complete workflow has not been rerun from this repository. Further details of the compiled experimental dataset are available from the corresponding authors upon reasonable request.
