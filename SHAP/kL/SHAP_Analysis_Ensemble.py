import os
import re
import ast
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

root_dir = r'<SHAP_DIR>'
train_data_path = r'<DATA_DIR>/data.xlsx'

param_pattern = re.compile(r"The_best_parameter\s*[:：]\s*({.*?})", re.DOTALL)

def read_file_safe(file_path):
    """Read with fallback encodings."""
    for enc in ['utf-8', 'gbk', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=enc) as f: return f.read()
        except: continue
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f: return f.read()

def convert_numpy_types(obj):
    """Convert NumPy literals for parsing."""
    if isinstance(obj, (np.integer)): return int(obj)
    elif isinstance(obj, (np.floating)): return float(obj)
    elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
    return obj

print(f"Reading: {train_data_path}")
train_data = pd.read_excel(train_data_path, sheet_name='Sheet1')

feature_columns = [
'T', 'VEnumberX', 'VEnumberZ', 'VEnumber', 'stanDElgaX', 'aveAWY', 'aveAWZ', 'aveAW', 'stanDAWX', 'stanDAWZ', 'stanDAW', 'aveAR', 'stanDARX', 'stanDAR', 'aveFionEX', 'aveFionE'
]
label_column = 'property'

trainX = train_data[feature_columns]
trainY = train_data[label_column]

shap_values_sum = np.zeros(trainX.shape)
model_count = 0

print("Computing SHAP values...")

for subdir, dirs, files in os.walk(root_dir):
    dat_file = next((os.path.join(subdir, f) for f in files if f.startswith('hyper_parameter_selection_') and f.endswith('_FF.dat')), None)

    if dat_file:
        try:

            content = read_file_safe(dat_file)
            match = param_pattern.search(content)
            if not match: continue

            tree = ast.parse(match.group(1), mode='eval')
            best_params = eval(compile(tree, filename='<string>', mode='eval'))
            best_params = convert_numpy_types(best_params)

            if 'max_depth' in best_params: best_params['max_depth'] = int(best_params['max_depth'])
            if 'n_estimators' in best_params: best_params['n_estimators'] = int(best_params['n_estimators'])
            best_params['random_state'] = 42
            best_params['objective'] = 'reg:squarederror'

            model = xgb.XGBRegressor(**best_params)
            model.fit(trainX, trainY)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(trainX)

            shap_values_sum += shap_values
            model_count += 1
            print(f"Done: {os.path.basename(subdir)}")

        except Exception as e:
            print(f"Skipped {subdir}: {e}")

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

if model_count > 0:
    print(f"\nCompleted {model_count} models. Saving results...")

    avg_shap_values = shap_values_sum / model_count

    symbol_mapping = {
        'T': r'$\mathit{T}$',
        'VEnumberX': r'$(\mathit{VEC})_{\mathit{X}}$',
        'VEnumberZ': r'$(\mathit{VEC})_{\mathit{Z}}$',
        'VEnumber': r'$\mathit{VEC}$',
        'stanDElgaX': r'$(\sigma_{\chi})_{\mathit{X}}$',
        'aveAWY': r'$(\bar{\mathit{M}})_{\mathit{Y}}$',
        'aveAWZ': r'$(\bar{\mathit{M}})_{\mathit{Z}}$',
        'aveAW': r'$\bar{\mathit{M}}$',
        'stanDAWX': r'$(\sigma_{\mathit{M}})_{\mathit{X}}$',
        'stanDAWZ': r'$(\sigma_{\mathit{M}})_{\mathit{Z}}$',
        'stanDAW': r'$\sigma_{\mathit{M}}$',
        'aveAR': r'$\bar{\mathit{r}}$',
        'stanDARX': r'$(\sigma_{\mathit{r}})_{\mathit{X}}$',
        'stanDAR': r'$\sigma_{\mathit{r}}$',
        'aveFionEX': r'$(\bar{\mathit{E}}_{\mathit{i}})_{\mathit{X}}$',
        'aveFionE': r'$\bar{\mathit{E}}_{\mathit{i}}$'
    }

    display_feature_columns = [symbol_mapping.get(f, f) for f in feature_columns]

    if 'T' in feature_columns:
        t_idx = feature_columns.index('T')
        feature_columns_no_T = [f for f in feature_columns if f != 'T']
        display_features_no_T = [f for f in display_feature_columns if f != r'$\mathit{T}$']
        trainX_no_T = trainX.drop(columns=['T'])
        avg_shap_values_no_T = np.delete(avg_shap_values, t_idx, axis=1)
    else:
        feature_columns_no_T = feature_columns
        display_features_no_T = display_feature_columns
        trainX_no_T = trainX
        avg_shap_values_no_T = avg_shap_values

    shap_output_path = os.path.join(root_dir, 'Ensemble_SHAP_Matrix.xlsx')
    pd.DataFrame(avg_shap_values, columns=feature_columns).to_excel(shap_output_path, index=False)
    print(f"Saved: {shap_output_path}")

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Calibri:bold'
    plt.rcParams['mathtext.it'] = 'Calibri:bold:italic'

    BASE_FONT_SIZE = 12
    AXES_LABEL_SIZE = 14
    TITLE_SIZE = 16
    TICK_LABEL_SIZE_X = 11
    TARGET_Y_LABEL_SIZE = 16

    plt.rcParams['font.size'] = BASE_FONT_SIZE
    plt.rcParams['axes.labelsize'] = AXES_LABEL_SIZE
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE_X

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        avg_shap_values_no_T,
        trainX_no_T,
        feature_names=display_features_no_T,
        max_display=10,
        show=False
    )
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=TARGET_Y_LABEL_SIZE)
    plt.title(r"Ensemble SHAP Summary Plot (Top 10, Excl. $\mathit{T}$)", fontsize=TITLE_SIZE)
    summary_img_path = os.path.join(root_dir, 'Ensemble_SHAP_Summary_Top10_NoT.png')
    plt.savefig(summary_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_img_path}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        avg_shap_values_no_T,
        trainX_no_T,
        feature_names=display_features_no_T,
        plot_type="bar",
        max_display=10,
        show=False
    )
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=TARGET_Y_LABEL_SIZE)
    plt.title(r"Ensemble SHAP Bar Plot (Top 10, Excl. $\mathit{T}$)", fontsize=TITLE_SIZE)
    bar_img_path = os.path.join(root_dir, 'Ensemble_SHAP_Bar_Plot_Top10_NoT.png')
    plt.savefig(bar_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_img_path}")

    ensemble_explanation_no_T = shap.Explanation(
        values=avg_shap_values_no_T,
        data=trainX_no_T.values,
        feature_names=display_features_no_T
    )
    plt.figure(figsize=(12, 8))
    shap.plots.heatmap(
        ensemble_explanation_no_T,
        max_display=11,
        show=False
    )

    ax = plt.gca()
    new_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        if text.startswith("Sum of"):
            new_labels.append("Sum of rest")
        else:
            new_labels.append(text)
    ax.set_yticklabels(new_labels)
    ax.tick_params(axis='y', labelsize=TARGET_Y_LABEL_SIZE)

    plt.title(r"Ensemble SHAP Heatmap (Excl. $\mathit{T}$)", fontsize=TITLE_SIZE)
    heatmap_img_path = os.path.join(root_dir, 'Ensemble_SHAP_Heatmap_NoT.png')
    plt.savefig(heatmap_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {heatmap_img_path}")

else:
    print("\nNo valid model parameters. Check input files.")
