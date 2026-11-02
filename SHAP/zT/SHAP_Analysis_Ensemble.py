import os
import re
import ast
import io
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from PIL import Image as PILImage

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
    'T', 'VEnumberX', 'VEnumberY', 'VEnumber',
    'aveElgaY', 'aveElga', 'stanDElgaX', 'stanDElgaZ', 'stanDElga',
    'aveAWX', 'aveAWY', 'aveAW', 'stanDAWX',
    'aveARY', 'aveAR', 'stanDARX', 'stanDAR',
    'aveFionEX', 'aveFionEY', 'aveFionEZ', 'aveFionE',
    'stanDFionEX', 'stanDFionE'
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

if model_count > 0:
    print(f"\nCompleted {model_count} models. Saving results...")

    avg_shap_values = shap_values_sum / model_count

    symbol_mapping = {
        'T': r'$\mathit{T}$',
        'VEnumberX': r'$(\mathrm{VEC})_{\mathit{X}}$',
        'VEnumberY': r'$(\mathrm{VEC})_{\mathit{Y}}$',
        'VEnumber': r'$\mathrm{VEC}$',
        'aveElgaY': r'$(\bar{\chi})_{\mathit{Y}}$',
        'aveElga': r'$\bar{\chi}$',
        'stanDElgaX': r'$(\mathrm{SD}_{\chi})_{\mathit{X}}$',
        'stanDElgaZ': r'$(\mathrm{SD}_{\chi})_{\mathit{Z}}$',
        'stanDElga': r'$\mathrm{SD}_{\chi}$',
        'aveAWX': r'$(\bar{\mathit{M}})_{\mathit{X}}$',
        'aveAWY': r'$(\bar{\mathit{M}})_{\mathit{Y}}$',
        'aveAW': r'$\bar{\mathit{M}}$',
        'stanDAWX': r'$(\mathrm{SD}_{\mathit{M}})_{\mathit{X}}$',
        'aveARY': r'$(\bar{\mathit{r}})_{\mathit{Y}}$',
        'aveAR': r'$\bar{\mathit{r}}$',
        'stanDARX': r'$(\mathrm{SD}_{\mathit{r}})_{\mathit{X}}$',
        'stanDAR': r'$\mathrm{SD}_{\mathit{r}}$',
        'aveFionEX': r'$(\bar{\mathit{E}}_{\mathit{i}})_{\mathit{X}}$',
        'aveFionEY': r'$(\bar{\mathit{E}}_{\mathit{i}})_{\mathit{Y}}$',
        'aveFionEZ': r'$(\bar{\mathit{E}}_{\mathit{i}})_{\mathit{Z}}$',
        'aveFionE': r'$\bar{\mathit{E}}_{\mathit{i}}$',
        'stanDFionEX': r'$(\mathrm{SD}_{\mathit{E}_{\mathit{i}}})_{\mathit{X}}$',
        'stanDFionE': r'$\mathrm{SD}_{\mathit{E}_{\mathit{i}}}$'
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
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'

    BASE_FONT_SIZE = 12
    AXES_LABEL_SIZE = 14
    TITLE_SIZE = 16
    TICK_LABEL_SIZE_X = 11
    TARGET_Y_LABEL_SIZE = 16

    plt.rcParams['font.size'] = BASE_FONT_SIZE
    plt.rcParams['axes.labelsize'] = AXES_LABEL_SIZE
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE_X

    DPI_VAL = 600

    def figure_to_pil(dpi):
        """Render figure to an image."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.05)
        buf.seek(0)
        img = PILImage.open(buf).convert('RGB')

        img.load()
        buf.close()
        return img

    def pad_to_height(img, target_h, bg=(255, 255, 255)):
        """Pad the image to target height."""
        if img.height == target_h:
            return img
        pad = target_h - img.height
        pad_top = pad // 2
        new_img = PILImage.new('RGB', (img.width, target_h), bg)
        new_img.paste(img, (0, pad_top))
        return new_img

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

    ax.set_xlabel("SHAP value (impact on model output)",
                  fontsize=AXES_LABEL_SIZE, fontweight='normal')
    plt.title(r"Ensemble SHAP Summary Plot (Top 10, Excl. $\mathit{T}$)",
              fontsize=TITLE_SIZE, fontweight='normal')

    img_summary = figure_to_pil(DPI_VAL)
    plt.close()
    print(f"Summary: {img_summary.size} px")

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

    for child in plt.gcf().axes:
        if child.get_xlabel():
            child.set_xlabel(child.get_xlabel(),
                             fontsize=AXES_LABEL_SIZE, fontweight='normal')
        if child.get_ylabel():
            child.set_ylabel(child.get_ylabel(),
                             fontsize=AXES_LABEL_SIZE, fontweight='normal')

    plt.title(r"Ensemble SHAP Heatmap (Excl. $\mathit{T}$)",
              fontsize=TITLE_SIZE, fontweight='normal')

    img_heatmap = figure_to_pil(DPI_VAL)
    plt.close()
    print(f"Heatmap: {img_heatmap.size} px")

    target_h = max(img_summary.height, img_heatmap.height)
    img_summary = pad_to_height(img_summary, target_h)
    img_heatmap = pad_to_height(img_heatmap, target_h)

    GAP_PX = 100
    combined_w = img_summary.width + GAP_PX + img_heatmap.width
    combined = PILImage.new('RGB', (combined_w, target_h), (255, 255, 255))
    combined.paste(img_summary, (0, 0))
    combined.paste(img_heatmap, (img_summary.width + GAP_PX, 0))

    combined_path = os.path.join(root_dir, 'Fig_SHAP_Summary_Heatmap_Combined.tif')
    combined.save(combined_path, format='TIFF',
                  compression='tiff_lzw', dpi=(DPI_VAL, DPI_VAL))
    print(f"Saved ({DPI_VAL} dpi TIFF): {combined_path}")
    print(f"Size: {combined.size} px"
          f" = {combined.size[0]/DPI_VAL:.2f} x {combined.size[1]/DPI_VAL:.2f} inch")

else:
    print("\nNo valid model parameters. Check input files.")
