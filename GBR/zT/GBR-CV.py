import os

import xgboost
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
import xgboost as xgb
import warnings
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True

# --- Build Dataset

import pandas as pd
import numpy as np

def data_load(filename):
    data = pd.read_excel(filename)
    result = data.values.tolist()
    nrows, ncols = data.shape

    ID = [int(x[0]) for x in result]
    formula = [x[1] for x in result]
    prototype = [x[2] for x in result]
    features = np.array([x[3:ncols-1] for x in result])
    label = np.array([x[-1] for x in result])
    features_name = data.columns[3:ncols-1].tolist()

    return ID, formula, prototype, features, label, features_name

def data_load_train(filename):
    return data_load(filename)

def data_load_test(filename):
    return data_load(filename)

# Handle missing values in the dataset using mean imputation
def imputer(dataArr):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(dataArr)
    dataArr_full = imp.transform(dataArr)
    return dataArr_full

# Normalize/standardize the features in X and the target variable Y
def normData(dataX, dataY):
    scaler = StandardScaler()
    X = scaler.fit_transform(dataX)
    Y = dataY
    return X, Y

# Load train and test datasets
train_filename = 'train.xlsx'
test_filename = 'test.xlsx'

ID_train, formula_train, prototype_train, features_train, label_train, features_name_train = data_load_train(train_filename)
ID_test, formula_test, prototype_test, features_test, label_test, features_name_test = data_load_test(test_filename)

# Apply preprocessing
features_train = imputer(features_train)
features_test = imputer(features_test)

# Normalize features and labels for both train and test sets
X_train, Y_train = normData(features_train, label_train)
X_test, Y_test = normData(features_test, label_test)

# Check if X_train and X_test are correctly loaded
print("Shape of X_train:", X_train.shape)  # Debugging: Check shape of X_train
print("Shape of X_test:", X_test.shape)  # Debugging: Check shape of X_test

# --- Model Evaluation

def model_evaluation_regression(true_labels, predicted_labels):
    # Calculate R2 score
    R2 = r2_score(true_labels, predicted_labels)
    # Calculate mean squared error
    MSE = mean_squared_error(true_labels, predicted_labels)
    # Calculate mean absolute error
    MAE = mean_absolute_error(true_labels, predicted_labels)
    # Calculate explained variance
    EV = explained_variance_score(true_labels, predicted_labels)

    return R2, MSE, MAE, EV

def hyper_parameters_plot(parameters, trials, Loop_Step, screen_step):
    # Create subplots for parameter visualization
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), dpi=330)
    cmap = plt.cm.jet

    for i, param in enumerate(parameters):
        xs = np.array([t['misc']['vals'][param] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))

        # Scatter plot of parameter values and losses
        axes[int(i / 3), int(i % 3)].scatter(xs, ys, s=50, linewidth=0.01, alpha=0.5,
                                             color=cmap(float(i) / len(parameters)))
        axes[int(i / 3), int(i % 3)].set_title(param, fontsize=20, fontweight="bold", fontname='Arial')

        # Set tick parameters and axis limits
        axes[int(i / 3), int(i % 3)].tick_params(labelsize=18, direction='out', width=2, length=6)
        axes[int(i / 3), int(i % 3)].set_ylim((0.5, 1))
        labels = axes[int(i / 3), int(i % 3)].get_xticklabels() + axes[int(i / 3), int(i % 3)].get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontweight('bold') for label in labels]
        axes[int(i / 3), int(i % 3)].spines['top'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['bottom'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['right'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['left'].set_linewidth(2.5)

    # Adjust subplot layout and save the figure
    plt.tight_layout()
    plt.savefig(result_path + 'hyper_parameters_trials_' + str(Loop_Step) + "_" + str(screen_step) + '.png',
                format='png')
    plt.close()

def plot_regression_results(train_labels, test_labels, pred_train_labels, std_pred_train_labels, pred_test_labels, std_pred_test_labels, Loop_Step, screen_step):
    # Calculate R2 score and mean absolute error for training set
    R2_train = r2_score(train_labels, pred_train_labels)
    MAE_train = mean_absolute_error(train_labels, pred_train_labels)
    MSE_train = mean_squared_error(train_labels, pred_train_labels)
    RMSE_train = np.sqrt(MSE_train)

    # Calculate R2 score and mean absolute error for test set
    R2_test = r2_score(test_labels, pred_test_labels)
    MAE_test = mean_absolute_error(test_labels, pred_test_labels)
    MSE_test = mean_squared_error(test_labels, pred_test_labels)
    RMSE_test = np.sqrt(MSE_test)

    tolerance = 0.1
    plt.figure(figsize=(10, 10), dpi=330)

    # Plot the ideal line
    plt.plot([test_labels.min() - tolerance, test_labels.max() + tolerance],
             [test_labels.min() - tolerance, test_labels.max() + tolerance],
             '--r', linewidth=3)

    # Plot the training set results with error bars
    plt.errorbar(train_labels, pred_train_labels, yerr=std_pred_train_labels, alpha=0.75, fmt='o', ms=13,
                 mfc=(0 / 255, 98 / 255, 132 / 255),
                 ecolor=(55 / 255, 60 / 255, 56 / 255), elinewidth=3, capsize=5, capthick=3,
                 label=r'train $R^2=$ %.2f  $RMSE=$ %.2f  $MAE=$ %.2f' % (R2_train, RMSE_train, MAE_train))

    # Plot the test set results with error bars
    plt.errorbar(test_labels, pred_test_labels, yerr=std_pred_test_labels, alpha=0.75, fmt='o', ms=13,
                 mfc=(232 / 255, 48 / 255, 21 / 255),
                 ecolor=(55 / 255, 60 / 255, 56 / 255), elinewidth=3, capsize=5, capthick=3,
                 label=r'test $R^2=$ %.2f $RMSE=$ %.2f $MAE=$ %.2f' % (R2_test, RMSE_test, MAE_test))

    # Set x and y limits
    plt.xlim([test_labels.min() - tolerance, test_labels.max() + tolerance])
    plt.ylim([test_labels.min() - tolerance, test_labels.max() + tolerance])

    # Set x and y labels, legend, and tick parameters
    plt.xlabel('DFT Calculated Energy (eV)', fontsize=26, fontweight="bold", fontname='Arial')
    plt.ylabel('ML Predicted Energy (eV)', fontsize=26, fontweight="bold", fontname='Arial')
    plt.legend(loc='upper left')
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    bwith = 3
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.setp(ltext, fontsize=24, fontweight='bold', fontname='Arial')
    plt.tick_params(direction='out', width=2, length=6)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontweight('bold') for label in labels]
    [label.set_fontsize(24) for label in labels]
    plt.tight_layout()
    plt.title('DFT Calculated Values vs ML Predicted Values', fontsize=26, fontweight="bold", fontname='Arial')

    # Save the figure and close the plot
    plt.savefig(result_path + "FM_Regression_" + str(Loop_Step) + "_" + str(screen_step) + ".png", format="png")
    plt.close()

# --- Parameter Selection

# --- Parameter Selection

def GBRegression(X_train, X_test, Y_train, Y_test, max_evals, Loop_Step, screen_step):

    GBR_model_score_filename = open(result_path + "hyper_parameter_selection_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", "a")
    predicted_Y_test_file = open(result_path + "predicted_Y_test_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", "a")
    predicted_Y_train_file = open(result_path + "predicted_Y_train_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", "a")

    # Define the parameter space for hyperopt
    parameter_space_gbr = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "max_depth": hp.quniform("max_depth", 1, 12, 1),
        "n_estimators": hp.quniform("n_estimators", 10, 300, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.6),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "min_child_weight": hp.uniform("min_child_weight", 0.5, 10),
        "gamma": hp.uniform("gamma", 0.01, 1)
    }

    count = 0  # Counter for parameter combinations

    def function(argsDict):
        # Extract parameter values from argsDict
        colsample_bytree = argsDict["colsample_bytree"]
        max_depth = argsDict["max_depth"]
        n_estimators = int(argsDict['n_estimators'])
        learning_rate = argsDict["learning_rate"]
        subsample = argsDict["subsample"]
        min_child_weight = argsDict["min_child_weight"]
        gamma = argsDict["gamma"]

        # Create and train the XGBoost regression model
        clf = xgb.XGBRegressor(
            nthread=4,
            colsample_bytree=colsample_bytree,
            max_depth=int(max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=42,
            objective="reg:squarederror"
        )
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)

        nonlocal count
        count += 1

        # Evaluate the model's performance
        R2, MSE, MAE, EV = model_evaluation_regression(Y_test, prediction)
        print(f"No.{count}, R2: {R2:.6f}, MSE: {MSE:.6f}, MAE: {MAE:.6f}, EV: {EV:.6f}", argsDict,
              file=GBR_model_score_filename)
        print(
            f"Screen_step: {screen_step}, Loop: {Loop_Step}, No. {count}, R2: {R2:.6f}, MSE: {MSE:.6f}, MAE: {MAE:.6f}, EV: {EV:.6f}")

        # Return the negative R2 value for hyperopt to maximize
        return -R2

    trials = Trials()
    best = fmin(function, parameter_space_gbr, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    parameters = ['colsample_bytree', 'max_depth', 'n_estimators', 'learning_rate', 'gamma', 'min_child_weight']
    hyper_parameters_plot(parameters, trials, Loop_Step, screen_step)
    print('best', best)

    # Retrieve the best hyperparameters
    colsample_bytree = best["colsample_bytree"]
    max_depth = best["max_depth"]
    n_estimators = int(best['n_estimators'])
    learning_rate = best["learning_rate"]
    subsample = best["subsample"]
    min_child_weight = best["min_child_weight"]
    gamma = best["gamma"]

    print("The_best_parameter:", best, file=GBR_model_score_filename)

    # Train the best model using the best hyperparameters
    best_model = xgb.XGBRegressor(
        nthread=4,
        colsample_bytree=colsample_bytree,
        max_depth=int(max_depth),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=42,
        objective="reg:squarederror"
    )
    best_model.fit(X_train, Y_train)
    best_model_pred_Y_test = best_model.predict(X_test)
    best_model_pred_Y_train = best_model.predict(X_train)

    # Model evaluation for the best model
    R2_test, MSE_test, MAE_test, EV_test = model_evaluation_regression(Y_test, best_model_pred_Y_test)
    R2_train, MSE_train, MAE_train, EV_train = model_evaluation_regression(Y_train, best_model_pred_Y_train)

    print("The_best_model_train_score:", R2_train, MSE_train, MAE_train, EV_train, file=GBR_model_score_filename)
    print("The_best_model_test_score:", R2_test, MSE_test, MAE_test, EV_test, file=GBR_model_score_filename)

    # Save the best model and its visualization
    best_model.save_model(result_path + "best_model_" + str(Loop_Step) + "_" + str(screen_step) + ".json")
    digraph = xgboost.to_graphviz(best_model)
    digraph_name = result_path + "plot_best_tree_" + str(Loop_Step) + "_" + str(screen_step) + ".gv"
    digraph.save(filename=digraph_name)

    # Write predicted values to files
    for y in best_model_pred_Y_test:
        print(y, file=predicted_Y_test_file)
    for y in best_model_pred_Y_train:
        print(y, file=predicted_Y_train_file)

    # Close the files
    GBR_model_score_filename.close()
    predicted_Y_test_file.close()
    predicted_Y_train_file.close()

    return best_model, best_model_pred_Y_train, best_model_pred_Y_test, \
        R2_train, MSE_train, MAE_train, EV_train, \
        R2_test, MSE_test, MAE_test, EV_test

# --- Prediction

# Function to create directory if it doesn't exist
def path_mkdir(path, result_subfile, screen_step):
    feature_engineering_result_file = result_file + str(result_subfile) + str(screen_step)
    if not os.path.exists(result_file):
        os.mkdir(path + result_file)
    if not os.path.exists(feature_engineering_result_file):
        os.mkdir(path + feature_engineering_result_file)
    result_path = path + feature_engineering_result_file + "/"
    return result_path

def train_test(X_train, X_test, Y_train, Y_test, features_name, screen_step):
    train_data_file = open(result_path + "train_data_" + str(screen_step) + ".dat", 'a')
    test_data_file = open(result_path + "test_data_" + str(screen_step) + ".dat", 'a')

    for x, y in zip(X_train, Y_train):
        print(f"Features: {x}, Label: {y}", file=train_data_file)

    for x, y in zip(X_test, Y_test):
        print(f"Features: {x}, Label: {y}", file=test_data_file)

    train_data_file.close()
    test_data_file.close()

    pred_testArr = []  # List to store predicted test values
    mean_pred_testArr = []  # List to store mean of predicted test values
    std_pred_testArr = []  # List to store standard deviation of predicted test values
    pred_trainArr = []  # List to store predicted train values
    mean_pred_trainArr = []  # List to store mean of predicted train values
    std_pred_trainArr = []  # List to store standard deviation of predicted train values
    features_importanceArr = []  # List to store feature importances
    mean_features_importanceArr = []  # List to store mean of feature importances
    std_features_importanceArr = []  # List to store standard deviation of feature importances

    for n in range(LoopStepMin, LoopStepMax):
        # Perform GBRegression and store the results
        best_model, best_model_pred_Y_train, best_model_pred_Y_test, R2_train, MSE_train, MAE_train, EV_train, R2_test, MSE_test, MAE_test, EV_test = \
            GBRegression(X_train, X_test, Y_train, Y_test, HyperParameter_Step, n, screen_step)
        pred_trainArr.append(best_model_pred_Y_train)
        pred_testArr.append(best_model_pred_Y_test)

        features_importance = best_model.feature_importances_
        features_importanceArr.append(features_importance)

    # Calculate mean and standard deviation of predicted test values
    for i in range(np.shape(pred_testArr)[1]):
        sample = np.array(pred_testArr)[:, i]
        testsample_prediction_mean = np.mean(sample)
        testsample_prediction_std = np.std(sample, ddof=0)
        mean_pred_testArr.append(testsample_prediction_mean)
        std_pred_testArr.append(testsample_prediction_std)

    # Calculate mean and standard deviation of predicted train values
    for j in range(np.shape(pred_trainArr)[1]):
        sample = np.array(pred_trainArr)[:, j]
        trainsample_prediction_mean = np.mean(sample)
        trainsample_prediction_std = np.std(sample, ddof=0)
        mean_pred_trainArr.append(trainsample_prediction_mean)
        std_pred_trainArr.append(trainsample_prediction_std)

    # Calculate mean and standard deviation of feature importances
    for k in range(np.shape(features_importanceArr)[1]):
        sample = np.array(features_importanceArr)[:, k]
        features_importance_mean = np.mean(sample)
        features_importance_std = np.std(sample, ddof=0)
        mean_features_importanceArr.append(features_importance_mean)
        std_features_importanceArr.append(features_importance_std)

    # Plot regression results
    plot_regression_results(Y_train, Y_test, np.array(mean_pred_trainArr), np.array(std_pred_trainArr),
                            np.array(mean_pred_testArr), \
                            np.array(std_pred_testArr), 'average', screen_step)

    # Open files for writing predicted train, test, and feature importance values
    train_font = open(result_path + "predicted_train_average_" + str(screen_step) + ".dat", "a")
    test_font = open(result_path + "predicted_test_average_" + str(screen_step) + ".dat", "a")
    features_importance_font = open(result_path + "features_importance_average_" + str(screen_step) + ".dat", "a")

    # Write mean and standard deviation of predicted train values to file
    for i, j in zip(mean_pred_trainArr, std_pred_trainArr):
        print(i, j, file=train_font)

    # Write mean and standard deviation of predicted test values to file
    for i, j in zip(mean_pred_testArr, std_pred_testArr):
        print(i, j, file=test_font)

    # Write feature names, mean, and standard deviation of feature importances to file
    for i, j, k in zip(features_name, mean_features_importanceArr, std_features_importanceArr):
        print(i, j, k, file=features_importance_font)

    # Close the files
    train_font.close()
    test_font.close()
    features_importance_font.close()

    # Return the calculated values
    return mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, mean_features_importanceArr, std_features_importanceArr

def predict(X_train, Y_train, X_test, Y_test, predictX, screen_step):
    print("Predicting......")
    predict_font = open(result_path + "predicted_predict_average_" + str(screen_step) + ".dat", "a")
    train_set = pd.DataFrame(X_train, columns=features_name)
    dtrain = xgboost.DMatrix(train_set)
    test_set = pd.DataFrame(X_test, columns=features_name)
    dtest = xgboost.DMatrix(test_set)
    predict_set = pd.DataFrame(predictX, columns=features_name)
    dpredict = xgboost.DMatrix(predict_set)
    pred_predict_Arr = []  # List to store predicted values for predictX
    mean_pred_predict_Arr = []  # List to store mean of predicted values for predictX
    std_pred_predict_Arr = []  # List to store standard deviation of predicted values for predictX

    for i in range(LoopStepMin, LoopStepMax):
        model_file = result_path + "best_model_" + str(i) + "_FF.model"
        best_model = xgboost.Booster(model_file=model_file)
        best_model_pred_Y_test = best_model.predict(dtest)
        best_model_pred_Y_train = best_model.predict(dtrain)
        best_model_pred_predictY = best_model.predict(dpredict)
        R2_test, MSE_test, MAE_test, EV_test = model_evaluation_regression(Y_test, best_model_pred_Y_test)
        R2_train, MSE_train, MAE_train, EV_train = model_evaluation_regression(Y_train, best_model_pred_Y_train)
        print("R2_train: %f, MSE_train: %f, MAE_train: %f, EV_train: %f" % (R2_train, MSE_train, MAE_train, EV_train))
        print("R2_test: %f, MSE_test: %f, MAE_test: %f, EV_test: %f" % (R2_test, MSE_test, MAE_test, EV_test))
        pred_predict_Arr.append(best_model_pred_predictY)

    # Calculate mean and standard deviation of predicted values for predictX
    for i in range(np.shape(pred_predict_Arr)[1]):
        sample = np.array(pred_predict_Arr)[:, i]
        predictsample_prediction_mean = np.mean(sample)
        predictsample_prediction_std = np.std(sample, ddof=0)
        mean_pred_predict_Arr.append(predictsample_prediction_mean)
        std_pred_predict_Arr.append(predictsample_prediction_std)

    # Write mean and standard deviation of predicted values for predictX to file
    for i, j in zip(mean_pred_predict_Arr, std_pred_predict_Arr):
        print(i, j, file=predict_font)

    # Return the calculated values
    return

# --- feature engineering

def feature_importance(features_name, features, screen_step):
    # Open files to write normalized and original dataset
    normdata_filename = open(result_path + "norm_dataset_" + str(screen_step) + ".dat", "a")
    data_filename = open(result_path + "dataset_" + str(screen_step) + ".dat", "a")

    if n_fixed_features != 0:
        # Extract vector features
        vector_features = features[:, n_fixed_features: np.shape(features)[1]]
        vector_features_name = list(features_name[n_fixed_features: np.shape(features)[1]])

        # Concatenate fixed features with vector features
        new_features = np.concatenate((fixed_features, vector_features), axis=1)

        # Normalize features and labels
        norm_features, norm_label = normData(new_features, label)

        # Concatenate fixed feature names with vector feature names
        new_features_name = np.concatenate((initial_fixed_features_name, vector_features_name), axis=0)

        # Write normalized dataset to file
        print("ID, formula, prototype", [name for name in new_features_name], "label", file=normdata_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, norm_features, label):
            print(i, j, k, [item for item in x], y, file=normdata_filename)

        # Write original dataset to file
        print("ID, formula, prototype", [name for name in new_features_name], "label", file=data_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, features, label):
            print(i, j, k, [item for item in x], y, file=data_filename)
    else:
        # Use all features as vector features
        vector_features = features
        vector_features_name = list(features_name)

        # Normalize vector features and labels
        norm_vector_features, norm_label = normData(vector_features, label)
        norm_features = norm_vector_features

        # Write normalized dataset to file
        print("ID, formula, prototype", [name for name in vector_features_name], "label", file=normdata_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, norm_features, label):
            print(i, j, k, [item for item in x], y, file=normdata_filename)

        # Write original dataset to file
        print("ID, formula, prototype", [name for name in vector_features_name], "label", file=data_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, vector_features, label):
            print(i, j, k, [item for item in x], y, file=data_filename)

    # Split data into train-test sets
    #train_test_data_Arr, train_test_label_Arr, predict_data_Arr, predict_label_Arr = split_train_predict(norm_features,
                                                                                                         #norm_label,
                                                                                                         #number_sample)
    #X_train, X_test, Y_train, Y_test = splitDataHO(train_test_data_Arr, train_test_label_Arr, TestSetRatio, RandomSeed)

    # Train and test the model
    mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
        mean_features_importanceArr, std_features_importanceArr = train_test(X_train, X_test, Y_train, Y_test, features_name, screen_step)

    normdata_filename.close()
    data_filename.close()

    # Return the results
    return mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
        mean_features_importanceArr, std_features_importanceArr, \
        fixed_features_name, fixed_features, vector_features_name, vector_features

def fixed_feature_engineering(mean_features_importanceArr, std_features_importanceArr, fixed_features_name,
                              fixed_features, vector_features_name, vector_features, screen_step):
    sorted_features_importance = []
    sorted_features_importance_std = []

    shape_fixed_features = np.shape(fixed_features)

    # Calculate importance of fixed features
    fixed_features_importance = np.float(np.sum(mean_features_importanceArr[0: shape_fixed_features[1]]))
    fixed_features_importance_std = np.sum(std_features_importanceArr[0: shape_fixed_features[1]]) / \
                                    shape_fixed_features[1]

    # Calculate importance of vector features
    vector_features_importance = mean_features_importanceArr[shape_fixed_features[1]: len(mean_features_importanceArr)]
    vector_features_importance_std = std_features_importanceArr[
                                     shape_fixed_features[1]: len(mean_features_importanceArr)]

    # Sort vector features based on importance
    sorted_idx = np.argsort(vector_features_importance)[::-1]
    sorted_vector_features_name = np.array(vector_features_name)[sorted_idx]
    sorted_vector_features_importance = np.array(vector_features_importance)[sorted_idx]
    sorted_vector_features_importance_std = np.array(vector_features_importance_std)[sorted_idx]

    row_number = np.shape(vector_features)[0]
    vfeature_column_number = np.shape(vector_features)[1]

    # Sort vector features based on importance
    sorted_vector_features = np.zeros((row_number, vfeature_column_number))
    for i, j in zip(sorted_idx, np.arange(0, vfeature_column_number)):
        sorted_vector_features[:, j] = vector_features[:, i]

    # Concatenate fixed features with sorted vector features
    sorted_features = np.concatenate((fixed_features, sorted_vector_features), axis=1)

    # Remove last feature name and feature column from sorted vector features
    selected_sorted_vector_features_name = np.delete(sorted_vector_features_name, -1)
    selected_sorted_features = np.delete(sorted_features, -1, axis=1)

    # Concatenate fixed feature names with sorted vector feature names
    sorted_features_name = np.concatenate((fixed_features_name, selected_sorted_vector_features_name), axis=0)

    # Append fixed features importance to the list
    sorted_features_importance.append(fixed_features_importance)

    # Append vector features importance to the list
    for item in sorted_vector_features_importance:
        sorted_features_importance.append(item)

    # Append fixed features importance standard deviation to the list
    sorted_features_importance_std.append(fixed_features_importance_std)

    # Append vector features importance standard deviation to the list
    for item in sorted_vector_features_importance_std:
        sorted_features_importance_std.append(item)

    # Write sorted relative feature importance to file
    font = open(result_path + "sorted_relative_feature_importance_" + str(screen_step) + ".dat", 'a')
    for i, j, k in zip(sorted_features_name, sorted_features_importance, sorted_features_importance_std):
        print(i, j, k, file=font)
    font.close()

    # Return the sorted features and their names
    return sorted_features_name, selected_sorted_features

def vector_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                               vector_features_name, vector_features, screen_step):
    sorted_features_importance = []
    sorted_features_importance_std = []

    # Calculate importance of vector features
    vector_features_importance = mean_features_importanceArr
    vector_features_importance_std = std_features_importanceArr

    # Sort vector features based on importance
    sorted_idx = np.argsort(vector_features_importance)[::-1]
    sorted_vector_features_name = np.array(vector_features_name)[sorted_idx]
    sorted_vector_features_importance = np.array(vector_features_importance)[sorted_idx]
    sorted_vector_features_importance_std = np.array(vector_features_importance_std)[sorted_idx]

    row_number = np.shape(vector_features)[0]
    vfeature_column_number = np.shape(vector_features)[1]

    # Sort vector features based on importance
    sorted_vector_features = np.zeros((row_number, vfeature_column_number))
    for i, j in zip(sorted_idx[1:vfeature_column_number], np.arange(0, vfeature_column_number)):
        sorted_vector_features[:, j] = vector_features[:, i]

    # The sorted features only include vector features
    sorted_features = sorted_vector_features

    # Remove last feature name and feature column from sorted vector features
    selected_sorted_vector_features_name = np.delete(sorted_vector_features_name, -1)
    selected_sorted_features = np.delete(sorted_features, -1, axis=1)

    # The sorted feature names only include vector feature names
    sorted_features_name = selected_sorted_vector_features_name

    # Append vector features importance to the list
    for item in sorted_vector_features_importance:
        sorted_features_importance.append(item)

    # Append vector features importance standard deviation to the list
    for item in sorted_vector_features_importance_std:
        sorted_features_importance_std.append(item)

    # Write sorted relative feature importance to file
    font = open(result_path + "sorted_relative_feature_importance_" + str(screen_step) + ".dat", 'a')
    for i, j, k in zip(sorted_features_name, sorted_features_importance, sorted_features_importance_std):
        print(i, j, k, file=font)
    font.close()

    # Return the sorted features and their names
    return sorted_features_name, selected_sorted_features

path = os.getcwd() + "/"
result_file = 'Results'  # Output results folders
result_subfile = "/Feature_Engineering_Result_"  # Output results subdirectory
n_fixed_features = 23  # the number of fixed features
fixed_features_name = list(['CGMD'])  # the defined name of fixed features
HyperParameter_Step = 1000  # Hyperparameter search steps
LoopStepMin = 0  # Minimum number of iteration loop
LoopStepMax = 20  # Maximum  number of iteration loop

fixed_features = features_train[:, 0: n_fixed_features]
initial_fixed_features_name = features_name_train[0:n_fixed_features]
ScreenStep = np.shape(features_train[:, n_fixed_features: np.shape(features_train)[1]])[1]

if np.shape(features_train)[1] > n_fixed_features:
    if n_fixed_features != 0:
        for i in range(0, ScreenStep):
            screen_step = i
            result_path = path_mkdir(path, result_subfile, screen_step)
            mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
            mean_features_importanceArr, std_features_importanceArr, \
            fixed_features_name, fixed_features, vector_features_name, vector_features = feature_importance(
                features_name_train,
                features_train,
                screen_step)
            selected_sorted_features_name, selected_sorted_features = \
                fixed_feature_engineering(mean_features_importanceArr, std_features_importanceArr, fixed_features_name,
                                    fixed_features, vector_features_name,
                                    vector_features, screen_step)
            features_name_train = np.concatenate(
                (initial_fixed_features_name, selected_sorted_features_name[1: len(selected_sorted_features_name)]),
                axis=0)
            features_train = selected_sorted_features

    else:
        for i in range(0, ScreenStep):
            screen_step = i
            result_path = path_mkdir(path, result_subfile, screen_step)
            mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
            mean_features_importanceArr, std_features_importanceArr, \
            fixed_features_name, fixed_features, vector_features_name, vector_features = feature_importance(
                features_name_train,
                features_train,
                screen_step)
            selected_sorted_features_name, selected_sorted_features = \
                vector_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                                           vector_features_name, vector_features, screen_step)
            features_name_train = selected_sorted_features_name

            features_train = selected_sorted_features

else:
    screen_step = "FF"
    result_path = path_mkdir(path, result_subfile, screen_step)

    # norm_features, norm_label = normData(features_train, label_train)

    predictX = X_test

    mean_pred_trainArr, std_pred_trainArr, \
        mean_pred_testArr, std_pred_testArr, \
        mean_features_importanceArr, std_features_importanceArr = train_test(X_train, X_test, Y_train, Y_test, features_name_train, screen_step)

    if len(Y_test) != len(label_test):
        predict(X_train, Y_train, X_test, Y_test, predictX, screen_step)

