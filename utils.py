# Preprocessing libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split

# Data wrangling libraries
import numpy as np

# Visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt


def regResults(obs, name, y_true, y_pred):
    """
    Function to return regression metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Return a dictionary with the metrics
    return {
        "Features": obs,
        "Model": name,
        "R² Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }

def splitDF(n, df, target):
    """
    Function to split the dataframe based on the requirements on the regression model we need to run
    """
    
    # First split: isolate the target variable
    y = df[target]
    X = df.drop([target], axis=1)
    
    # Second split: train and test as needed to run cross_val_predict
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Further split training: training and validation sets for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=123)
   
    if n == 1:
        return X, y
    
    elif n == 2:
        return X_train_full, y_train_full, X_test, y_test

    elif n == 3:
        return X_test, X_train, X_val, y_test, y_train, y_val
    
    else:
        raise ValueError(f"Invalid mode n={n}. Must be 1, 2 or 3.")

def plotCatBoostImportance(catboost_importance):
    """
    Plot the features importance in descending order
    """
   
    plt.figure(figsize=(8, 5))
    sns.barplot(data=catboost_importance, x="Importance", y="Feature", color="steelblue")
    plt.title(f"CatBoost - Feature Importances", fontsize=13, fontweight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()