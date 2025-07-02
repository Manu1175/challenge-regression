def modelCAT(df, target, obs, results_df=None):
    """
    Function to run CatBoost algorithm with CV=10 and default parameters
    Returns the regression results and the feature importances
    """

    # Split features and target
    X, y = splitDF(1, df, target)
    
    # Initialize model
    model = CatBoostRegressor(verbose=0, random_state=123)
    
    # Setup 10-fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=123)
    
    # Cross-validated predictions
    y_cv_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Fit on full data
    model.fit(X, y)
    
    # Predict on full data
    y_pred = model.predict(X)
    
    # Get regression results
    results = regResults(obs, "CatBoost", y, y_pred)
    
    # Prepare results dictionary for appending
    combined_result = {
        "Model": "CatBoost",
        "Features": obs,
        "Runtime (min)": None,
        # Train metrics (full data)
        **{k + " (Train)": v for k, v in results.items() if k not in ["Model", "Observations", "Features"]},
        # CV metrics from cross_val_predict
        **{k + " (Test)": v for k, v in regResults(obs, "CatBoost CV", y, y_cv_pred).items()
           if k not in ["Model", "Observations", "Features"]}
    }
    
    # Append to results_df
    if results_df is not None:
        results_df = pd.concat([results_df, pd.DataFrame([combined_result])], ignore_index=True)
    else:
        results_df = pd.DataFrame([combined_result])
    
    # Feature importances sorted descending
    importances = model.get_feature_importance()
    feature_names = X.columns
    catboost_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    return results_df, catboost_importance


# Rerunning our best model with all features
# and finding most important features

start = time.time()
results_df, catboost_importance = modelCAT(df_log, target, "Rerun - All features", results_df)
end = time.time()

code_run = round((end - start) / 60, 3)
results_df.at[results_df.index[-1], "Runtime (min)"] = code_run

results_df = results_df.sort_values("RMSE (Test)", ascending=True)

print("\nSummary of Regression Results:")
print(results_df)


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


plotCatBoostImportance(catboost_importance)