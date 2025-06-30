# challenge-regression
# Description
1. Data Cleaning
  * Starting from the raw immoweb websites, we managed to clean with the following result:

The dataset contains **38,979** records with the following columns:

| Column Name              | Description                           | Data Type | Notes                       |
|--------------------------|-------------------------------------|-----------|-----------------------------|
| `bedroomcount`           | Number of bedrooms                   | float64   |                             |
| `habitablesurface`       | Usable living surface area (m²)     | float64   |                             |
| `haslift`                | Presence of a lift (elevator)        | int64     | 1 = yes, 0 = no             |
| `hasgarden`              | Presence of a garden                  | int64     | 1 = yes, 0 = no             |
| `hasswimmingpool`        | Presence of a swimming pool           | int64     | 1 = yes, 0 = no             |
| `hasterrace`             | Presence of a terrace                 | int64     | 1 = yes, 0 = no             |
| `price`                  | Property price (in euros)             | float64   |                             |
| `hasparking`             | Presence of parking                   | int64     | 1 = yes, 0 = no             |
| `epcscore_encoded`       | Encoded energy performance score     | float64   | Numerical encoding of EPC    |
| `buildingcondition_encoded` | Encoded building condition         | float64   | Numerical encoding           |
| `region_Brussels`        | Region indicator for Brussels         | float64   | 1 = Brussels, else 0         |
| `region_Flanders`        | Region indicator for Flanders         | float64   | 1 = Flanders, else 0         |
| `region_Wallonia`        | Region indicator for Wallonia         | float64   | 1 = Wallonia, else 0         |
| `type_encoded`           | Encoded property type                 | int64     | Categorical encoding         |
| `latitude`               | Geographic latitude                   | float64   |                             |
| `longitude`              | Geographic longitude                  | float64   |

   - Categorigal OE conversion of property type
   - We converted postcodes in geolocalisation and in 2 columns: latitude and longitude. 
   - Regions have been split through OneHotEncoder. 
   - Both EPC Scores and Building Conditions have been perprocessed through Ordinal Encoder. 
   - We created a column hasparking to merge indoor and outdoor. 
   - All boolean features have been filled with 1 or 0.
   - Outliers have been eliminated through 1.5 X the interquantiles + surfaces less than 1500 and bedrooms less than 30.
   - And postal codes with less than 10 properties have been eliminated since this can distorte the prediction, means, stds

2. Data formatting

  Dataset has been split using a Scaler:
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,
random_state=42)
   
3. Model selection
   
* The following models have been selected for all features in the dataset:
  - LinearRegression combined with PolynomialFeatures
  - Ridge
  - Lasso
  - ElasticNet
  - GradientBoostingRegressor
  - RandomForestRegressor
  - XGBRegressor

   
4. Apply your model

  ## Model Performance Comparison

| Algorithm                  | r2_train | r2_test | rmse_train | rmse_test | mae_train | mae_test | evs_train | evs_test |
|---------------------------|----------|---------|------------|-----------|-----------|----------|-----------|----------|
| LinearRegression/Polynomial | 0.63     | 0.63    | 100,055€   | 100,489€  | 73,721€   | 73,847€  | 0.63      | 0.63     |
| Ridge                     | 0.52     | 0.51    | 114,502€   | 114,729€  | 85,019€   | 85,013€  | 0.52      | 0.51     |
| Lasso                     | 0.52     | 0.51    | 114,502€   | 114,729€  | 85,019€   | 85,013€  | 0.52      | 0.51     |
| ElasticNet                | 0.48     | 0.47    | 119,011€   | 119,271€  | 89,862€   | 89,776€  | 0.48      | 0.47     |
| GradientBoosting          | 0.70     | 0.68    | 91,105€    | 93,194€   | 66,976€   | 68,222€  | 0.70      | 0.68     |
| RandomForest              | 0.96     | 0.71    | 33,330€    | 88,819€   | 23,531€   | 63,308€  | 0.96      | 0.71     |
| XGBoost                   | 0.83     | 0.74    | 67,642€    | 84,451€   | 49,157€   | 60,220€  | 0.83      | 0.74     |
   
5. Model evaluation

# Installation

## Requirements:

The project requires the following Python packages:

| Package                | Version      | Package              | Version     |
|------------------------|--------------|----------------------|-------------|
| appnope                | 0.1.4        | matplotlib-inline    | 0.1.7       |
| asttokens              | 3.0.0        | nest-asyncio        | 1.6.0       |
| category_encoders      | 2.8.1        | numpy               | 2.3.1       |
| comm                   | 0.2.2        | packaging           | 25.0        |
| contourpy              | 1.3.2        | pandas              | 2.3.0       |
| cycler                 | 0.12.1       | parso               | 0.8.4       |
| debugpy                | 1.8.14       | patsy               | 1.0.1       |
| decorator              | 5.2.1        | pexpect             | 4.9.0       |
| executing              | 2.2.0        | pillow              | 11.2.1      |
| fonttools              | 4.58.4       | platformdirs        | 4.3.8       |
| ipykernel              | 6.29.5       | prompt_toolkit      | 3.0.51      |
| ipython                | 9.3.0        | psutil              | 7.0.0       |
| ipython_pygments_lexers| 1.1.1        | ptyprocess          | 0.7.0       |
| jedi                   | 0.19.2       | pure_eval           | 0.2.3       |
| joblib                 | 1.5.1        | Pygments            | 2.19.2      |
| jupyter_client         | 8.6.3        | pyparsing           | 3.2.3       |
| jupyter_core           | 5.8.1        | python-dateutil     | 2.9.0.post0 |
| kiwisolver             | 1.4.8        | pytz                | 2025.2      |
| matplotlib             | 3.10.3       | pyzmq               | 27.0.0      |
| scikit-learn           | 1.7.0        | scipy               | 1.16.0      |
| seaborn                | 0.13.2       | six                 | 1.17.0      |
| stack-data             | 0.6.3        | statsmodels         | 0.14.4      |
| threadpoolctl          | 3.6.0        | tornado             | 6.5.1       |
| traitlets              | 5.14.3       | tzdata              | 2025.2      |
| wcwidth                | 0.2.13       |                      |             |

# Usage

# Visuals
## Correlation of all features after preprocessing cleaned data:
![image](https://github.com/user-attachments/assets/b41c64cd-8658-4480-9fc6-420a5709e586)

## Showing features importance using Random Forrester Regressor:
![image](https://github.com/user-attachments/assets/54068764-336d-4250-b69a-93f49ac03f67)

# Contributors
 * Caterina
 * Jordi
 * Emmanuel
