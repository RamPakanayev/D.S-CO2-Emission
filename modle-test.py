import learn as learn
import numpy as np
import pandas as pd
from numpy.distutils.fcompiler import none
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from constants import RELEVANT_PROPERTIES_FOR_MODEL
from sklearn.ensemble import RandomForestRegressor
import csv
import pickle

#active this function just once
#    df = pd.read_csv('big table complete-before binary cols.csv', header=0, sep=',',thousands=',')
 #   df=df.drop(['Unnamed: 0'],axis=1)
  #  df['countries'] = pd.factorize(df['countries'])[0]
   # df.to_csv("fixedDataTest.csv")
    #return df

def train_model():
    country_data = pd.read_csv('./fixedDataTest.csv')
    # cars_data = pd.read_csv('./EnvironmentTable-Final.csv',thousands=",")
    for col in country_data.columns:
        if col not in RELEVANT_PROPERTIES_FOR_MODEL:  # page of contries enter here
            country_data = country_data.drop([col], axis=1)

    country_data = pd.get_dummies(country_data, drop_first=True)  # ), drop_first=True)
    country_data = country_data[['CO2_TotalMt', 'countries', 'Year', 'CO2_Kg1k',
    'CO2_Tons_per_capita', 'Generation_GWh', 'consumption_GWh', 'consumption_per_capita_kW',
    'installed_capacity_MW', 'Renewable_installed_capacity_MW', 'Renewable_generation_GWh',
    'Renewable_percentage',
    'Population']]
    print("Columns")

   # print(country_data.columns)
    #print(country_data.corr())
    x = country_data.iloc[:, 1:]
    y = country_data.iloc[:, 0]

    print("Hyperparameter Optimization")
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}
    #print(grid)

    print("Train Test Split")
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

    print("Training the Model")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    hyp = RandomizedSearchCV(estimator=model,
                             param_distributions=grid,
                             n_iter=10,
                             scoring='neg_mean_squared_error',
                             cv=5, verbose=2,
                             random_state=42, n_jobs=1)
    print("Train")
    hyp.fit(x_train, y_train)
    print("Predict")
    y_pred = hyp.predict(x_test)
    print(y_pred)
    r2_result = r2_score(y_pred, y_test)
    print(f"Model score: {r2_result}")
    print("Save")
    file = open("fileNew.pkl", "wb")
    pickle.dump(hyp, file)
    #return y_pred, y_test


def predict_now(countries, Year, CO2_Kg1k, CO2_Tons_per_capita, Generation_GWh, consumption_GWh,
                        consumption_per_capita_kW, installed_capacity_MW, Renewable_installed_capacity_MW,
                        Renewable_generation_GWh, Renewable_percentage, Population):
    model = pickle.load(open("./fileNew.pkl", "rb"))
    prediction = model.predict([[countries, Year, CO2_Kg1k, CO2_Tons_per_capita, Generation_GWh, consumption_GWh,
                        consumption_per_capita_kW, installed_capacity_MW, Renewable_installed_capacity_MW,
                        Renewable_generation_GWh, Renewable_percentage, Population]])
    prediction_result = round(prediction[0], 2)
    print(f"The prediction is: {prediction_result}")


if __name__ == '__main__':
    train_model()

    #china 2012
    print("China predicting co2 in 2017:")
    result1 = predict_now(countries=37, Year=2017, CO2_Kg1k=0.55, CO2_Tons_per_capita=7.75
                         , Generation_GWh=6286207, consumption_GWh=5953577,
                         consumption_per_capita_kW=4252.2, installed_capacity_MW=1790785,
                         Renewable_installed_capacity_MW=621741,
                         Renewable_generation_GWh=1667023, Renewable_percentage=26.52,
                         Population=1400110000)

    print("South-Africa predicting co2 in 2016:")
    result2 = predict_now(countries=43,
                         Year=2017,
                         CO2_Kg1k=0.64,
                         CO2_Tons_per_capita=8.13
                         , Generation_GWh=235720,
                         consumption_GWh=207189,
                         consumption_per_capita_kW=3628.7,
                         installed_capacity_MW=53862,
                         Renewable_installed_capacity_MW=6484,
                         Renewable_generation_GWh=9391,
                         Renewable_percentage=3.98,
                         Population=57098000)


    print("Australia predicting co2 in 2019:")
    result3 = predict_now(countries=20,
                         Year=2019,
                         CO2_Kg1k=0.33,
                         CO2_Tons_per_capita=16.49
                         , Generation_GWh=249996,
                         consumption_GWh=237388,
                         consumption_per_capita_kW=9288.2,
                         installed_capacity_MW=75540,
                         Renewable_installed_capacity_MW=27174,
                         Renewable_generation_GWh=51591,
                         Renewable_percentage=20.64,
                         Population=25558000)

    print("Japan predicting co2 in 2019:")
    result4 = predict_now(countries=19,
                          Year=2019,
                          CO2_Kg1k=0.22,
                          CO2_Tons_per_capita=8.98
                          , Generation_GWh=984243,
                          consumption_GWh=940149,
                          consumption_per_capita_kW=7450.3,
                          installed_capacity_MW=322606,
                          Renewable_installed_capacity_MW=92832,
                          Renewable_generation_GWh=222934,
                          Renewable_percentage=22.65,
                          Population=126190000)


    print("Israel predicting co2 in 2015:")
    result5 = predict_now(countries=25,
                          Year=2015,
                          CO2_Kg1k=0.21,
                          CO2_Tons_per_capita=8.9
                          , Generation_GWh=60445,
                          consumption_GWh=52776,
                          consumption_per_capita_kW=6300.1,
                          installed_capacity_MW=16788,
                          Renewable_installed_capacity_MW=813,
                          Renewable_generation_GWh=1214,
                          Renewable_percentage=2.01,
                          Population=8377000)