import pandas as pnd
from sklearn.linear_model import LinearRegression

def load_data():
    return pnd.read_csv("bmi_and_life_expectancy.csv")

def create_linear_regression_model(bmi_life_data):
    bmi_life_model = LinearRegression()
    return bmi_life_model.fit(bmi_life_data[['BMI']],bmi_life_data[['Life expectancy']])

if __name__ == '__main__':
    bmi_life_data = load_data()
    bmi_life_model = create_linear_regression_model(bmi_life_data)
    laos_life_exp = bmi_life_model.predict([[21.07931]])
    print (laos_life_exp)
