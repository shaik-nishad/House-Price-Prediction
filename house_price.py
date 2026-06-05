# House Price Prediction using Linear Regression
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1000, 1200, 1500, 1800, 2000],
    'Price': [2000000, 2500000, 3000000, 3500000, 4000000]
}

df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']

model = LinearRegression()

model.fit(X, y)

area = int(input("enter house area in square feet: "))
prediction = model.predict([[area]])

print("Estimated House Price:", prediction[0])