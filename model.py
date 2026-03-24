import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dummy dataset
data = {
    'area': [500, 800, 1000, 1200, 1500],
    'bedrooms': [1, 2, 2, 3, 4],
    'price': [100000, 150000, 200000, 250000, 300000]
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")