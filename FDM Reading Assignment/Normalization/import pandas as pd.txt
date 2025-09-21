import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create a new sample dataset with 'Temperature' and 'Humidity'
data_new = pd.DataFrame({
    'Temperature': [15, 25, 30, 10, 20],
    'Humidity': [50, 65, 75, 40, 55]
})

print(f"Before Normalization:\n{data_new}")

# Apply Min-Max Normalization to the new dataset
scaler_new = MinMaxScaler()
normalized_data_new = pd.DataFrame(scaler_new.fit_transform(data_new), columns=data_new.columns)

print(f"\nAfter Normalization:\n{normalized_data_new}")