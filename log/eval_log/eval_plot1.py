import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'eval.csv'

# Read specific columns (16 to 31) from the CSV file
data = pd.read_csv(file_path, usecols=range(17, 32), index_col=False).to_numpy()

# Display the first few rows of the DataFrame
train_data = data[:, :10]
test_data = data[:, 10:]



train_mean = np.mean(np.mean(train_data, axis=-1), axis=-1)
train_std = np.std(np.mean(train_data, axis=-1), axis=-1)

test_mean = np.mean(np.mean(test_data, axis=-1), axis=-1)
test_std = np.std(np.mean(test_data, axis=-1), axis=-1)

print(train_mean, train_std)
print(test_mean, test_std)