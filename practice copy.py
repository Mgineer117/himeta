import numpy as np
n = 10
n -= 1
# Original 1D array
x = np.arange(30)

# Expanding to 2D array
expanded_array = np.tile(x[:, np.newaxis], (1, 5))
rows, cols = expanded_array.shape

converted_data = np.zeros((rows, cols))

idx = rows - n
converted_data[:idx, :] = expanded_array[n:, :]
converted_data[idx:, :] = expanded_array[-1, :]

print(expanded_array)
print(converted_data)
