import numpy as np

# Create an example array
arr = np.array([1, 2, 3, 4, 5, 6])

# Use np.where to find indices of elements that are greater than 3
indices = np.where(arr > 3)

print(f"Array: {arr}")
print(f"Indices of elements greater than 3: {len(indices[0])}")
print(f"Elements greater than 3: {arr[indices]}")
