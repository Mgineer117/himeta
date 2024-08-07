import numpy as np

# Sample 1D array
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Indices to delete
indices_to_delete = [1, 3, 5]

# Delete the specified indices
new_arr = np.delete(arr, indices_to_delete)

print("Original array:", arr)
print("Array after deleting indices:", new_arr)
