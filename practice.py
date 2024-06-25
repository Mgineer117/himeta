# Define multiple dictionaries
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
dict3 = {'d': 5}

# Use update method
combined_dict = dict1.copy()  # Make a copy of dict1 to avoid modifying it
combined_dict.update(dict2)
combined_dict.update(dict3)

print(combined_dict)
# Output: {'a': 1, 'b': 3, 'c': 4, 'd': 5}
