import torch

# Generate a sample 10000 x 5 tensor of probabilities
# Here, we use random values as an example
torch.manual_seed(0)  # For reproducibility
probabilities = torch.rand(10000, 5)

# Step 1: Compute the index of the maximum probability for each row
max_indices = torch.argmax(probabilities, dim=1)
print(max_indices[:20])

# Step 2: Identify where the max probability index changes
change_indices = (max_indices[:-1] != max_indices[1:]).nonzero(as_tuple=True)[0] + 1

# Print the result
print(change_indices[:20])

a = torch.tensor([10, 20, 30])

prev_i = 0
for i in a:
    boolean = torch.logical_and(change_indices > prev_i, change_indices <= i)
    change_indices[boolean]
    prev_i  = i + 1
