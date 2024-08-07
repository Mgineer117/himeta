import torch

class ElevatedSumCalculator:
    def find_elevated_sum(self, actions, y, masks):
        y = torch.argmax(y, dim=-1)
        task_idx = (y[:-1] != y[1:]).nonzero(as_tuple=True)[0] + 1
        mask_idx = torch.where(masks == 0)[0]

        elevated_sum = torch.zeros_like(actions)

        prev_idx = 0
        for m_idx in mask_idx:
            boolean = torch.logical_and(task_idx >= prev_idx, task_idx <= m_idx)
            if len(task_idx[boolean]) != 0:
                for y_idx in task_idx[boolean]:
                    elevated_sum[prev_idx] = actions[prev_idx]
                    for t in range(prev_idx+1, y_idx + 1):
                        elevated_sum[t] = elevated_sum[t - 1] + actions[t]
                    prev_idx = y_idx + 1

                elevated_sum[prev_idx] = actions[prev_idx]
                for t in range(prev_idx+1, m_idx + 1):
                    elevated_sum[t] = elevated_sum[t - 1] + actions[t]
                prev_idx = m_idx + 1
            else:
                elevated_sum[prev_idx] = actions[prev_idx]
                for t in range(prev_idx+1, m_idx + 1):
                    elevated_sum[t] = elevated_sum[t - 1] + actions[t]
                prev_idx = m_idx + 1

        return elevated_sum

# Define the input tensors
n = 3

# Generate actions tensor such that each row has increasing values
actions = torch.tensor([[i + 1] * 4 for i in range(1000)], dtype=torch.float32)

# Define y and masks based on the given pattern
y = torch.tensor([3, 5, 10, 160, 320, 480, 660, 820, 980] , dtype=torch.int32).unsqueeze(1)
masks = torch.tensor([499, 998], dtype=torch.int32)

# Initialize the calculator and compute the elevated sum
calculator = ElevatedSumCalculator()
result = calculator.find_elevated_sum(actions, y, masks)

# Print the result
print(result)
