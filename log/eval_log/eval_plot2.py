import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'eval.csv'

train_names = ['reach', 'push', 'pick-place', 'door-open', 'drawer-close', 'button-press', 'peg-insert-side', 'window-open',' sweep', 'basketball']
test_names = ['drawer-open', 'door-close', 'shelf-place', 'sweep-into', 'lever-pull']

# Read specific columns (16 to 31) from the CSV file
data = pd.read_csv(file_path)#.to_numpy()#.to_numpy()
headers = data.columns
data = data.to_numpy()

train_r_data = {}
test_r_data = {}

train_s_data = {}
test_s_data = {}

for i, title in enumerate(headers):
    components = title.split('/')
    if components[0] == 'eval_reward_mean':
        if components[1] in train_names:
            train_r_data[components[1]] = data[-3, i]
        elif components[1] in test_names:
            test_r_data[components[1]] = data[-3, i]
    elif components[0] == 'eval_success_mean':
        if components[1] in train_names:
            train_s_data[components[1]] = data[-3, i]
        elif components[1] in test_names:
            test_s_data[components[1]] = data[-3, i]
    else:
        pass

data = []
for k, v in train_r_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'train reward {np.mean(data, axis=-1)}/{np.std(data, axis=-1)}')

data = []
for k, v in test_r_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'test reward {np.mean(data, axis=-1)}/{np.std(data, axis=-1)}')

data = []
for k, v in train_s_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'train success {np.mean(data, axis=-1)}/{np.std(data, axis=-1)}')

data = []
for k, v in test_s_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'test success {np.mean(data, axis=-1)}/{np.std(data, axis=-1)}')

'''
data = []
for k, v in train_r_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'train reward {np.mean(np.mean(data, axis=-1), axis=-1)}/{np.std(np.mean(data, axis=-1), axis=-1)}')

data = []
for k, v in test_r_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'test reward {np.mean(np.mean(data, axis=-1), axis=-1)}/{np.std(np.mean(data, axis=-1), axis=-1)}')

data = []
for k, v in train_s_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'train success {np.mean(np.mean(data, axis=-1), axis=-1)}/{np.std(np.mean(data, axis=-1), axis=-1)}')

data = []
for k, v in test_s_data.items():
    data.append(v)
data = np.stack(data, axis=-1)
print(f'test success {np.mean(np.mean(data, axis=-1), axis=-1)}/{np.std(np.mean(data, axis=-1), axis=-1)}')
'''