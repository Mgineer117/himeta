import numpy as np
import matplotlib.pyplot as plt

file_path = 'eval.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1) 

data = data[:, 17:32]

names = ['Reach', 'Push', 'Pick-Place', 'Door-open', 'Drawer-close', 'Button-press', 'Peg-insert-side', 'Window-open', 'Sweep', 'Basketball']

X = np.arange(0, 5001, 200)

Y = np.array(data)
Y = Y.T

for i, name in enumerate(names):
    plt.plot(X, Y[i, :])
    plt.title(name)
    plt.savefig('figs/' + name + '.png')
    plt.close()


plt.plot(X, np.mean(Y, axis=0))
plt.savefig('figs/_MEAN.png')
#plt.show()


#data = data.T

#X = np.linspace(0, 5000, 200)
#


