from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from pylab import randn
axes = plt.axes()
axes.set_xlim([0, 80])
axes.set_ylim([0, 240])



X = np.random.randint(1,80,55)
Y = np.random.randint(1,240,55)

plt.xlabel('Age')
plt.ylabel('Minutes Watching Tv Daily')

plt.scatter(X,Y)
plt.show()
