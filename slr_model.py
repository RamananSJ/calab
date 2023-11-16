#without dataset
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,3,4,8])
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(slope)
print(intercept)
def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

#with dataset
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

slope, intercept, r, p, std_err = stats.linregress(x, y)
print(slope)
print(intercept)
def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

