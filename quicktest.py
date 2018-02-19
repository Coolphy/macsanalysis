from matplotlib import pyplot as plt
import numpy as np
xx = np.arange(0, 10, 1)
yy = xx ** 2

"""
plt.plot(xx, yy,
plt.xlabel("x")
plt.show()
"""


def myplot(xx, yy, *argv, **kwargs):
    return plt.plot(xx, yy, *argv, **kwargs)


myfigure = myplot(xx,yy,marker="x",linestyle='None')
plt.show()

