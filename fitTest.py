import PyQt5
import numpy as np

from ufit.lab import *

xx = np.linspace(-10,10,501)
yy = 0.01 * np.random.rand(size(xx)) + np.exp(- xx ** 2)
err = 0.001 * np.random.rand(size(xx))
plt.errorbar(xx, yy, err)


data1 = as_data(xx, yy, err, name='mplot')
m = Background(bkgd=1) + Gauss('p1', pos='delta', ampl=5, fwhm=0.5)
m.add_params(delta=0)

fitresult = m.fit(data1)
fitresult.printout()
fitresult.plot()
xlim((-5, 5))
plt.show()
