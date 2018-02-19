import PyQt5
import numpy as np
import matplotlib.pyplot as plt
import ufit

xx = np.linspace(-10,10,501)
yy = 0.01 * np.random.rand(np.size(xx)) + np.exp(- xx ** 2)
err = 0.001 * np.random.rand(np.size(xx))
#plt.errorbar(xx, yy, err)

data1 = ufit.as_data(x=xx, y=yy, err= rr, name='mplot')
m = ufit.Background(bkgd=1) + ufit.Gauss('p1', pos='delta', ampl=5, fwhm=0.5)
m.add_params(delta=0)

fitresult = m.fit(data1)
fitresult.printout()
fitresult.plot()
plt.xlim((-5, 5))
plt.show()
