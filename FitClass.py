from DataClass import *
import PyQt5
import matplotlib.pyplot as plt
import numpy as np
import ufit

class Fit1D:
    def __init__(self, plot1D=None, model=None, name=None):
        self.name = name
        if plot1D:
            self.add_data(plot1D)
        if model:
            self.add_model(model)
        else:
            self.model=None

    def add_model(self, model):
        self.model = model

    def add_data(self, plot1D):
        """
        read Plot1D type of data. Eliminate "nan" points in the raw data.
        :param plot1D: instance of Plot1D class.
        :return:
        """
        index_not_nan = ~np.isnan(plot1D.intensity)
        self.grid_xx= plot1D.grid_xx[index_not_nan]
        self.intensity = plot1D.intensity[index_not_nan]
        self.error = plot1D.error[index_not_nan]
        if self.name is None:
            self.name = plot1D.name


    def fit(self):
        if self.model is None:
            raise ValueError("No model specified, please check.")
        data = ufit.as_data(x=self.grid_xx, y=self.intensity, dy=self.error, name=self.name)
        fitresult = self.model.fit(data)
        fitresult.printout()
        plt.figure()
        fitresult.plot()
        self.fitresult = fitresult

