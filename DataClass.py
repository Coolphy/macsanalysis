import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class MACSData:

    def importdata(self, filename):
        """
        read in MACS projection data type.
        @param filename: string type, full name including suffix is required. Example: "mydata.txt"
        @return: np.ndarray type.
        """
        df = pd.read_csv(filename, skiprows=1, header=None, delim_whitespace=True)
        self.data = df.values
        print("Datafile " + filename + " has been successfully imported. Data dimensions: " + str(self.data.shape))
        return self.data


    def fold(self, foldmode = 0):
        """
        fold data
        @param foldmode: one of following options, 1, 2, 12
                         0 no change
                         1 for folding along ax1
                         2 for folding along ax2
                         12 for folding alont ax1 and ax2
        @return: self.data changes by this method.
        """
        if foldmode is 0:
            return
        if foldmode == 1:
            self.data[:, 1] = abs(self.data[:, 1])
            return
        if foldmode == 2:
            self.data[:, 0] = abs(self.data[:, 0])
            return
        if foldmode == 12:
            self.data[:, 0] = abs(self.data[:, 0])
            self.data[:, 1] = abs(self.data[:, 1])
            return


    def plot(self, view_ax=12,
             bin_ax1=[-20,0.02,20], bin_ax2=[-20,0.02,20], bin_ax3=[-20,0.5,40],
             foldmode=0,
             view_ax1=[], view_ax2=[], view_ax3=[]):
        """
        Plot MACS data based on given parameters and return corresponding figure class.
        @param view_ax: the viewing axis,
                        12 for slice formed by ax1, ax2,
                        13 for slice formed by ax1, ax3,
                        23 for slice formed by ax2, ax3;
                        1 for cut along ax1,
                        2 for cut along ax2,
                        3 for cut ax3
        @param bin_ax1: [ax1_bin_min, ax1_bin_step, ax1_bin_max]
        @param bin_ax2: [ax2_bin_min, ax2_bin_step, ax2_bin_max]
        @param bin_ax3: [ax3_bin_min, ax3_bin_step, ax3_bin_max]
        @param view_ax1: [ax1_plot_min, ax1_plot_max]
        @param view_ax2: [ax2_plot_min, ax2_plot_max]
        @param view_ax3: [ax3_plot_min, ax3_plot_max]
        @return: 1D or 2D macs figure class.
        """
        self.fold(foldmode=foldmode)

        if bin_ax1[0] < -10:
            bin_ax1[0] = min(self.data[:, 0])
        if bin_ax1[-1] > 10:
            bin_ax1[-1] = max(self.data[:, 0])
        if bin_ax2[0] < -10:
            bin_ax2[0] = min(self.data[:, 1])
        if bin_ax2[-1] > 10:
            bin_ax2[-1] = max(self.data[:, 1])
        if bin_ax3[0] < -10:
            bin_ax3[0] = min(self.data[:, 2])
        if bin_ax3[-1] > 20:
            bin_ax3[-1] = max(self.data[:, 2])

        data_selected = self.__selectdata__(bin_ax1, bin_ax2, bin_ax3)

        # Generate the grid_xx and grid_yy for creating plot2D instance.
        if view_ax == 12:
            grid_xx, grid_yy = self.__mgrid_generate__(bin_ax1, bin_ax2)
        if view_ax == 13:
            grid_xx, grid_yy = self.__mgrid_generate__(bin_ax1, bin_ax3)
        if view_ax == 23:
            grid_xx, grid_yy = self.__mgrid_generate__(bin_ax2, bin_ax3)

        # Generate the grid_xx
        




    def __mgrid_generate__(self, bin_xx, bin_yy):
        """
        Generate grid based on bins.
        @param bin_xx:
        @param bin_yy:
        @return: same return as np.mgrid
        """
        grid_xx, grid_yy = np.mgrid[slice(bin_xx[0] - bin_xx[1]/2, bin_xx[2] + bin_xx[1] + bin_xx[1]/2, bin_xx[1]),
                                    slice(bin_yy[0] - bin_yy[1]/2, bin_yy[2] + bin_yy[1] + bin_yy[1]/2, bin_yy[1])]
        return grid_xx, grid_yy



        
    def __selectdata__(self, bin_ax1, bin_ax2, bin_ax3):
        """
        internal method, select data in the given box
        @param bin_ax1:
        @param bin_ax2:
        @param bin_ax3:
        @return: np.ndarray data type.
        """
        data = self.data
        return data[(data[:,0] >= bin_ax1[0]) and (data[:,0] <= bin_ax1[-1])
                    and (data[:,1] >= bin_ax2[0]) and (data[:,1] <= bin_ax2[-1])
                    and (data[:,2] >= bin_ax3[0]) and (data[:,2] <= bin_ax3[-1])]


    def __init__(self, filename):
        self.filename = filename
        self.importdata(filename)
        


class plot1D:
    def plot(self, view_ax1, view_ax2, view_ax3):
        fig, ax = plt.figure()
        ax.errorbar(x=self.data[:,0], y=self.data[:,1], yerr=self.data[:,2])
        plt.show()
        plt.ion()
        return fig, ax


    def __init__(self, data):
        self.data = data


class plot2D:

    def plot(self, view_ax1, view_ax2, view_ax3):
        print()