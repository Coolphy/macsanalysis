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
        if filename is None:
            print("Warning: no file name specified, please check")
            return
        df = pd.read_csv(filename, skiprows=1, header=None, delim_whitespace=True)
        self.data = df.values
        print("Datafile " + filename + " has been successfully imported. Data dimensions: " + str(self.data.shape))
        return self.data


    def fold(self, foldmode=0):
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
             view_ax1=None, view_ax2=None, view_ax3=None):
        """
        Plot MACS data based on given parameters and return corresponding figure class.
        @param view_ax: the viewing axis, supporting 12, 21, 13, 31, 23, 32, 1, 2, 3
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

        points = self.__select_data__(bin_ax1, bin_ax2, bin_ax3)
        bin_ax = [bin_ax1, bin_ax2, bin_ax3]

        # Generate plot2D class
        if view_ax > 10:

            view_xx = view_ax // 10 - 1 # -1 for 0 index
            view_yy = view_ax % 10 - 1
            bin_xx = bin_ax[view_xx]
            bin_yy = bin_ax[view_yy]

            size_xx = int(np.floor((bin_xx[-1] - bin_xx[0] + 3/2*bin_xx[1]) / bin_xx[1]))
            size_yy = int(np.floor((bin_yy[-1] - bin_yy[0] + 3/2*bin_yy[1]) / bin_yy[1]))
            _intensity = np.zeros((size_xx, size_yy))
            _error = np.zeros((size_xx, size_yy))
            _point_num = np.zeros((size_xx, size_yy))

            for point in points:
                mm = int(np.floor((point[view_xx] - (bin_xx[0] - bin_xx[1]/2)) / bin_xx[1]))
                nn = int(np.floor((point[view_yy] - (bin_yy[0] - bin_yy[1]/2)) / bin_yy[1]))
                _intensity[mm, nn] += point[3]
                _error[mm, nn] = np.sqrt(point[4]**2 + _error[mm, nn]**2)
                _point_num[mm, nn] += 1

            intensity = np.zeros((size_xx, size_yy))
            error = np.zeros((size_xx, size_yy))
            for mm in range(0, size_xx):
                for nn in range(0, size_yy):
                    if (_point_num[mm, nn] == 0):
                        intensity[mm, nn] = None
                        error[mm, nn] = None
                    else:
                        intensity[mm, nn] = _intensity[mm, nn]/_point_num[mm, nn]
                        error[mm, nn] = _error[mm, nn]/_point_num[mm, nn]

            grid_xx, grid_yy = self.__mgrid_generate__(bin_xx, bin_yy)
            plot2D = Plot2D(grid_xx=grid_xx, grid_yy=grid_yy, intensity=intensity, error=error)
            plot2D.plot(view_ax1, view_ax2, view_ax3)
            return plot2D




    def __mgrid_generate__(self, bin_xx, bin_yy):
        """
        Generate grid based on bins.
        @param bin_xx:
        @param bin_yy:
        @return: same return as np.mgrid
        """
        grid_xx, grid_yy = np.mgrid[slice(bin_xx[0] - bin_xx[1]/2, bin_xx[-1] + bin_xx[1]/2 + bin_xx[1], bin_xx[1]),
                                    slice(bin_yy[0] - bin_yy[1]/2, bin_yy[-1] + bin_yy[1]/2 + bin_yy[1], bin_yy[1])]
        return grid_xx, grid_yy

    def __select_data__(self, bin_ax1, bin_ax2, bin_ax3):
        """
        internal method, select data in the given box
        @param bin_ax1:
        @param bin_ax2:
        @param bin_ax3:
        @return: np.ndarray data type.
        """
        data = self.data
        return data[((data[:,0] >= bin_ax1[0]) & (data[:,0] <= bin_ax1[-1])
                    & (data[:,1] >= bin_ax2[0]) & (data[:,1] <= bin_ax2[-1])
                    & (data[:,2] >= bin_ax3[0]) & (data[:,2] <= bin_ax3[-1]))]

    def __init__(self, filename=None):
        if filename is not None:
            self.filename = filename
            self.importdata(filename)
        else:
            print("Warning: no filename is specified.")



class Plot1D:
    def plot(self, view_ax1, view_ax2, view_ax3):
        fig, ax = plt.figure()
        ax.errorbar(x=self.data[:,0], y=self.data[:,1], yerr=self.data[:,2])
        plt.show()
        plt.ion()
        return fig, ax


    def __init__(self, data):
        self.data = data


class Plot2D:

    def plot(self, view_ax1=None, view_ax2=None, view_ax3=None):
        plt.pcolor(self.grid_xx, self.grid_yy, self.intensity)
        plt.clim(0, 1)
        plt.set_cmap('jet')
        plt.colorbar()
        plt.show()

    def __init__(self, grid_xx, grid_yy, intensity, error):
        self.grid_xx = grid_xx
        self.grid_yy = grid_yy
        self.intensity = intensity
        self.error = error
