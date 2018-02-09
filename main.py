import sys, os
directory = "E:\Dropbox\Research\experiment data\CeCoIn5_MACS_20180125\macsanalysis"
os.chdir(directory)
sys.path.append(directory)

from DataClass import *
ltFilename = "CeCoIn5_0p1K_4T.txt"
ltData = MACSData(ltFilename) # low temperature data
htFilename = "CeCoIn5_2p5K_4T.txt"
htData = MACSData(htFilename) # high temperature data
subFilename = "CeCoIn5_subtraction_4T.txt"
subData = MACSData(subFilename) # temperature subtraction data

#ltData.plot(view_ax=12, bin_ax1=[-1,0.02,1],bin_ax2=[-2,0.02,0.5],bin_ax3=[0.49,0.05,0.51])
temp = ltData.plot(view_ax=1, bin_ax1=[0,0.02,2], bin_ax2=[0.3,0.7], bin_ax3=[0.29,0.31], foldmode=12)
