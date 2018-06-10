import numpy as np
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt

df=pd.read_csv("Uy0a.csv")
dt=8.510498e-06
n=len(df)

#time domain
t=np.linspace(0,n*dt,n)
yt=df.iloc[:,0]

#fft
yf=fftpack.fft(yt)/(n/2.0)
freq=fftpack.fftfreq(n,dt)

#plot
plt.plot(t,yt)
plt.ylabel("Uy [m]")
plt.xlabel("Time [sec]")
plt.show()
plt.plot(freq[1:n//2], np.angle(yf[1:n//2])*180.0/np.pi)
plt.xlim(0,6000)
plt.ylabel("Phase [deg]")
plt.show()
plt.plot(freq[1:n//2], np.abs(yf[1:n//2]))
plt.xlim(0,6000)
plt.ylim(1.0e-9,1.0e-2)
plt.yscale("log")
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.annotate('163.5Hz', xy=(1.635178765297e+02, 1.0E-4), xytext=(1.635178765297e+02, 1.0E-3),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('886.4Hz', xy=(8.863935606048e+02, 1.0E-5), xytext=(8.863935606048e+02, 1.0E-4),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('2129.5Hz', xy=(2.129515131439e+03, 1.0E-5), xytext=(2.129515131439e+03, 1.0E-4),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('3577.3Hz', xy=(3.577305392139e+03, 1.0E-6), xytext=(3.577305392139e+03, 1.0E-5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('5152.8Hz', xy=(5.152815326664e+03, 1.0E-6), xytext=(5.152815326664e+03, 1.0E-5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

            
plt.show()

