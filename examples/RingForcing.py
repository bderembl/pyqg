import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pyqg
from pyqg.diagnostic_tools import spec_var

# the model object
year = 1.

k_in = 10
k_out = 20

N = 128
param = pyqg.RingForcing(k_in_forc=k_in, k_out_forc=k_out,mag_noise_forc=1e4)

m = pyqg.BTModel(L=2.*pi,nx=N, tmax = 200*year,
        beta = 0., H = 1., rek = 0., rd = None, dt = 0.005,
                     taveint=year, ntd=2,parameterization=param)

# start from rest
qi = np.zeros((1,N,N))
m.set_q(qi)

# run the model and plot some figs
plt.rcParams['image.cmap'] = 'RdBu_r'

plt.ion()

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=15*m.dt):

    plt.clf()
    p1 = plt.imshow(m.q[0] + m.beta * m.y)
    plt.clim([-30., 30.])
    plt.title('t='+str(m.t))
    
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.01)

    plt.draw()

plt.show()
plt.ion()



# forcing

# nhx,nhy = m.wv.shape
# wvx = np.sqrt((m.k)**2.+(m.l)**2.)

# mask = np.ones_like(wvx)
# mask[wvx<=k_in] = 0.
# mask[wvx>k_out] = 0.

# Ring_hat = mask*(np.random.randn(nhx,nhy) +1j*np.random.randn(nhx,nhy))

# Ring = m.ifft( Ring_hat[np.newaxis,:,:] )
# Ring = Ring - Ring.mean()
