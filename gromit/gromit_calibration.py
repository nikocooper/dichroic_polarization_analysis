'''
Calibration of Gromit on 4/24/26 from an air measurement. 
Retardance of the QWPs seems to be fitting high, the QWPs in question 
are the BVO AQWP3s, which nominally have 90 deg of retardance at 610 nm.
'''

from katsu.mueller import (
    linear_polarizer,
    linear_retarder,
    linear_diattenuator
)

import numpy as np
import matplotlib.pyplot as plt
rad2deg = 180 / np.pi
deg2rad = 1 / rad2deg
plt.style.use('bmh')
import os
from PIL import Image


def load_images(
    directory,
    extension=".tiff"
):
    """
    Load numbered images from a directory.

    Images must be named like: 1.tiff, 2.tiff, ...

    Parameters
    ----------
    directory : str
        Folder containing numbered images.
    extension : str
        File extension (default ".tiff").

    Returns
    -------
    list of 2D numpy arrays
        Images in numeric order. images[i] corresponds to (i+1).tiff
    """


    # collect and sort numeric filenames
    file_numbers = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(extension):
            try:
                num = int(os.path.splitext(fname)[0])
                file_numbers.append(num)
            except ValueError:
                pass

    file_numbers.sort()

    images = []

    for num in file_numbers:
        path = os.path.join(directory, f"{num}{extension}")
        img = np.array(Image.open(path), dtype=np.float64)

        images.append(img)

    return images


powers = load_images(r"gromit\042426_air610")
powers = np.array(powers)
power = powers[:, 563, 1196]
NMEAS=len(power) - 1

# angles of the PSG and PSA QWPs, in radians
psg_angles = np.linspace(0, np.pi, NMEAS+1)
psg_angles = psg_angles
psa_angles = psg_angles * 5
print("PSG angles:", psg_angles * rad2deg)
print("PSA angles:", psa_angles * rad2deg)
# setup parameters of Gromit
starting_angles={'psg_polarizer':0,
                'psg_qwp':0,
                'psa_qwp':-45 * deg2rad,
                'psa_polarizer':0}
ret = np.pi/2
# construct the model measurement
psg_angles_longer = np.linspace(0, np.pi,1000)
psg_angles_longer = psg_angles_longer
psa_angles_longer = psg_angles_longer * 5

psg_qwp = linear_retarder(starting_angles['psg_qwp'] + psg_angles_longer,ret, shape=[1000])
psg_hpl = linear_polarizer(starting_angles['psg_polarizer'], shape=[1000])

psa_qwp = linear_retarder(starting_angles['psa_qwp'] + psa_angles_longer, ret, shape=[1000])
psa_hpl = linear_polarizer(starting_angles['psa_polarizer'], shape=[1000]) 


Mg =  psg_qwp @ psg_hpl
Ma = psa_hpl @ psa_qwp
Msys = Ma @ Mg

power_model = Msys[:,0,0]


plt.figure()
plt.title('Before Fit')
plt.plot(psg_angles * rad2deg, power, linestyle='None', marker='o')
plt.plot(psg_angles_longer * rad2deg, power_model * np.max(power) * 2)
plt.ylabel('Counts')
plt.xlabel('PSG Angle, deg')
plt.show()

# returns a model of the power measurement for given parameters, used in optimization
def drrp_sinusoid_step(thetas_g, thetas_a, psg_pol_angle, psa_pol_angle, psg_ret, psa_ret, psg_qwp_angle, psa_qwp_angle):
    
    psg_qwp_ret = psg_ret
    psa_qwp_ret = psa_ret
    
    psg_dia_tmin = 0
    psa_dia_tmin = 0
    
    
    thetas_g = thetas_g 
    thetas_a = thetas_a
    
    starting_angles={'psg_polarizer':psg_pol_angle,
                     'psg_qwp':psg_qwp_angle,
                     'psa_qwp':psa_qwp_angle,
                     'psa_polarizer':psa_pol_angle}

    nmeas = len(thetas_g)

    psg_qwp = linear_retarder(starting_angles['psg_qwp'] + thetas_g, psg_qwp_ret, shape=[nmeas])
    psg_hpl = linear_diattenuator(starting_angles['psg_polarizer'], psg_dia_tmin, shape=[nmeas])

    psa_qwp = linear_retarder(starting_angles['psa_qwp'] + thetas_a, psa_qwp_ret, shape=[nmeas])
    psa_hpl = linear_diattenuator(starting_angles['psa_polarizer'], psa_dia_tmin, shape=[nmeas])
    
    psg = psg_qwp @ psg_hpl
    psa = psa_hpl @ psa_qwp
    M = psa @ psg

    return M[:,0,0]

# Cost function for optimization
def minimize_step_error(x):
    
    nmeas = len(psg_angles)
    y = x[0] 
    z = x[1] 
    psg_ret = x[-4]
    psa_ret = x[-3]
    psg_angle = x[-2]
    psa_angle = x[-1]
    
    pow = drrp_sinusoid_step(psg_angles, psa_angles, y, z, psg_ret, psa_ret, psg_angle, psa_angle)
    scaled_data = (power / np.max(power) * 0.5)
    diff = pow - scaled_data
    cost = np.mean((diff)**2)
    
    return cost

from scipy.optimize import minimize

results = minimize(minimize_step_error,
                   x0=(0, # psg lp sarting angle
                       0, # psa lp starting angle
                       90*deg2rad,
                       90*deg2rad,
                       0 * deg2rad, # psg offset
                       -45 * deg2rad), # psa offset
                   method='BFGS',
                   options={'maxiter':1000, 'xrtol':1e-10})
params = ['PSG step size multiplier',
          'PSA step size multiplier',
          'PSG QWP ret',
          'PSA QWP ret',
          'PSG Offset Angle, deg',
          'PSA Offset Angle, deg']

for i, (p,u) in enumerate(zip(params,results.x)):
    
    print(p,u*rad2deg)

power_model_fit = drrp_sinusoid_step(psg_angles_longer, psa_angles_longer, *results.x)
plt.style.use('bmh')

plt.figure()
plt.title('After Fit')
plt.plot(psg_angles * rad2deg, power, linestyle='None', marker='o', label='Measured Data')
plt.plot(psg_angles_longer * rad2deg, power_model_fit * np.max(power) * 2, label='Model')
plt.ylabel('Counts')
plt.xlabel('PSG Angle, deg')
plt.legend()
plt.show()