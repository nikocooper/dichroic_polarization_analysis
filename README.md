This repository is a collection of code used to measure and model the Mueller matrix of a Thorlabs dichroic mirror, DMLP550. The polarization_raytracing directory contains files required to perform a polarization raytrace of a model of a dichroic mirror, based on a list of layer heights and refractive indices from Optilayer. The gromit directory contains files for running a dual rotating retarder polarimeter (DRRP) named Gromit with Zaber rotation stages, as well as code to calibrate the DRRP, and to do the data reduction from a set of intensity measurements to a Mueller and Jones pupil, or a averaged Mueller/Jones matrix with propagated error.

This code is dependent upon several python packages to run:
1. katsu, by Jaren Ashcraft: https://github.com/Jashcraf/katsu/tree/main
2. genpolab, by Ramya Anche: https://github.com/ramya-anche/genpolab
3. zosapi, for interfacing with Zemax: https://github.com/x68507/zosapi

In addition, the function trace_through_zos() in zemax_raytrace.py was taken from Jaren Ashcraft's Poke: https://github.com/Jashcraf/poke


Note: an addition to the full_mueller_polarimetry() function in katsu.poalrimetry was made to include error propagation;
this is not included in the base katsu package, so a printout is included in mueller_pup_air.py for reference.
