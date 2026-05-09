'''
Dichroic mirror poalrization analysis using ploarization raytracing using Zemax for raytrace data
and genpolab for Jones pupil calculation. Relevant functions can be added to index.py as needed for 
other coating materials. Both zemax and genpolab expect the dochroic layers to be oriented front to back.
This runs a Zemax raytrace for each wavelength, only runs for one AOI and one beam (reflected/transmitted) at a time.
'''

import numpy as np
from genpolab import polab as pol
from genpolab import genfig as gen
import polarization_raytracing.zemax_raytrace as zemax_raytrace
import polarization_raytracing.index as index

# functions for wavelength dependent refractive indices
H=index.n_tio2_cauchy
L=index.n_mgf2_cauchy
A=index.n_al2o3_sellmeier3
B=index.n_hfo2_sellmeier
D=index.n_fused_silica_sellmeier

# wavelengths in microns
ws = [0.4,0.405,0.41,0.415,0.42,0.425,0.43,0.435,0.44,0.445,0.45,0.455,0.46,0.465,0.47,0.475,0.48,0.485,0.49,0.495,0.5,0.505,0.51,0.515,0.52,0.525,0.53,0.535,0.54,0.545,0.55,0.555,0.56,0.565,0.57,0.575,0.58,0.585,0.59,0.595,0.6,0.605,0.61,0.615,0.62,0.625,0.63,0.635,0.64,0.645,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7]

# dichroic layer thickness in meters, subsstrate -> air
d_layers_dichroic = [3.1356900000000005e-07, 9.722000000000001e-09, 5.9528e-08, 1.5288e-08, 6.0022e-08, 1.9367e-08, 8.8033e-08, 2.2352700000000001e-07, 1.7353000000000003e-08, 2.7948000000000003e-08, 1.01716e-07, 2.1206e-08, 2.9711e-08, 4.5159e-08, 1.7064e-08, 6.3859e-08, 2.3839e-08, 5.5419e-08, 8.0283e-08, 3.294400000000001e-08, 2.2931000000000002e-08, 5.5188000000000004e-08, 1.9695000000000002e-08, 6.2366e-08, 3.5831000000000005e-08, 2.3058e-08, 5.1664000000000005e-08, 3.9708e-08, 3.2002000000000004e-08, 4.8063000000000005e-08, 2.0965000000000003e-08, 1.6198e-08, 5.6241e-08, 4.4428e-08, 2.8933000000000002e-08, 7.1444e-08, 2.9965000000000004e-08, 1.5495e-08, 6.6603e-08, 1.8690000000000004e-08, 5.877600000000001e-08, 9.833400000000001e-08, 8.2877e-08, 5.981400000000001e-08, 8.576000000000002e-09, 4.7315000000000004e-08, 6.934900000000001e-08, 2.1045000000000004e-08, 4.5371e-08, 7.1838e-08, 3.0357e-08, 2.5035000000000002e-08, 8.403400000000001e-08, 7.132500000000001e-08, 2.5954000000000002e-08, 8.396000000000001e-09, 7.635400000000001e-08, 1.3041e-08, 6.414700000000001e-08, 5.083100000000001e-08, 2.5812000000000003e-08, 4.4293000000000005e-08, 2.6380000000000002e-08, 3.3469e-08, 3.1711e-08, 8.249800000000002e-08, 2.2515000000000003e-08, 4.3463000000000003e-08, 4.040400000000001e-08, 4.0689e-08, 4.446e-08, 3.2206000000000004e-08, 2.0275e-08, 6.794e-08, 2.3507000000000003e-08, 2.3003e-08, 8.842300000000001e-08, 6.241500000000001e-08, 6.4221e-08, 2.8598000000000002e-08, 1.3048e-07, 3.5526e-08, 7.4803e-08, 5.5961e-08, 5.8407e-08, 3.4530000000000005e-08, 1.25175e-07, 5.1751e-08, 1.0592000000000001e-08, 6.956000000000001e-08, 3.7911e-08, 8.143899999999999e-08, 6.2294e-08, 3.0935e-08, 1.3751e-08, 2.4120000000000004e-08, 2.5788000000000002e-08, 1.32135e-07, 7.000700000000001e-08, 6.4934e-08, 5.8342e-08, 5.8733e-08, 1.2785000000000002e-08, 4.6098e-08, 4.6748e-08, 4.8394e-08, 6.036200000000001e-08, 5.5031000000000005e-08, 6.1193e-08, 8.3563e-08, 8.052e-09, 8.4086e-08, 5.4818e-08, 5.4684e-08, 8.8322e-08, 5.6119e-08, 8.081100000000001e-08, 4.891e-08, 8.2827e-08, 7.777e-08, 3.9768e-08, 5.802300000000001e-08, 1.55921e-07, 2.6781e-08, 7.111100000000001e-08, 6.943300000000001e-08, 9.3379e-08, 3.8933e-08, 7.435800000000001e-08, 5.7034e-08, 2.86074e-07, 1.5009000000000002e-08, 6.35e-08, 2.0331000000000002e-08, 1.6627e-08, 1.25729e-07]
d_layers_dichroic.reverse() # correct raytracing orientation is air -> substrate
# backside anti-reflective coating layer thickness in meters, substrate -> air
ar_d_layers = [4.4629e-08, 9.534000000000001e-09, 1.90025e-07, 7.4079e-08, 1.60396e-07, 6.734e-09, 9.4987e-08]
ar_d_layers.reverse()

# build list of coatings, one for each wavelength
coats = []
for w in ws:
    # list of refractive indices for each layer, substrate -> air
    n_layers_dichroic=[D(w), H(w), D(w), H(w), B(w), H(w), B(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), B(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), L(w), H(w), L(w), H(w), L(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), B(w), D(w), H(w), D(w), B(w), L(w), H(w), B(w), L(w), H(w), B(w), H(w), A(w), H(w), A(w), H(w), L(w), A(w), H(w), L(w), H(w), L(w), H(w), L(w), H(w), L(w), B(w), L(w), H(w), L(w), H(w), A(w), L(w), H(w), A(w), H(w), A(w), H(w), L(w), B(w), L(w), H(w), A(w), H(w), B(w), H(w), B(w), A(w), H(w), A(w), H(w), L(w), H(w), L(w), H(w), L(w), A(w), L(w), H(w), A(w), L(w), H(w), A(w), L(w), H(w), A(w), H(w), B(w), L(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), D(w), H(w), B(w), L(w), H(w), L(w)]
    n_layers_dichroic.reverse() # correct orientation
    ar_n_layers = [A(w), H(w), A(w), L(w), B(w), H(w), L(w)] # AR coating refractive indices
    ar_n_layers.reverse()
    di_coat = []
    for i in range(len(d_layers_dichroic)):
        di_coat.append( (n_layers_dichroic[i], d_layers_dichroic[i]) )
    di_coat.append((1.5, 5e-3)) # 5mm substrate with n=1.5
    for i in range(len(ar_d_layers)):
        di_coat.append( (ar_n_layers[i], ar_d_layers[i]) )
    di_coat.append(1.0) # end in air
    coats.append(di_coat)
# set up list of surfaces to raytrace through, one for each wavelength.
surfs = []
for i in range(len(ws)):
    surf = {
            'surf':3,
            'coating':coats[i],
            'mode':'reflect',
        }
    surfs.append(surf)
# raytracing parameters
nrays=1 #square grid of rays of nrays x nrays
pupil_radius=20e-3
max_fov=1e-3
fov=[0,0]
normalized_fov = np.array(fov) / max_fov
# Generate a square grid in the pupil
x = np.linspace(-pupil_radius, pupil_radius, nrays)
y = np.linspace(-pupil_radius, pupil_radius, nrays)
X, Y = np.meshgrid(x, y)

# Flatten and normalize coordinates
x_flat = np.ravel(X) / pupil_radius
y_flat = np.ravel(Y) / pupil_radius

# Construct base rays array: [x, y, l, m] for each ray
base_rays = np.array([
    x_flat,
    y_flat,
    0 * x_flat + normalized_fov[0],  # field x
    0 * y_flat + normalized_fov[1]   # field y
])
# main loop to collect polarization data for each wavelength
raysets = [base_rays]
ps = []
ss = []
thss = []
thps = []
rets = []
dias = []
for i in range(len(ws)):
    # raytrace through Zemax
    pos, dir, norm, opd, mask = zemax_raytrace.trace_through_zos(raysets, r"C:\Users\nocoo\Downloads\Tilted_mirror.zmx", [surfs[i]], nrays, 1, True)
    rayset_index = 0
    surface_index = 0 # only one surface in surfs, so index is always 0. can theoretically be set to other values, but untested and may cause issues if surfs is changed to have more than one surface.
    w = ws[i]
    # remake list of refractive indices for each layer for dichroic stack and AR coating
    n_layers_dichroic=[D(w), H(w), D(w), H(w), B(w), H(w), B(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), B(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), H(w), D(w), L(w), H(w), L(w), H(w), L(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), B(w), D(w), H(w), D(w), B(w), L(w), H(w), B(w), L(w), H(w), B(w), H(w), A(w), H(w), A(w), H(w), L(w), A(w), H(w), L(w), H(w), L(w), H(w), L(w), H(w), L(w), B(w), L(w), H(w), L(w), H(w), A(w), L(w), H(w), A(w), H(w), A(w), H(w), L(w), B(w), L(w), H(w), A(w), H(w), B(w), H(w), B(w), A(w), H(w), A(w), H(w), L(w), H(w), L(w), H(w), L(w), A(w), L(w), H(w), A(w), L(w), H(w), A(w), L(w), H(w), A(w), H(w), B(w), L(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), A(w), H(w), D(w), H(w), B(w), L(w), H(w), L(w)]
    n_layers_dichroic.reverse()
    ar_n_layers = [A(w), H(w), A(w), L(w), B(w), H(w), L(w)]
    ar_n_layers.reverse()
    n_rays = pos[0].shape[2]  # number of rays

    sur_list = []

    for ray_ind in range(n_rays): # extract ray data for each ray
        lx = dir[0][rayset_index, surface_index, ray_ind]
        ly = dir[1][rayset_index, surface_index, ray_ind]
        lz = dir[2][rayset_index, surface_index, ray_ind]
        nx = norm[0][rayset_index, surface_index, ray_ind]
        ny = norm[1][rayset_index, surface_index, ray_ind]
        nz = norm[2][rayset_index, surface_index, ray_ind]
        ray_dict = {
            'X-cor': pos[0][rayset_index, surface_index, ray_ind],
            'Y-cor': pos[1][rayset_index, surface_index, ray_ind],
            'Z-cor': pos[2][rayset_index, surface_index, ray_ind],
            'Ray-Stat': 0 if mask[rayset_index, surface_index, ray_ind] != 0 else 1,
            'ref-x': lx,
            'ref-y': -ly,
            'ref-z': lz,
            'nor-x': nx,
            'nor-y': ny,
            'nor-z': nz,
            'Inc-ang': np.degrees(np.arccos(np.abs(lx*nx + ly*ny + lz*nz))) 
        }
        sur_list.append(ray_dict)


    # Double pole coordinate axes
    aloc= np.array([0,1,0]) # exitant chief ray direction cosine
    xin=np.array([1,0,0]) # local x axis of input ray
    xout=np.array([1,0,0]) # local x axis of output ray
    # polarization anaylsis with Fresnel coefficients from raytraced data
    (vignetted_list,raytrace_list,PRM_list,O_e_list,O_x_list)= pol.calc_prt(sur_list, n_rays, aloc, xin, xout, n_layers=n_layers_dichroic, d_layers=d_layers_dichroic, ar_n_layers=ar_n_layers, ar_d_layers=ar_d_layers, wav=ws[i])
    jones_list = pol.calc_jones(PRM_list, O_e_list, O_x_list) # list of 3x3 jones matrices
    j_list = []
    for J3 in jones_list:
        J2 = J3[0:2, 0:2] # select xy Jones matrix
        j_list.append(J2)
    mean_jones = np.mean(j_list, axis=0)
    print(mean_jones)
    s = mean_jones[0,0]
    p = mean_jones[1,1]
    ths = np.angle(s)
    thp = np.angle(p)

    ss.append(np.abs(s))
    ps.append(np.abs(p))
    thss.append(ths)
    thps.append(thp)
    rets.append(np.abs(ths - thp))
    diats = np.abs(ss[-1] **2 - ps[-1] **2)/ (np.abs(ss[-1])**2 + np.abs(ps[-1])**2)
    dias.append(diats)
import matplotlib.pyplot as plt
      

plt.figure()
plt.plot(ws, np.array(ss) **2, label='s-pol')
plt.plot(ws, np.array(ps) **2, label='p-pol')

plt.xlabel('Wavelength (um)')
plt.ylabel('Rp, Rs')
plt.legend()
plt.title('Dichroic Coating Reflectance')


plt.show()

for i in range(len(rets)):
    print("retardance at", ws[i], "um:", rets[i])
    print("diattenuation at", ws[i], "um:", dias[i])
