#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:06:43 2021

@author: andrei
"""

import numpy as np
from drawing_routine import *
from ocelot.optics.wave import *

# ocelog.setLevel(logging.INFO)
_logger = logging.getLogger(__name__)
#%%
n_s = 200
l_w = 0.018 # [m] undulator period 
L_w = l_w * n_s

E_ph = 12.40 # eV
w = E_ph / hr_eV_s 
xlamds = 2 * np.pi * speed_of_light / w

sigma_r = np.sqrt(2*xlamds*L_w)/4/np.pi #natural radiation size in the waist
sigma_rp = np.sqrt(xlamds/2/L_w) #natural radiation divergence at the waist


#### #1
# ebeam_sigma_x = 1.5e-05
# ebeam_sigma_y = 5e-06
# ebeam_sigma_xp = 5e-07
# ebeam_sigma_yp = 7e-06
#### #2
# ebeam_sigma_x = 50e-6
# ebeam_sigma_y = 10e-6
# ebeam_sigma_xp = 10e-06
# ebeam_sigma_yp = 50e-06
#### #3
ebeam_sigma_x = 0.1e-6#38e-06
ebeam_sigma_y = 0.1e-6#4.68e-06
ebeam_sigma_xp = 25e-06
ebeam_sigma_yp = 20e-06

ebeam_sigma_z = 2000e-6
ebeam_sigma_gamma = 1e-4 #TODO: relative electron energy spread

N_b = 200 #number of statistical realizations
N_e = 200 #number of macro electrons 
Nz, Ny, Nx = N_b, 101, 101 # the shape of the dfl.fld

# Nz, Ny, Nx = N_b, 100, 100 # the shape of the dfl.fld
# seed=1
filePath = '/home/andrei'

e_beam_param = r'$N_x$ = {}, '.format(round((ebeam_sigma_x)**2/xlamds/L_w, 3)) + r'$N_y$ = {}, '.format(round((ebeam_sigma_y)**2/xlamds/L_w, 3)) + \
                r'$D_x$ = {}, '.format(round((ebeam_sigma_xp)**2 * L_w/xlamds, 3)) + r'$D_y$ = {}, '.format(round((ebeam_sigma_yp)**2 * L_w/xlamds, 3)) + \
                r'$N_b$ = {} '.format(N_b) + r'$N_e = {}$'.format(N_e)
print(e_beam_param)

# Monte Calro
Lz, Ly, Lx = 1000e-6, 20e-3, 20e-3#27.37e-3 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

### creating RadiationField object
dfl = RadiationField((Nz, Ny, Nx))
dfl.dx, dfl.dy, dfl.dz = dx, dy, dz
dfl.xlamds = xlamds
dfl.filePath = filePath
dfl.to_domain('sf')

### creating RadiationField object


fieldname_MC = ''
approximation = "far_field"
# approximation = "near_field"

ebeam_sigma_x = 1e-06
ebeam_sigma_y = 1e-06
ebeam_sigma_xp = 0.01e-06
ebeam_sigma_yp = 0.01e-06
dfl_0 = deepcopy(dfl)
dfl_0 = undulator_field_dfl_MP(dfl_0, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')

# plot_dfl(dfl_0, domains='s')

ebeam_sigma_x = 100e-6#38e-06
ebeam_sigma_y = 100e-6#4.68e-06
ebeam_sigma_xp = 0.01e-06
ebeam_sigma_yp = 0.01e-06
dfl_20 = deepcopy(dfl)
dfl_20 = undulator_field_dfl_MP(dfl_20, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')

ebeam_sigma_x = 200e-6#38e-06
ebeam_sigma_y = 200e-6#4.68e-06
ebeam_sigma_xp = 0.01e-06
ebeam_sigma_yp = 0.01e-06
dfl_40 = deepcopy(dfl)
dfl_40= undulator_field_dfl_MP(dfl_40, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')

ebeam_sigma_x = 400e-6#38e-06
ebeam_sigma_y = 400e-6#4.68e-06
ebeam_sigma_xp = 0.01e-06
ebeam_sigma_yp = 0.01e-06
dfl_60 = deepcopy(dfl)
dfl_60 = undulator_field_dfl_MP(dfl_60, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')
#%%
filePath = '/home/andrei/Documents/diploma/Diploma/images/'

fig, ax = plt.subplots()
fig.set_size_inches(6, 5, forward=True)

# ax = fig.add_subplot(1, 1, 1)
x = dfl_0.scale_y()*1e6/25
I_0 = np.sum(dfl_0.intensity(), axis=0)[:, dfl_0.Nx()//2+1]
I_20 = np.sum(dfl_20.intensity(), axis=0)[:, dfl_20.Nx()//2+1]
I_40 = np.sum(dfl_40.intensity(), axis=0)[:, dfl_40.Nx()//2+1]
I_60 = np.sum(dfl_60.intensity(), axis=0)[:, dfl_60.Nx()//2+1]

I_0 = I_0 / np.max(I_0)
I_20 = I_20 / np.max(I_20)
I_40 = I_40 / np.max(I_40)
I_60 = I_60 / np.max(I_60)

plt.plot(x, I_0, color = 'blue', label = 'дифр. огр.')
plt.plot(x, I_20, color = 'green', label = '100 мкм')
plt.plot(x, I_40, color = 'orange', label = '200 мкм')
plt.plot(x, I_60, color = 'red', label = '400 мкм')

plt.ylim(0)
plt.xlim(np.min(x), np.max(x))
ax.set_xlabel(r'$\theta$, мкрад', fontsize=16)
ax.set_ylabel('пр.е', fontsize=16)
plt.tight_layout()
plt.grid()
plt.legend(fontsize=12, bbox_to_anchor=(0, 0.99), loc='upper left')
plt.show()
filePath = '/home/andrei/Documents/diploma/Diploma/images/'

plt.savefig(filePath + 'diff_divergence_coh' + '.png', dpi=200)


# plot_dfls([dfl_incoh, dfl_coh], domains='s', scale='mm',
#               slice_xy=True, fig_name='diff_divergence_coh', filePath=filePath, savefig=True)
















