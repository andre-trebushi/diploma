#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:41:19 2021

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

E_ph = 300 # eV
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

N_b = 100 #number of statistical realizations
N_e = 100 #number of macro electrons 
Nz, Ny, Nx = N_b, 101, 101 # the shape of the dfl.fld

# Nz, Ny, Nx = N_b, 100, 100 # the shape of the dfl.fld
# seed=1
filePath = '/home/andrei'

e_beam_param = r'$N_x$ = {}, '.format(round((ebeam_sigma_x)**2/xlamds/L_w, 3)) + r'$N_y$ = {}, '.format(round((ebeam_sigma_y)**2/xlamds/L_w, 3)) + \
                r'$D_x$ = {}, '.format(round((ebeam_sigma_xp)**2 * L_w/xlamds, 3)) + r'$D_y$ = {}, '.format(round((ebeam_sigma_yp)**2 * L_w/xlamds, 3)) + \
                r'$N_b$ = {} '.format(N_b) + r'$N_e = {}$'.format(N_e)
print(e_beam_param)

# Monte Calro
Lz, Ly, Lx = 1000e-6, 6e-3, 6e-3#27.37e-3 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

### creating RadiationField object
dfl = RadiationField((Nz, Ny, Nx))
dfl.dx, dfl.dy, dfl.dz = dx, dy, dz
dfl.xlamds = xlamds
dfl.filePath = filePath
dfl.to_domain('sf')

### creating RadiationField object
dfl1 = RadiationField((Nz, Ny, Nx))
dfl1.dx, dfl1.dy, dfl1.dz = dx, dy, dz
dfl1.xlamds = xlamds
dfl1.filePath = filePath
dfl1.to_domain('sf')

fieldname_MC = ''
approximation = "far_field"
# approximation = "near_field"

ebeam_sigma_x = 38e-06
ebeam_sigma_y = 4.68e-06
ebeam_sigma_xp = 0.01e-06
ebeam_sigma_yp = 0.01e-06
dfl_coh = undulator_field_dfl_MP(dfl1, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')

ebeam_sigma_x = 100e-6#38e-06
ebeam_sigma_y = 100e-6#4.68e-06
ebeam_sigma_xp = 0.01e-06
ebeam_sigma_yp = 0.01e-06
dfl_incoh = undulator_field_dfl_MP(dfl, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')

filePath = '/home/andrei/Documents/diploma/Diploma/images/'

plot_dfls(dfl_incoh, dfl_coh, domains='s', scale='mm', label_first="некогерентное", label_second="когерентное",
              slice_xy=False, fig_name='diff_divergence_incoh', filePath=filePath, savefig=True)















