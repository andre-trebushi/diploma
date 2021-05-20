#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:55:17 2021

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
ebeam_sigma_x = 1e-06
ebeam_sigma_y = 1e-06
ebeam_sigma_xp = 20e-06
ebeam_sigma_yp = 20e-06
# ebeam_sigma_xp = 10e-06
# ebeam_sigma_yp = 10e-06

ebeam_sigma_z = 2000e-6
ebeam_sigma_gamma = 1e-4 #TODO: relative electron energy spread

N_b = 200 #number of statistical realizations
N_e = 200 #number of macro electrons 
Nz, Ny, Nx = N_b, 151, 151 # the shape of the dfl.fld

# Nz, Ny, Nx = N_b, 100, 100 # the shape of the dfl.fld
# seed=1
# filePath = '/home/andrei'

e_beam_param = r'$N_x$ = {}, '.format(round((ebeam_sigma_x)**2/xlamds/L_w, 3)) + r'$N_y$ = {}, '.format(round((ebeam_sigma_y)**2/xlamds/L_w, 3)) + \
                r'$D_x$ = {}, '.format(round((ebeam_sigma_xp)**2 * L_w/xlamds, 3)) + r'$D_y$ = {}, '.format(round((ebeam_sigma_yp)**2 * L_w/xlamds, 3)) + \
                r'$N_b$ = {} '.format(N_b) + r'$N_e = {}$'.format(N_e)
print(e_beam_param)


str_simulation_param = 'ebeam_sigma_x = {}\n'.format(ebeam_sigma_x) + \
                       'ebeam_sigma_y = {}\n'.format(ebeam_sigma_y) + \
                       'ebeam_sigma_xp = {}\n'.format(ebeam_sigma_xp) + \
                       'ebeam_sigma_yp = {}\n'.format(ebeam_sigma_yp) + \
                       'N_b = {}\n'.format(N_b) + \
                       'N_e = {}\n'.format(N_e) + \
                       'grid mesh x = {}\n'.format(Nx) + 'grid mesh y = {}\n'.format(Ny) 

script_name = os.path.basename(__file__)
#%% 
simulation_name = os.path.basename(__file__) + str_simulation_param#/'diff_coh_incoh_rad'
### make a directory on your machine        
###saving simulation parameters in a .txt file
filePath = '/home/andrei/Documents/diploma/code/' + simulation_name + '/'
os.makedirs(filePath, exist_ok=True)
f = open(filePath + 'prm.txt', "w")
f.write(str_simulation_param)
f.close()


script_dir = os.getcwd() + '/' + script_name
new_script_dir = filePath + script_name
### seed for comparing fields
seed = 1234
#%%
###
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

dfl_coh = undulator_field_dfl_MP(dfl1, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='coh')

dfl_incoh = undulator_field_dfl_MP(dfl, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
                                            approximation=approximation, mode='incoh')


#%%
from ocelot.common.math_op import *  # import of mathematical functions like gauss_fit

def squarify(fig):
    w, h = fig.get_size_inches()
    if w > h:
        t = fig.subplotpars.top
        b = fig.subplotpars.bottom
        axs = h*(t-b)
        l = (1.-axs/w)/2
        fig.subplots_adjust(left=l, right=1-l)
    else:
        t = fig.subplotpars.right
        b = fig.subplotpars.left
        axs = w*(t-b)
        l = (1.-axs/h)/2
        fig.subplots_adjust(bottom=l, top=1-l)

dfl_coh.to_domain('sf')
dfl_incoh.to_domain('sf')

fig, ax = plt.subplots()
fig.set_size_inches(6, 5, forward=True)

# ax = fig.add_subplot(1, 1, 1)
x = dfl_coh.scale_y()*1e6/25
I = np.sum(dfl_coh.intensity(), axis=0)[:, dfl_coh.Nx()//2+1]
x_line_f, rms_coh = gauss_fit(x, I)
print('coh = {}'.format(rms_coh))
ax.plot(x, I, color='green', label=r'когерентное, $\sigma_r$ = {} мкрад'.format(round(rms_coh,1)))
ax.set_ylim(0, 1.15*np.max(I))


ax1 = ax.twinx()  

x = dfl_incoh.scale_y()*1e6/25
I = np.sum(dfl_incoh.intensity(), axis=0)[:, dfl_incoh.Nx()//2+1]
x_line_f, rms_incoh = gauss_fit(x, I)
print('incoh = {}'.format(rms_incoh))
ax1.plot(x, I, color='blue', label=r'некогерентное, $\sigma_r$ = {} мкрад'.format(round(rms_incoh,1)))

print(rms_incoh/rms_coh)    
ax.grid()
ax.set_xlim(np.min(dfl_coh.scale_y()*1e6/25), np.max(dfl_coh.scale_y()*1e6/25))
ax1.set_ylim(0, 1.15*np.max(I))

ax.set_xlabel(r'$\theta$, мкрад', fontsize=16)
ax1.set_xlabel(r'$\theta$, мкрад', fontsize=16)
ax.set_ylabel('пр.е', fontsize=16)
ax1.set_ylabel('пр.е', fontsize=16)
plt.tight_layout()

ax.legend(fontsize=12, bbox_to_anchor=(0, 0.91), loc='upper left')
ax1.legend(fontsize=12, bbox_to_anchor=(0, 1.01), loc='upper left')

squarify(fig)
fig.canvas.mpl_connect("resize_event", lambda evt: squarify(fig))

plt.show()
plt.savefig(filePath + 'diff_coh_incoh_rad' + '.png', dpi=200)

filePath = '/home/andrei/Documents/diploma/Diploma/images/'

# plot_two_dfls(dfl_incoh, dfl_coh, domains='s', scale='mm', label_first="некогерентное", label_second="когерентное",
#               slice_xy=True, fig_name='diff_coh_incoh_rad', filePath=filePath, savefig=False)
# plot_dfl(dfl_coh, domains='kf', phase=True, fig_name = fieldname_MC, cbar=True)
# plot_dfl(dfl_incoh, domains='kf', phase=True, fig_name = fieldname_MC)










