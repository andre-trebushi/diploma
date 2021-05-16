#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:18:55 2021

@author: andrei
"""

import numpy as np
import matplotlib.pyplot as plt
# from analytics_to_OCELOT import *
from ocelot.optics.wave import *
from ocelot.gui.dfl_plot import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
import scipy.special as sc
from scipy import signal
from scipy import misc
import copy
# from drawing_routine import *

ocelog.setLevel(logging.INFO)
_logger = logging.getLogger(__name__)

def undulator_field_dfl_SERVAL(dfl, L_w, sig_x=0, sig_y=0, sig_xp=0, sig_yp=0, k_support = 'intensity', s_support='conv_intensities', showfig=False, seed=None):
    filePath = '/home/andrei/Documents/diploma/Diploma/images/'
    _logger.info('Generating undulator field with Serval algorithm')
    w_0 = 2*np.pi * speed_of_light / dfl.xlamds
    
    if showfig:
        # plot_dfl_2Dsf(dfl, scale='um', domains='sf', savefig=True, 
        #               fig_name = '1-X_noise', filePath=filePath)
        plot_dfl(dfl, line_off_xy = False, fig_name = '1-X_noise')
    
    dfl.to_domain('sf')
    
    x, y = np.meshgrid(dfl.scale_x(), dfl.scale_y())#, indexing='ij')
    
    mask_xy_ebeam = np.exp(- x**2 / 4 / sig_x**2 - y**2 / 4 / sig_y**2) # 4 because amplitude, not intensity
    # mask_xy_ebeam = np.exp(- x**2 / 2 / sig_x**2 - y**2 / 2 / sig_y**2) # 4 because amplitude, not intensity

    mask_xy_ebeam /= np.sum(mask_xy_ebeam)
    
    # mask_xy_radiation = np.sqrt((1j*(np.pi - 2*special.sici(w_0*(x**2 + y**2)/speed_of_light/L_w)[0]))**2)
    # mask_xy_radiation = 1j*(np.pi - 2*special.sici(w_0*(x**2 + y**2)/speed_of_light/L_w)[0])

    # mask_xy_radiation = (1j*(np.pi - 2*special.sici(w_0*(x**2 + y**2)/speed_of_light/L_w)[0]))**2
    if s_support == 'conv_intensities':
        _logger.info(ind_str +'s_support == "conv"')
        mask_xy_radiation = 1j*(np.pi - 2*scipy.special.sici(w_0*(x**2 + y**2)/speed_of_light/L_w)[0])
        mask_xy = scipy.signal.fftconvolve(mask_xy_radiation**2, mask_xy_ebeam**2, mode='same')
        mask_xy = np.sqrt(mask_xy)
    elif s_support == 'conv_amplitudes':
        _logger.info(ind_str +'s_support == "conv"')
        mask_xy_radiation = 1j*(np.pi - 2*scipy.special.sici(w_0*(x**2 + y**2)/speed_of_light/L_w)[0])
        mask_xy = scipy.signal.fftconvolve(mask_xy_radiation, mask_xy_ebeam, mode='same')
    else:
        _logger.info(ind_str +'s_support == "beam"')
        mask_xy = mask_xy_ebeam
    
    _logger.info(ind_str +'Multiplying by real space mask')
    dfl.fld *= mask_xy
    # dfl.fld *= np.sqrt(mask_xy)
    _logger.info(2*ind_str +'done')

    if showfig:
        plot_dfl(dfl, domains='s', line_off_xy = False, fig_name = '2-X_e-beam-size')
        # plot_dfl_2Dsf(dfl, scale='um', domains='sf', savefig=True,
        #               fig_name = '2-X_e-beam-size', filePath=filePath)
        plot_dfl(dfl, domains='k', line_off_xy = False, fig_name = '2-X_e-beam-size')
        # plot_dfl_2Dsf(dfl, scale='um', domains='kf', savefig=True,
        #               fig_name = '2-X_e-beam-divergence', filePath=filePath)
                
    dfl.to_domain('kf')

    k_x, k_y = np.meshgrid(dfl.scale_x(), dfl.scale_y())
    mask_kxky_ebeam = np.exp(-k_y**2 / 4 / sig_yp**2 - k_x**2 / 4 / sig_xp**2 ) # 4 because amplitude, not intensity
    # mask_kxky_ebeam = np.exp(-k_y**2 / 2 / sig_yp**2 - k_x**2 / 2 / sig_xp**2 ) # 2 because intensity
    mask_kxky_ebeam /= np.sum(mask_kxky_ebeam)
    
    # mask_kxky_radiation = np.sqrt((np.sinc(w_0 * L_w * (k_x**2 + k_y**2) / 4 / speed_of_light / np.pi))**2)# Geloni2018 Eq.3, domega/omega = 2dgamma/gamma, divided by pi due to np.sinc definition
    # mask_kxky_radiation = (np.sinc(w_0 * L_w * (k_x**2 + k_y**2) / 4 / speed_of_light / np.pi))# Geloni2018 Eq.3, domega/omega = 2dgamma/gamma, divided by pi due to np.sinc definition
        
    mask_kxky_radiation = np.sinc(w_0 * L_w * (k_x**2 + k_y**2) / 4 / speed_of_light / np.pi)# Geloni2018 Eq.3, domega/omega = 2dgamma/gamma, divided by pi due to np.sinc definition

    if k_support == 'intensity':
        _logger.info(ind_str +'k_support == "intensity"')
        mask_kxky = scipy.signal.fftconvolve(mask_kxky_ebeam**2, mask_kxky_radiation**2, mode='same')
        mask_kxky = np.sqrt(mask_kxky[np.newaxis, :, :])
        mask_kxky /= np.sum(mask_kxky)
    elif k_support == 'amplitude':
        _logger.info(ind_str +'k_support == "amplitude"')
        mask_kxky = scipy.signal.fftconvolve(mask_kxky_ebeam, mask_kxky_radiation, mode='same')
        mask_kxky /= np.sum(mask_kxky)
    else:
        raise ValueError('k_support should be either "intensity" or "amplitude"')
    
    # dfl.fld *= mask_kxky[np.newaxis, :, :]
    _logger.info(ind_str +'Multiplying by inverse space mask')
    dfl.fld *= mask_kxky
    _logger.info(2*ind_str +'done')

    if showfig:
        plot_dfl(dfl, domains='s', fig_name = '3-X_radaition_size')
        # plot_dfl_2Dsf(dfl, scale='um', domains='sf', savefig=True, 
        #               fig_name = '3-X_radaition_size', filePath=filePath)
        plot_dfl(dfl, domains='k', fig_name = '3-X_radiation_divergence')
        # plot_dfl_2Dsf(dfl, scale='um', domains='kf', savefig=True,
        #               fig_name = '3-X_radaition_divergence', filePath=filePath)
    return dfl 

n_s = 200
l_w = 0.018 # [m] undulator period 
L_w = l_w * n_s

E_ph = 2167 # eV
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
# ebeam_sigma_x = 0.0001
# ebeam_sigma_y = 2e-05
# ebeam_sigma_xp = 2.5e-06
# ebeam_sigma_yp = 2.5e-05
#### #3
# ebeam_sigma_x = 100e-06
# ebeam_sigma_y = 200e-06
# ebeam_sigma_xp = 50e-06
# ebeam_sigma_yp = 25e-06
#### #4
ebeam_sigma_x = 38e-06
ebeam_sigma_y = 4.68e-06
ebeam_sigma_xp = 25e-06
ebeam_sigma_yp = 20e-06

ebeam_sigma_z = 2000e-6
ebeam_sigma_gamma = 1e-4 #TODO: relative electron energy spread

N_b = 400 #number of statistical realizations
N_e = 100 #number of macro electrons 
Nz, Ny, Nx = N_b, 551, 551 # the shape of the dfl.fld

str_simulation_param = 'ebeam_sigma_x = {}\n'.format(ebeam_sigma_x) + \
                       'ebeam_sigma_y = {}\n'.format(ebeam_sigma_y) + \
                       'ebeam_sigma_xp = {}\n'.format(ebeam_sigma_xp) + \
                       'ebeam_sigma_yp = {}\n'.format(ebeam_sigma_yp) + \
                       'N_b = {}\n'.format(N_b) + \
                       'N_e = {}\n'.format(N_e) + \
                       'grid mesh x = {}\n'.format(Nx) + 'grid mesh y = {}\n'.format(Ny) 

script_name = os.path.basename(__file__)
simulation_name = "{:.2E}".format(ebeam_sigma_x) + '_um_' + \
                  "{:.2E}".format(ebeam_sigma_y) + '_um_' + \
                  "{:.2E}".format(ebeam_sigma_xp) + '_urad_' + \
                  "{:.2E}".format(ebeam_sigma_yp) + '_urad_' + str(script_name.split('.')[0])

e_beam_param = r'N_x = {}, '.format(round((ebeam_sigma_x)**2/(xlamds*L_w), 5)) + r'N_y = {}, '.format(round((ebeam_sigma_y)**2/(xlamds*L_w), 5)) + \
               r'D_x = {}, '.format(round((ebeam_sigma_xp)**2 / (xlamds/L_w), 5)) + r'D_y = {}, '.format(round((ebeam_sigma_yp)**2 / (xlamds/L_w), 5)) + \
               r'N_b = {} '.format(N_b) + r'N_e = {}$'.format(N_e)
print(e_beam_param)


### make a directory on your machine        
###saving simulation parameters in a .txt file
filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/'
os.makedirs(filePath, exist_ok=True)
f = open(filePath + 'prm.txt', "w")
f.write(str_simulation_param)
f.close()

script_dir = os.getcwd() + '/' + script_name
new_script_dir = filePath + script_name
### seed for comparing fields
seed = 1234
###
#%%

# Define mesh size
Lz, Ly, Lx = 10000e-6, 300e-6, 800e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# create radiation field
dfl_SERVAL = RadiationField((N_b, Ny, Nx))  
dfl_SERVAL.dx, dfl_SERVAL.dy, dfl_SERVAL.dz = dx, dy, dz
dfl_SERVAL.xlamds = xlamds # SVEA carrieer frequency

dfl_SERVAL.fld = np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) + 1j * np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) # Gaussian noise

dfl_SERVAL.filePath = filePath+'.dfl'
# dfl_omega_0 = 2*np.pi * speed_of_light / dfl.xlamds # set undulator resonance to SVEA carrier frequency (to the middle of the photon energy domain mesh)
# radiation_omega_resonance = dfl_omega_0 * 1.01 # shift undulator resonance to the right

fieldname_SERVAL = '0-source_SERVAL'
dfl_SERVAL = undulator_field_dfl_SERVAL(dfl_SERVAL, L_w=L_w, 
                                        sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, 
                                        sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp,
                                        k_support = 'intensity', s_support='conv_intensities', showfig=False)

# plot_dfl(dfl_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)
# plot_dfl(dfl_SERVAL, domains='kf', phase=True, fig_name = fieldname_SERVAL)

# #%%
# Lz, Ly, Lx = 100000e-6, 6000e-6, 6000e-6 #size of realspace grid [m]
# Nz, Ny, Nx = N_b, 151, 151 # the shape of the dfl.fld
# dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# ### creating RadiationField object
# dfl_MC = RadiationField((Nz, Ny, Nx))
# dfl_MC.dx, dfl_MC.dy, dfl_MC.dz = dx, dy, dz
# dfl_MC.xlamds = xlamds
# dfl_MC.filePath = filePath
# dfl_MC.to_domain('sf')

# fieldname_MC = '1-far_zone_50_m_MC'
# # approximation = "near_field"
# approximation = "far_field"

# dfl_MC = undulator_field_dfl_MP(dfl_MC, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
#                                             sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, 
#                                             approximation=approximation, mode='incoh', seed=seed)
# plot_dfl(dfl_MC, domains='sf', phase=True, fig_name = fieldname_MC)
#%%

dfl_prop_SERVAL = deepcopy(dfl_SERVAL)
fieldname_SERVAL = '1-far_zone_25_m_SERVAL'
dfl_prop_SERVAL.prop_m(25, m=[7, 17])

dfl_prop_SERVAL.to_domain(domains='sf') 

plot_dfl(dfl_prop_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)



# corr_MC = dfl_xy_corr(dfl_MC, norm=0)
corr_SERVAL = dfl_xy_corr(dfl_prop_SERVAL, norm=0)

# plot_dfl(corr_MC, domains='sf', phase=True, fig_name = 'corr_MC')
plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr_SERVAL')

# plot_two_dfls(corr_MC, corr_SERVAL, domains='s', fig_name='corr' + simulation_name, 
#               slice_xy=False, phase=False, label_first='corr_MC', 
#               label_second='corr_SERVAL', title=None, filePath=filePath, savefig=False)
filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'SERVAL'
write_field_file(dfl_prop_SERVAL, filePath=filePath)

#%%
filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'SERVAL'
dfl_prop_SERVAL = read_field_file(filePath_SERVAL)

from drawing_routine import plot_2D_1D
plot_dfl_2Dsf(dfl_prop_SERVAL, scale='mm', domains='sf', fig_name='all')

copy_dfl_prop_SERVAL = deepcopy(dfl_prop_SERVAL)

copy_dfl_prop_SERVAL.fld = dfl_prop_SERVAL.fld[3:4,:,:]
plot_dfl_2Dsf(copy_dfl_prop_SERVAL, scale='mm', domains='sf', fig_name='1')

copy_dfl_prop_SERVAL.fld = dfl_prop_SERVAL.fld[4:5,:,:]
plot_dfl_2Dsf(copy_dfl_prop_SERVAL, scale='mm', domains='sf', fig_name='2')

copy_dfl_prop_SERVAL.fld = dfl_prop_SERVAL.fld[5:6,:,:]
plot_dfl_2Dsf(copy_dfl_prop_SERVAL, scale='mm', domains='sf', fig_name='3')

copy_dfl_prop_SERVAL.fld = dfl_prop_SERVAL.fld[6:7,:,:]
plot_dfl_2Dsf(copy_dfl_prop_SERVAL, scale='mm', domains='sf', fig_name='4')
# plot_dfl(dfl_SERVAL_slit, domains='sf', phase=True, fig_name = 'slit')
# plot_dfl(dfl_SERVAL_slit, domains='kf', phase=True, fig_name = 'slit')
Lz, Ly, Lx = 100000e-6, 6000e-6, 6000e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
#%%
### creating RadiationField object
Lz, Ly, Lx = 10000e-6, 6000e-6, 6000e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

dfl_MC = RadiationField((Nz, Ny, Nx))
dfl_MC.dx, dfl_MC.dy, dfl_MC.dz = dx, dy, dz
dfl_MC.xlamds = xlamds
dfl_MC.filePath = filePath
dfl_MC.to_domain('sf')
fieldname_MC = '1-far_zone_50_m_MC'
# approximation = "near_field"
approximation = "far_field"
#%%
dfl_MC = undulator_field_dfl_MP(dfl_MC, z=25, L_w=L_w, E_ph=E_ph, N_e=1, N_b=1,
                                            sig_x=0, sig_y=0, sig_xp=0, sig_yp=0, 
                                            approximation=approximation, mode='incoh', seed=seed)
plot_dfl_2Dsf(dfl_MC, scale='mm', domains='sf', fig_name='SINGLE')
#%%
dfl_SRW =deepcopy(dfl_MC)
dfl_SRW = undulator_field_dfl_SP(dfl_SRW, z=25, L_w=L_w, E_ph=E_ph, N_e=1500,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, 
                                            approximation=approximation, mode='incoh', seed=seed)
#%%
from drawing_routine import plot_2D_1D,plot_dfl_2Dsf
plot_dfl_2Dsf(dfl_SRW, scale='mm', domains='sf', fig_name='SRW')

#%%

# corr_MC = dfl_xy_corr(dfl_MC, norm=0)
corr_SERVAL = dfl_xy_corr(dfl_prop_SERVAL, norm=0)

# plot_dfl(corr_MC, domains='sf', phase=True, fig_name = 'corr_MC')
# plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr_SERVAL')
#%%
from drawing_routine import *
import matplotlib.lines as lines

fig_name="corr_before_slit"

fig = plot_dfl_2Dsf(corr_SERVAL, scale='mm', domains='sf',
              x_lim=500e-3, y_lim=500e-3, savefig=False, show_fig=0, filePath=None, fig_name=fig_name)

ax = fig.axes[0]


kwargs = {'linewidths': 2}
slit_width = 30e-3
slit_separation = 75e-3
ap_x= slit_separation
ax.vlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)
ax.vlines(slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)

ax.hlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)
ax.hlines(slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)

slit_separation = 150e-3
# ap_x= slit_separation
ax.vlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)
ax.vlines(slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)

ax.hlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)
ax.hlines(slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)

slit_separation = 300e-3
# ap_x= slit_separation
ax.vlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)
ax.vlines(slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)

ax.hlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)
ax.hlines(slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)

plt.tight_layout()
plt.show()
fig.savefig(filePath + fig_name + '.' + 'png', format='png', dpi=300)


fig_name="field_before_slit"
fig = plot_dfl_2Dsf(dfl_prop_SERVAL, scale='mm', domains='sf',
              x_lim=1500e-3, y_lim=1500e-3, savefig=False, show_fig=0, filePath=None, fig_name=fig_name)

ax = fig.axes[0]


kwargs = {'linewidths': 2}
slit_width = 30e-3
slit_separation = 75e-3
ap_x= 1
ax.vlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)
ax.vlines(slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)

ax.hlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)
ax.hlines(slit_separation/2, -ap_x/2, ap_x/2, colors='green', **kwargs)

slit_separation = 150e-3
# ap_x= slit_separation
ax.vlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)
ax.vlines(slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)

ax.hlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)
ax.hlines(slit_separation/2, -ap_x/2, ap_x/2, colors='red', **kwargs)

slit_separation = 300e-3
# ap_x= slit_separation
ax.vlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)
ax.vlines(slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)

ax.hlines(-slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)
ax.hlines(slit_separation/2, -ap_x/2, ap_x/2, colors='orange', **kwargs)
plt.tight_layout()
plt.show()
fig.savefig(filePath + fig_name + '.' + 'png', format='png', dpi=300)

#%%
kwargs = {'linewidths': 5}
fig = plot_dfls([corr_SERVAL], dfls_labels=['корреляция'],
          x_lim=500e-3, y_lim=500e-3, slice_xy=True, savefig=False, filePath=filePath, fig_name='correlation', show_fig=0, **kwargs)

ax_x = fig.axes[0]
ax_y = fig.axes[1]

kwargs = {'linewidths': 2}
ax_x.vlines(-0.075/2, 0, 1, colors='green', **kwargs)
ax_x.vlines(0.075/2, 0, 1, colors='green', **kwargs)

ax_x.vlines(-0.15/2, 0, 1, colors='red', **kwargs)
ax_x.vlines(0.15/2, 0, 1, colors='red', **kwargs)

ax_x.vlines(-0.3/2, 0, 1, colors='orange', **kwargs)
ax_x.vlines(0.3/2, 0, 1, colors='orange', **kwargs)

ax_y.vlines(-0.075/2, 0, 1, colors='green', **kwargs)
ax_y.vlines(0.075/2, 0, 1, colors='green', **kwargs)

ax_y.vlines(-0.15/2, 0, 1, colors='red', **kwargs)
ax_y.vlines(0.15/2, 0, 1, colors='red', **kwargs)

ax_y.vlines(-0.3/2, 0, 1, colors='orange', **kwargs)
ax_y.vlines(0.3/2, 0, 1, colors='orange', **kwargs)

fig.savefig(filePath + 'correlation' + '.' + 'pnd', format='png', dpi=300)

plt.show()

#%%
filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'SERVAL'
dfl_prop_SERVAL = read_field_file(filePath_SERVAL)

dfl_SERVAL1 = deepcopy(dfl_prop_SERVAL)
dfl_SERVAL2 = deepcopy(dfl_prop_SERVAL)

slit_width = 30e-6
slit_separation = 75e-6
dfl_SERVAL1 = dfl_ap_rect(dfl_SERVAL1, ap_x=slit_width, ap_y=1e-3, center=(-slit_separation/2, 0))
dfl_SERVAL2 = dfl_ap_rect(dfl_SERVAL2, ap_x=slit_width, ap_y=1e-3, center=( slit_separation/2, 0))
dfl_SERVAL1.fld = dfl_SERVAL1.fld/(np.max(dfl_SERVAL1.intensity(), axis=(1,2))[:,np.newaxis,np.newaxis])**(1/2)
dfl_SERVAL2.fld = dfl_SERVAL2.fld/(np.max(dfl_SERVAL2.intensity(), axis=(1,2))[:,np.newaxis,np.newaxis])**(1/2)
# plot_dfl(dfl_SERVAL1, domains='sf', phase=True, fig_name = 'slit 1')
# plot_dfl(dfl_SERVAL2, domains='sf', phase=True, fig_name = 'slit 2')

dfl_SERVAL_slit = deepcopy(dfl_prop_SERVAL)
dfl_SERVAL_slit.fld = dfl_SERVAL1.fld + dfl_SERVAL2.fld
del(dfl_SERVAL1)
del(dfl_SERVAL2)
#%%
from drawing_routine import plot_2D_1D
plot_dfl_2Dsf(dfl_SERVAL_slit, scale='urad', domains='kf', fig_name='all')

copy_dfl_SERVAL_slit = deepcopy(dfl_SERVAL_slit)

copy_dfl_SERVAL_slit.fld = dfl_SERVAL_slit.fld[3:4,:,:]
plot_dfl_2Dsf(copy_dfl_SERVAL_slit, scale='urad', domains='kf', fig_name='1')
#%%
copy_dfl_SERVAL_slit.fld = dfl_SERVAL_slit.fld[4:5,:,:]
plot_dfl_2Dsf(copy_dfl_SERVAL_slit, scale='urad', domains='kf', fig_name='2')

copy_dfl_SERVAL_slit.fld = dfl_SERVAL_slit.fld[5:6,:,:]
plot_dfl_2Dsf(copy_dfl_SERVAL_slit, scale='urad', domains='kf', fig_name='3')

copy_dfl_SERVAL_slit.fld = dfl_SERVAL_slit.fld[6:7,:,:]
plot_dfl_2Dsf(copy_dfl_SERVAL_slit, scale='urad', domains='kf', fig_name='4')
# plot_dfl(dfl_SERVAL_slit, domains='sf', phase=True, fig_name = 'slit')
# plot_dfl(dfl_SERVAL_slit, domains='kf', phase=True, fig_name = 'slit')
#%%
from drawing_routine import plot_2D_1D

# kwargs = {'color': 'green'}
# kwargs = {'color': 'red'}
kwargs = {'color': 'orange'}

plot_2D_1D(dfl_SERVAL_slit, domains="k", scale='urad', 
           slice_over='y', slice_xy=True, 
           fig_name='y_slits_width_{}_separation_{}_'.format(slit_width, slit_separation), filePath=filePath, savefig=True, **kwargs)
#%%
filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'SERVAL'
dfl_prop_SERVAL = read_field_file(filePath_SERVAL)

dfl_SERVAL1 = deepcopy(dfl_prop_SERVAL)
dfl_SERVAL2 = deepcopy(dfl_prop_SERVAL)

slit_width = 30e-6
slit_separation = 75e-6
dfl_SERVAL1 = dfl_ap_rect(dfl_SERVAL1, ap_x=slit_width, ap_y=1e-3, center=(-slit_separation/2, 0))
dfl_SERVAL2 = dfl_ap_rect(dfl_SERVAL2, ap_x=slit_width, ap_y=1e-3, center=(slit_separation/2, 0))

dfl_SERVAL1.fld = dfl_SERVAL1.fld/(np.max(dfl_SERVAL1.intensity(), axis=(1,2))[:,np.newaxis,np.newaxis])**(1/2)
dfl_SERVAL2.fld = dfl_SERVAL2.fld/(np.max(dfl_SERVAL2.intensity(), axis=(1,2))[:,np.newaxis,np.newaxis])**(1/2)

# plot_dfl(dfl_SERVAL1, domains='sf', phase=True, fig_name = 'slit 1')
# plot_dfl(dfl_SERVAL2, domains='sf', phase=True, fig_name = 'slit 2')

dfl_SERVAL_slit = deepcopy(dfl_prop_SERVAL)
dfl_SERVAL_slit.fld = dfl_SERVAL1.fld + dfl_SERVAL2.fld
del(dfl_SERVAL1)
del(dfl_SERVAL2)
#%%
plot_dfl_2Dsf(dfl_SERVAL_slit, domains='k', scale='urad')
#%%
from drawing_routine import plot_2D_1D

kwargs = {'color': 'green'}
# kwargs = {'color': 'red'}
# kwargs = {'color': 'orange'}
plot_2D_1D(dfl_SERVAL_slit, domains="k", scale='urad', 
           slice_over='x', slice_xy=True, 
           fig_name='x_slits_width_{}_separation_{}'.format(slit_width, slit_separation), filePath=filePath, savefig=True, **kwargs)
#%%
dfl_SERVAL_focus = deepcopy(dfl_SERVAL_slit)

f = 25*10/(25+10)
dfl_SERVAL_focus.curve_wavefront(r=f)

m_x = 0.03
m_y = 0.02
dfl_SERVAL_focus.prop_m(10, m=[m_x, m_y])

plot_dfl(dfl_SERVAL_focus, domains='sf', phase=True, fig_name = 'focus')
























