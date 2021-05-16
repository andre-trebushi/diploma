#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:54:10 2021

@author: andrei
"""
import numpy as np
import matplotlib.pyplot as plt
from drawing_routine import *
from ocelot.optics.wave import *
from ocelot.gui.dfl_plot import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
import scipy.special as sc
from scipy import signal
from scipy import misc
import copy

ocelog.setLevel(logging.INFO)
_logger = logging.getLogger(__name__)


def undulator_field_dfl_SERVAL(dfl, L_w, sig_x=0, sig_y=0, sig_xp=0, sig_yp=0, k_support = 'intensity', s_support='conv_intensities', showfig=False, seed=None):
    filePath = '/home/andrei/Documents/diploma/Diploma/images/'
    _logger.info('Generating undulator field with Serval algorithm')
    w_0 = 2*np.pi * speed_of_light / dfl.xlamds
    
    if showfig:
        plot_dfl_2Dsf(dfl, scale='um', domains='sf', savefig=True, 
                      fig_name = '1-X_noise', filePath=filePath)
        # plot_dfl(dfl, line_off_xy = False, fig_name = '1-X_noise')
    
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
        # plot_dfl(dfl, domains='s', line_off_xy = False, fig_name = '2-X_e-beam-size')
        plot_dfl_2Dsf(dfl, scale='um', domains='sf', savefig=True,
                      fig_name = '2-X_e-beam-size', filePath=filePath)
        # plot_dfl(dfl, domains='k', line_off_xy = False, fig_name = '2-X_e-beam-size')
        plot_dfl_2Dsf(dfl, scale='um', domains='kf', savefig=True,
                      fig_name = '2-X_e-beam-divergence', filePath=filePath)
                
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
        # plot_dfl(dfl, domains='s', fig_name = '3-X_radaition_size')
        plot_dfl_2Dsf(dfl, scale='um', domains='sf', savefig=True, 
                      fig_name = '3-X_radaition_size', filePath=filePath)
        # plot_dfl(dfl, domains='k', fig_name = '3-X_radiation_divergence')
        plot_dfl_2Dsf(dfl, scale='um', domains='kf', savefig=True,
                      fig_name = '3-X_radaition_divergence', filePath=filePath)
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
# ebeam_sigma_x = 50e-6
# ebeam_sigma_y = 10e-6
# ebeam_sigma_xp = 10e-06
# ebeam_sigma_yp = 50e-06
#### #3
ebeam_sigma_x = 38e-06
ebeam_sigma_y = 4.68e-06
ebeam_sigma_xp = 25e-06
ebeam_sigma_yp = 20e-06

ebeam_sigma_z = 2000e-6
ebeam_sigma_gamma = 1e-4 #TODO: relative electron energy spread

N_b = 400 #number of statistical realizations
N_e = 150 #number of macro electrons 
Nz, Ny, Nx = 1, 301, 301 # the shape of the dfl.fld

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

e_beam_param = r'$N_x$ = {}, '.format(round((ebeam_sigma_x)**2/xlamds/L_w, 3)) + r'$N_y$ = {}, '.format(round((ebeam_sigma_y)**2/xlamds/L_w, 3)) + \
               r'$D_x$ = {}, '.format(round((ebeam_sigma_xp)**2 * L_w/xlamds, 3)) + r'$D_y$ = {}, '.format(round((ebeam_sigma_yp)**2 * L_w/xlamds, 3)) + \
               r'$N_b$ = {} '.format(N_b) + r'$N_e = {}$'.format(N_e)
print(e_beam_param)

#%% 
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
Lz, Ly, Lx = 10000e-6, 140e-6, 300e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# create radiation field
dfl_SERVAL = RadiationField((501, 451, 451))  
dfl_SERVAL.dx, dfl_SERVAL.dy, dfl_SERVAL.dz = dx, dy, dz
dfl_SERVAL.xlamds = xlamds # SVEA carrieer frequency

dfl_SERVAL.fld = np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) + 1j * np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) # Gaussian noise

dfl_SERVAL.filePath = filePath+'.dfl'
# dfl_omega_0 = 2*np.pi * speed_of_light / dfl.xlamds # set undulator resonance to SVEA carrier frequency (to the middle of the photon energy domain mesh)
# radiation_omega_resonance = dfl_omega_0 * 1.01 # shift undulator resonance to the right

dfl1 = deepcopy(dfl_SERVAL)
dfl2 = deepcopy(dfl_SERVAL)
dfl3 = deepcopy(dfl_SERVAL)

Lz, Ly, Lx = 10000e-6, 7500e-6, 7500e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
dfl4 = RadiationField((N_b, Ny, Nx)) 
dfl4.dx, dfl4.dy, dfl4.dz = dx, dy, dz
dfl4.xlamds = xlamds # SVEA carrieer frequency


fieldname_SERVAL = '1-far_zone_50_m_SERVAL'
approximation = 'far_field'
dfl_SERVAL_conv_intensity = undulator_field_dfl_SERVAL(dfl1, L_w=L_w, 
                                        sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, 
                                        sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp,
                                        k_support = 'intensity', s_support='conv_intensities', showfig=False)

# plot_dfl(dfl_SERVAL_conv_intensity, domains='sf', fig_name='dfl_SERVAL_conv_intensity')
# plot_dfl(dfl_SERVAL_conv_intensity, domains='kf', fig_name='dfl_SERVAL_conv_intensity')
#%%
dfl_SERVAL_conv_amplitude = undulator_field_dfl_SERVAL(dfl2, L_w=L_w, 
                                        sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, 
                                        sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp,
                                        k_support = 'intensity', s_support='conv_amplitudes', showfig=False)

# plot_dfl(dfl_SERVAL_conv_amplitude, domains='sf', fig_name='dfl_SERVAL_conv_intensity')

dfl_SERVAL_beam = undulator_field_dfl_SERVAL(dfl3, L_w=L_w, 
                                        sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, 
                                        sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp,
                                        k_support = 'intensity', s_support='beam', showfig=False)

# plot_dfl(dfl_SERVAL_beam, domains='sf', fig_name='dfl_SERVAL_beam')


# dfl_SP = undulator_field_dfl_MP(dfl4, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
#                                             sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
#                                             approximation=approximation, mode='incoh')
#%%
filePath_SP = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_example_beamline/' + 'MCA'
dfl_SP = read_field_file(filePath_SP)

# plot_dfl(dfl_SP, domains='sf', fig_name='before prop')
dfl_SP.prop_m(-25, m=[0.05, 0.02])
# plot_dfl(dfl_SERVAL_intensity_conv)
# plot_dfl(dfl_SERVAL_amplitude_beam)
# plot_dfl(dfl_SP, domains='sf')
# plot_dfl(dfl_SP, domains='kf')
#%%
dfl_SERVAL_conv_intensity.to_domain('sf')
dfl_SERVAL_conv_amplitude.to_domain('sf')
dfl_SERVAL_beam.to_domain('sf')
dfl_SP.to_domain('sf')
#%%
filePath = '/home/andrei/Documents/diploma/Diploma/images/'
fig_name = 'SERVAL_envelopes_comparison_source'
dfls_labels = [r'свёртка интесивностей', r'свёртка амплитуд',r'электронный пучок',r'МСА']
colors = ['green','orange', 'red', 'blue']

from drawing_routine import plot_dfls
filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/'

plot_dfls([dfl_SERVAL_conv_intensity, dfl_SERVAL_conv_amplitude, dfl_SERVAL_beam, dfl_SP], dfls_labels, colors,
          x_lim=150e-3, y_lim=50e-3, slice_xy=True, savefig=True, filePath=filePath, fig_name=fig_name)#, dfl_SERVAL_amplitude_beam, dfl_SP])
#%%

dfl_SP_prop = deepcopy(dfl_SP)
dfl_SERVAL_conv_intensity_prop = deepcopy(dfl_SERVAL_conv_intensity)
dfl_SERVAL_conv_amplitude_prop = deepcopy(dfl_SERVAL_conv_amplitude)
dfl_SERVAL_beam_prop = deepcopy(dfl_SERVAL_beam)

m=[14, 20]
dfl_SP_prop.prop_m(25, m=[1/0.04, 1/0.015])
dfl_SERVAL_conv_intensity_prop.prop_m(25, m=m)
# dfl_SERVAL_conv_amplitude_prop.prop_m(25, m=m)
# dfl_SERVAL_beam_prop.prop_m(25, m=m)

dfl_SERVAL_conv_intensity_prop.to_domain('sf')
# plot_dfl(dfl_SERVAL_conv_intensity_prop, domains='sf', fig_name='dfl_SERVAL_conv_intensity_prop')
# plot_dfl(dfl_SERVAL_conv_intensity_prop, domains='kf', fig_name='dfl_SERVAL_conv_intensity_prop')

dfl_SERVAL_conv_amplitude_prop.prop_m(25, m=m)
dfl_SERVAL_beam_prop.prop_m(25, m=m)
#%%
dfls_labels = [r'СЕРВАЛ', r'МСА']
colors = ['green', 'blue']
fig_name = 'SERVAL_envelopes_comparison_far_zone'

from drawing_routine import plot_dfls
filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/'

plot_dfls([dfl_SERVAL_conv_intensity_prop, dfl_SP_prop], dfls_labels, colors, 
          x_lim=3000e-3, y_lim=3000e-3, slice_xy=True, savefig=True, filePath=filePath, fig_name=fig_name)#, dfl_SERVAL_amplitude_beam, dfl_SP])

#%%
corr_SERVAL_conv_intensity = dfl_xy_corr(dfl_SERVAL_conv_intensity_prop, norm=0)
corr_SERVAL_conv_amplitude = dfl_xy_corr(dfl_SERVAL_conv_amplitude_prop, norm=0)
corr_SERVAL_beam = dfl_xy_corr(dfl_SERVAL_beam_prop, norm=0)
corr_SP = dfl_xy_corr(dfl_SP_prop, norm=0)

# plot_dfl(corr_SERVAL_conv_intensity, domains='sf', phase=True, fig_name = 'corr_MC')
# plot_dfl(corr_SP, domains='sf', phase=True, fig_name = 'corr_SERVAL')

dfls_labels = [r'свёртка интесивностей', r'свёртка амплитуд',r'электронный пучок',r'М.С.А']
colors = ['green','orange', 'red', 'blue']

fig_name = 'SERVAL_corr_comparison'

from drawing_routine import plot_dfls
filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/'

plot_dfls([corr_SERVAL_conv_intensity, corr_SERVAL_conv_amplitude, corr_SERVAL_beam, corr_SP], dfls_labels, colors,
          x_lim=500e-3, y_lim=1500e-3, slice_xy=True, savefig=True, filePath=filePath, fig_name=fig_name)#, dfl_SERVAL_amplitude_beam, dfl_SP])


















