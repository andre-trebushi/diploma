#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:15:25 2021

@author: andrei
"""

import matplotlib.pyplot as plt
import numpy as np
from ocelot.optics.wave import *
from ocelot.gui.dfl_plot import plot_dfl, plot_1d_hprofile
ocelog.setLevel(logging.INFO)
from copy import deepcopy

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
#%%
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

N_b = 800 #number of statistical realizations
N_e = 100 #number of macro electrons 
Nz, Ny, Nx = N_b, 351, 351 # the shape of the dfl.fld

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
Lz, Ly, Lx = 10000e-6, 400e-6, 400e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# create radiation field
dfl_SERVAL = RadiationField((N_b, Ny, Nx))  
dfl_SERVAL.dx, dfl_SERVAL.dy, dfl_SERVAL.dz = dx, dy, dz
dfl_SERVAL.xlamds = xlamds # SVEA carrieer frequency

dfl_SERVAL.fld = np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) + 1j * np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) # Gaussian noise
# dfl_SERVAL.fld = np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) + 1j * np.random.randn(dfl_SERVAL.Nz(), dfl_SERVAL.Ny(), dfl_SERVAL.Nx()) # Gaussian noise

dfl_SERVAL.filePath = filePath+'.dfl'
# dfl_omega_0 = 2*np.pi * speed_of_light / dfl.xlamds # set undulator resonance to SVEA carrier frequency (to the middle of the photon energy domain mesh)
# radiation_omega_resonance = dfl_omega_0 * 1.01 # shift undulator resonance to the right

fieldname_SERVAL = '0-source_SERVAL'
dfl_SERVAL = undulator_field_dfl_SERVAL(dfl_SERVAL, L_w=L_w, 
                                        sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, 
                                        sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp,
                                        k_support = 'intensity', s_support='conv_intensities', showfig=False)

plot_dfl(dfl_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)
plot_dfl(dfl_SERVAL, domains='kf', phase=True, fig_name = fieldname_SERVAL)
#%%
dfl_prop_SERVAL = deepcopy(dfl_SERVAL)
fieldname_SERVAL = '1-far_zone_25_m_SERVAL'
dfl_prop_SERVAL.prop_m(25, m=[12, 12])

dfl_prop_SERVAL.to_domain(domains='sf') 

plot_dfl(dfl_prop_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)
plot_dfl(dfl_prop_SERVAL, domains='kf', phase=True, fig_name = fieldname_SERVAL)

corr_SERVAL = dfl_xy_corr(dfl_prop_SERVAL, norm=1)
plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr')

#%%
filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'SERVAL'
write_field_file(dfl_prop_SERVAL, filePath=filePath_SERVAL)

#%%
# ebeam_sigma_x = 1e-10
# ebeam_sigma_y = 1e-10
# ebeam_sigma_xp = 1e-10
# ebeam_sigma_yp = 1e-10

# Lz, Ly, Lx = 100000e-6, 2000e-6, 2000e-6 #size of realspace grid [m]
# dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# ### creating RadiationField object
# dfl_MC = RadiationField((1, 551, 551))
# dfl_MC.dx, dfl_MC.dy, dfl_MC.dz = dx, dy, dz
# dfl_MC.xlamds = xlamds
# dfl_MC.filePath = filePath
# dfl_MC.to_domain('sf')

# fieldname_MC = '1-far_zone_50_m_MC'
# # approximation = "near_field"
# approximation = "far_field"

# dfl_MC = undulator_field_dfl_MP(dfl_MC, z=25, L_w=L_w, E_ph=E_ph, N_e=100, N_b=1,
#                                             sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, 
#                                             approximation=approximation, mode='incoh', seed=seed)
# plot_dfl(dfl_MC, domains='sf', phase=True, fig_name = fieldname_MC)
# #%%
# corr_SERVAL = dfl_xy_corr(dfl_MC, norm=1)
# plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr')

#%%
fieldname_SERVAL = 'SERVAL'
filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'SERVAL'
dfl_SERVAL = read_field_file(filePath_SERVAL)

plot_dfl(dfl_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)
plot_dfl(dfl_SERVAL, domains='kf', phase=True, fig_name = fieldname_SERVAL)
#%%
# generating highly polished mirror by height errors RMS
hrms=3e-9
hprofile = generate_1d_profile(hrms=hrms,               # [m] height errors root mean square
                               length=0.4,             # [m] length of the mirror surface
                               points_number=5001,      # number of points (pixels) at the mirror surface
                               wavevector_cutoff=0,     # [1/m] point on k axis for cut off large wave lengths in the PSD (with default value 0 effects on nothing)
                               psd=None,                # [m^3] 1d array; power spectral density of surface (if not specified, will be generated)
                               seed=666)                # seed for np.random.seed() to allow reproducibility

# plotting 1d height profile
plot_1d_hprofile(hprofile, fig_name='mirror1 height profile and PSD')


# plotting generated RadiationField
# dfl = deepcopy(dfl_SERVAL)
# del(dfl_SERVAL)
dfl = deepcopy(dfl_prop_SERVAL)

# plot_dfl(dfl, phase=1, fig_name='radiation before mirror1')
# reflecting generated RadiationField from the imperfect mirror
f = 25 * 25 /(25 + 25)

dfl.curve_wavefront(r=12.5)
dfl_reflect_surface(dfl,                        # ocelot.optics.wave.RadiationField, which will be reflected from imperfect mirror surface (hprofile2)
                    angle=np.pi * 1 / 180,      # [radians] angle of incidence with respect to the surface
                    hrms=None,                  # [m] height errors root mean square
                    height_profile=hprofile,    # HeightProfile object of the reflecting surface (if not specified, will be generated using hrms)
                    axis='x')                   # direction along which reflection takes place

# plotting RadiationField after reflection
# plot_dfl(dfl, phase=1, domains='sf',fig_name='radiation after reflection from mirror1')
# plot_dfl(dfl, phase=1, domains='kf', fig_name='radiation after reflection from mirror1')

# propagating RadiationField for 5 meters


dfl.prop_m(z=12.5, m=0.5)

# plotting RadiationField after propagation
plot_dfl(dfl, domains='sf', phase=1, fig_name='radiation after reflection from mirror1 and propagation at 5 m')
plot_dfl(dfl, domains='kf', phase=1, fig_name='radiation after reflection from mirror1 and propagation at 5 m')

filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'x_SERVAL_radiaiton_after_reflection_and_12_5_of_free_space_{}_rms'.format(hrms*1e9)
write_field_file(dfl, filePath=filePath_SERVAL)

corr_SERVAL = dfl_xy_corr(dfl, norm=1)
plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr_after_reflection')

dfl2 = deepcopy(dfl)

dfl2.prop_m(z=12.5, m=[0.25, 0.25])

# plotting RadiationField after propagation
plot_dfl(dfl2, phase=1, fig_name='radiation in focus')
plot_dfl(dfl2, phase=1, domains='kf', fig_name='radiation in focus')

corr_SERVAL = dfl_xy_corr(dfl2, norm=0)
plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr_after_reflection')

filePath_SERVAL = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/' + 'x_SERVAL_radiaiton_in_focus_{}_rms'.format(hrms*1e9)
write_field_file(dfl2, filePath=filePath_SERVAL)









