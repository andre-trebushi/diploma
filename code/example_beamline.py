#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:22:33 2021

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

ocelog.setLevel(logging.INFO)
_logger = logging.getLogger(__name__)

def plot_two_dfls(dfl_first, dfl_second, domains='s', scale='mm', label_first=None, label_second=None, title=None,
            x_lim=None, y_lim=None, slice_xy=False, phase=False, savefig=False, showfig=True, filePath=None, fig_name=None):
    """ 
    Parameters
    ----------
    dfl_first : RadiationField object
        3d or 2d coherent radiation distribution, *.fld variable is the same as Genesis dfl structure
    dfl_second : RadiationField object
        3d or 2d coherent radiation distribution, *.fld variable is the same as Genesis dfl structure
    domains : str, optional
        A transverse domain in which the fields will be represented. The default is 's'.
    label_first : str, optional
        Label of the first field to be plotted in the legend. When the default is None no label will be displayed.
    label_second : str, optional
        Label of the second field to be plotted in the legend. When the default is None no label will be displayed.
    x_lim : float, optional
        Limit of the x scale whether in um or urad. When the default is None minimum scale of two field will be plotted.
    y_lim : float, optional
        Limit of the y scale whether in um or urad. When the default is None minimum scale of two field will be plotted.
    slice_xy : bool type, optional
        if True, slices will be plotted; if False, projections will be plotted.
    phase : bool type, optional
        can replace XY projection with the phasefront distribution. The default is False.
    figname : str, optional
        The name of the figure to be plotted and saved. The default is 'figname'.

    Returns
    -------
    None.

    """
    
    _logger.info('Plotting two dfls in process...')
    start_time = time.time()

    cmap_ph = plt.get_cmap('hsv')
    
    dfl1 = deepcopy(dfl_first)
    dfl2 = deepcopy(dfl_second)
    
    if filePath is None:
        filePath = dfl1.filePath
    
    if fig_name is None:
        if dfl1.fileName() == '':
            fig = plt.figure('Radiation comparison' + domains)
        else:
            fig = plt.figure('Radiation comparison ' + domains + ' ' + dfl1.fileName() + ' ' + dfl2.fileName())
    else:
        fig = plt.figure(fig_name + domains)
    
    figsize = 12
    # fig = plt.figure(fig_name)
    plt.clf()
     
    fig.set_size_inches(figsize, figsize, forward=True)
    if title is not None: 
        fig.suptitle(title, fontsize=18, ha='right')
        
    ax_x = fig.add_subplot(2, 2, 1)
    ax_y = fig.add_subplot(2, 2, 2)
    
    ax_x2 = ax_x.twinx()  # instantiate a second axes that shares the same x-axis
    ax_y2 = ax_y.twinx()

    _logger.info(ind_str + "domains={}".format(str(domains)))

    if domains == 'k':
        if scale=='urad':
            label_x = r'$\theta_y, мкрад$'
            label_y = r'$\theta_x, мкрад$'
            scale_order = 1e6
        elif scale=='mkrad':
            label_x = r'$x, мрад$'
            label_y = r'$y, мрад$'
            scale_order = 1e3
    elif domains == 's':
        if scale=='um':
            label_x = r'$x, мкм$'
            label_y = r'$y, мкм$'
            scale_order = 1e6
        elif scale=='mm':
            label_x = r'$x, мм$'
            label_y = r'$y, мм$'
            scale_order = 1e3
    dfl1.to_domain(domains)
    dfl2.to_domain(domains)
    
    if None in [x_lim, y_lim]:
        x_lim = min(np.max(dfl1.scale_x())*scale_order, np.max(dfl2.scale_x())*scale_order)
        y_lim = min(np.max(dfl1.scale_y())*scale_order, np.max(dfl2.scale_y())*scale_order)

    _logger.debug(ind_str + "x_lim = {}, y_lim = {}".format(x_lim, y_lim))
    ax_y.set_xlim(-y_lim, y_lim)
    ax_y2.set_xlim(-y_lim, y_lim)
    ax_x.set_xlim(-x_lim, x_lim)
    ax_x2.set_xlim(-x_lim, x_lim)        
    
    ax_y.grid()
    ax_x.grid()
    
    if slice_xy is True:   
        fig.text(0.01, 0.01, 'x- y- срез', fontsize=18)
        I_1x = np.sum(dfl1.intensity(), axis=0)[dfl1.Ny()//2, :]
        I_2x = np.sum(dfl2.intensity(), axis=0)[dfl2.Ny()//2, :]
        I_1y = np.sum(dfl1.intensity(), axis=0)[:, dfl1.Nx()//2]
        I_2y = np.sum(dfl2.intensity(), axis=0)[:, dfl2.Nx()//2]
    elif slice_xy is False:
        fig.text(0.01, 0.01, 'x- y- проекция', fontsize=18)
        I_1x = np.sum(dfl1.intensity(), axis=(0,1))
        I_2x = np.sum(dfl2.intensity(), axis=(0,1))
        I_1y = np.sum(dfl1.intensity(), axis=(0,2))
        I_2y = np.sum(dfl2.intensity(), axis=(0,2))
    else: 
        raise AttributeError('slice_xy is a boolean type')
        _logger.error(ind_str + 'slice_xy is a boolean type')
    std_1x = std_moment(dfl1.scale_x()*scale_order, I_1x)  
    std_2x = std_moment(dfl2.scale_x()*scale_order, I_2x) 
    print(std_1x, std_2x, std_1x/std_2x)
    
    std_1y = std_moment(dfl1.scale_x()*scale_order, I_1y)  
    std_2y = std_moment(dfl2.scale_x()*scale_order, I_2y) 
    print(std_1y, std_2y, std_1y/std_2y)
    
    ax_x.plot(dfl1.scale_x()*scale_order, I_1x, c='b', label=label_first)
    ax_x2.plot(dfl2.scale_x()*scale_order, I_2x, c='green', label=label_second)
    
    ax_y.plot(dfl1.scale_y()*scale_order, I_1y, c='b', label=label_first)
    ax_y2.plot(dfl2.scale_y()*scale_order, I_2y, c='green', label=label_second)    
    
    if None not in [label_first, label_second]:
        ax_y2.legend(fontsize=12, bbox_to_anchor=(0, 0.92), loc='upper left')#, loc=1)
        ax_y.legend(fontsize=12, bbox_to_anchor=(0, 0.995), loc='upper left')#, loc=2)
        ax_x2.legend(fontsize=12, bbox_to_anchor=(0, 0.92), loc='upper left')#, loc=1)
        ax_x.legend(fontsize=12, bbox_to_anchor=(0, 0.995), loc='upper left')#, loc=2)
        
    ax_xy1 = fig.add_subplot(2, 2, 3)
    ax_xy2 = fig.add_subplot(2, 2, 4)

    if phase == True:
        xy_proj_ph1 = np.angle(np.sum(dfl1.fld, axis=0))
        xy_proj_ph2 = np.angle(np.sum(dfl2.fld, axis=0))
        ax_xy1.pcolormesh(dfl1.scale_x()*scale_order, dfl1.scale_y()*scale_order, xy_proj_ph1, cmap=cmap_ph, vmin=-np.pi, vmax=np.pi)        
        ax_xy2.pcolormesh(dfl2.scale_x()*scale_order, dfl2.scale_y()*scale_order, xy_proj_ph2, cmap=cmap_ph, vmin=-np.pi, vmax=np.pi)
    else:
        ax_xy1.pcolormesh(dfl1.scale_x()*scale_order, dfl1.scale_y()*scale_order, np.sum(dfl1.intensity(), axis=0))         
        ax_xy2.pcolormesh(dfl2.scale_x()*scale_order, dfl2.scale_y()*scale_order, np.sum(dfl2.intensity(), axis=0))

    ax_xy2.set_xlim(-x_lim, x_lim)
    ax_xy2.set_ylim(-y_lim, y_lim)
    ax_xy1.set_xlim(-x_lim, x_lim)
    ax_xy1.set_ylim(-y_lim, y_lim)

    ax_xy1.set_title(label_first, fontsize=14, color='b')      
    ax_xy2.set_title(label_second, fontsize=14, color='green')
          
    ax_xy1.set_ylabel(label_y, fontsize=16)
    ax_xy1.set_xlabel(label_x, fontsize=16)
    ax_xy2.set_ylabel(label_y, fontsize=16)
    ax_xy2.set_xlabel(label_x, fontsize=16)
    ax_x.set_xlabel(label_x, fontsize=16)
    ax_y.set_xlabel(label_y, fontsize=16)
    ax_x.set_ylabel('пр.е', fontsize=16)
    ax_y.set_ylabel('пр.е', fontsize=16)
    ax_x.set_ylim(0)
    ax_y.set_ylim(0)
    ax_x2.set_ylim(0)
    ax_y2.set_ylim(0)
    
    ax_x.set_box_aspect(1)
    ax_y.set_box_aspect(1)
    ax_xy1.set_box_aspect(1)
    ax_xy2.set_box_aspect(1)
    
    plt.tight_layout()
    
    if savefig != False:
        if savefig == True:
            savefig = 'png'
        _logger.debug(ind_str + 'saving *{:}.{:}'.format(fig_name, savefig))
        fig.savefig(filePath + fig_name + '.' + str(savefig), format=savefig, dpi=300)
    _logger.debug(ind_str + 'done in {:.2f} seconds'.format(time.time() - start_time))

    plt.draw()

    if showfig == True:
        _logger.debug(ind_str + 'showing two dfls')
        rcParams["savefig.directory"] = os.path.dirname(filePath)
        plt.show()
    else:
        plt.close(fig)   
    _logger.info(ind_str + 'plotting two dfls done in {:.2f} seconds'.format(time.time() - start_time))

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

N_b = 300 #number of statistical realizations
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
Lz, Ly, Lx = 100000e-6, 6000e-6, 6000e-6 #size of realspace grid [m]
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

### creating RadiationField object
dfl_MC = RadiationField((Nz, Ny, Nx))
dfl_MC.dx, dfl_MC.dy, dfl_MC.dz = dx, dy, dz
dfl_MC.xlamds = xlamds
dfl_MC.filePath = filePath
dfl_MC.to_domain('sf')

fieldname_MC = '1-far_zone_50_m_MC'
# approximation = "near_field"
approximation = "far_field"

dfl_MC = undulator_field_dfl_MP(dfl_MC, z=25, L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
                                            sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, 
                                            approximation=approximation, mode='incoh', seed=seed)
#%%
dfl_MC.to_domain(domains='sf') 
plot_dfl(dfl_MC, domains='sf', phase=True, fig_name=fieldname_MC)
plot_dfl(dfl_MC, domains='kf', phase=True, fig_name = fieldname_MC)

filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/'
# write_field_file(dfl_MC, filePath=filePath, fileName=fieldname_MC)

filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_example_beamline/MCA'
# write_field_file(dfl_MC, filePath=filePath)
#%%
# Define mesh size
Lz, Ly, Lx = 10000e-6, 300e-6, 400e-6 #size of realspace grid [m]
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
plot_dfl(dfl_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)
plot_dfl(dfl_SERVAL, domains='f', phase=True, fig_name = fieldname_SERVAL)

filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_example_beamline/SERVAL'
# write_field_file(dfl_SERVAL, filePath=filePath)
#%%
filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_example_beamline/SERVAL'
dfl_SERVAL = read_field_file(filePath)


dfl_prop_SERVAL = deepcopy(dfl_SERVAL)
fieldname_SERVAL = '1-far_zone_25_m_SERVAL'
dfl_prop_SERVAL.prop_m(25, m=[15, 15])
dfl_prop_SERVAL.to_domain('st')

dfl_prop_SERVAL.to_domain(domains='sf') 

plot_dfl(dfl_prop_SERVAL, domains='sf', phase=True, fig_name = fieldname_SERVAL)
plot_dfl(dfl_prop_SERVAL, domains='kf', phase=True, fig_name = fieldname_SERVAL)

filePath = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/' + simulation_name + '/'
# write_field_file(dfl_prop_SERVAL, filePath=filePath, fileName=fieldname_SERVAL)



# fieldname_MC = '1-far_zone_50_m_MC'
# fieldname_SERVAL = '1-far_zone_50_m_SERVAL'

# filepath_MC = filePath + fieldname_MC
# filepath_SERVAL = filePath + fieldname_SERVAL

# dfl_MC = read_field_file(filepath_MC)
# dfl_SERVAL = read_field_file(filepath_SERVAL)

# simulation_name = simulation_name.replace('.', '_')

#%%
filePath_MC = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_example_beamline/MCA'
dfl_MC = read_field_file(filePath_MC)

plot_two_dfls(dfl_MC, dfl_prop_SERVAL, domains='s', fig_name='1-far_zone_25_m' + simulation_name, 
              slice_xy=False, phase=False, label_first='МСA', 
              label_second='СЕРВАЛ', title=None, filePath=filePath, savefig=True)

corr_MC = dfl_xy_corr(dfl_MC, norm=1)
corr_SERVAL = dfl_xy_corr(dfl_prop_SERVAL, norm=1)

# # plot_dfl(corr_MC, domains='sf', phase=True, fig_name = 'corr_MC')
# # plot_dfl(corr_SERVAL, domains='sf', phase=True, fig_name = 'corr_SERVAL')

plot_two_dfls(corr_MC, corr_SERVAL, domains='s', fig_name='corr'+ simulation_name, 
              slice_xy=True, phase=False, label_first='МСА', 
              label_second='СЕРВАЛ', title=None, filePath=filePath, savefig=True)

#%%
dfl2_MC = deepcopy(dfl_MC)
dfl2_SERVAL = deepcopy(dfl_prop_SERVAL)

dfl2_MC.to_domain(domains='sf')
dfl2_SERVAL.to_domain(domains='sf')

ap_x = 1e-3
ap_y = 1e-3
dfl_ap_rect(dfl2_MC, ap_x=ap_x, ap_y=ap_y)
dfl_ap_rect(dfl2_SERVAL, ap_x=ap_x, ap_y=ap_y)

# interpL = 0.25
# interpN = 4
# dfl_interp(dfl2_MC, interpN=(1, interpN), interpL=(1, interpL), method='quintic')
# dfl_interp(dfl2_SERVAL, interpN=(1, interpN), interpL=(1, interpL), method='quintic')

dfl2_MC.prop(10)
dfl2_SERVAL.prop(10)

fieldname = '2-far_zone_60_m_after_aperture'
# plot_dfl(dfl2_SRW, domains='sf', phase=True, fig_name = fieldname)
# plot_dfl(dfl2_SRW, domains='kf', phase=True, fig_name = fieldname)

# plot_dfl(dfl2_MC, domains='s', phase=True, fig_name = filePath + fieldname)

#%%
plot_two_dfls(dfl2_MC, dfl2_SERVAL, domains='s', fig_name=fieldname + simulation_name, 
              slice_xy=False, phase=False, label_first='MCA', 
              label_second='SERVAL', title=None, filePath=filePath, savefig=True)

# plot_two_dfls(dfl2_MC, dfl2_SERVAL, domains='k', fig_name=fieldname + simulation_name, 
#               slice_xy=False, phase=False, label_first='MCA', 
#               label_second='SERVAL', title=None, filePath=filePath, savefig=False)

#%%
dfl3_MC = deepcopy(dfl2_MC)
dfl3_SERVAL = deepcopy(dfl2_SERVAL)

f = 35*10/(35+10)
dfl3_MC.curve_wavefront(r=f)
dfl3_SERVAL.curve_wavefront(r=f)

m_x = 0.03
m_y = 0.02
dfl3_MC.prop_m(10, m=[m_x, m_y])
dfl3_SERVAL.prop_m(10, m=[m_x, m_y])

#%%
fieldname = '3-70_m_focal_plane'
plot_two_dfls(dfl3_MC, dfl3_SERVAL, domains='s', fig_name=fieldname  + simulation_name, 
              slice_xy=True, phase=False, label_first='MCA', 
              label_second='SERVAL', title=None, filePath=filePath, savefig=True)


# save of your python script in the simulation directory
# write_script(script_dir, new_script_dir)



















