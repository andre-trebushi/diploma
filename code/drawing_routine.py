#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:36:45 2021

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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # for wigner log scale
from ocelot.gui.dfl_plot import *
import numpy as np 
import h5py
from ocelot.optics.wave import *

# sim_dir = r'd:\DESYcloud\projects\2020_Partially_coherent\sim_dir'

# ocelog.setLevel(logging.INFO)
# _logger = logging.getLogger(__name__)

# n_s = 80
# l_w = 0.04 # [m] undulator period 
# L_w = l_w * n_s

# E_ph = 4500 # eV
# w = E_ph / hr_eV_s 
# xlamds = 2 * np.pi * speed_of_light / w

# sigma_r = np.sqrt(2*xlamds*L_w)/4/np.pi #natural radiation size in the waist
# sigma_rp = np.sqrt(xlamds/2/L_w) #natural radiation divergence at the waist

# #### #1
# ebeam_sigma_x = 30e-06
# ebeam_sigma_y = ebeam_sigma_x
# ebeam_sigma_xp = 5e-06
# ebeam_sigma_yp = ebeam_sigma_xp

# ebeam_sigma_x_z = 200000e-6
# ebeam_sigma_gamma = 1e-4 #TODO: relative electron energy spread

# N_b = 1 #number of statistical realizations
# N_e = 50 #number of macro electrons 
# Nz, Ny, Nx = N_b, 301, 301 # the shape of the dfl.fld

# # Nz, Ny, Nx = N_b, 100, 100 # the shape of the dfl.fld
# # seed=1
# filePath = '/home/andrei'

# e_beam_param = r'$N_x$ = {}, '.format(round((ebeam_sigma_x)**2/xlamds/L_w, 3)) + r'$N_y$ = {}, '.format(round((ebeam_sigma_y)**2/xlamds/L_w, 3)) + \
#                r'$D_x$ = {}, '.format(round((ebeam_sigma_xp)**2 * L_w/xlamds, 3)) + r'$D_y$ = {}, '.format(round((ebeam_sigma_yp)**2 * L_w/xlamds, 3)) + \
#                r'$N_b$ = {} '.format(N_b) + r'$N_e = {}$'.format(N_e)
# print(e_beam_param)

# # Monte Calro
# Lz, Ly, Lx = 1000e-6, 15.4e-3, 15.4e-3#27.37e-3 #size of realspace grid [m]
# dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# ### creating RadiationField object
# dfl = RadiationField((Nz, Ny, Nx))
# dfl.dx, dfl.dy, dfl.dz = dx, dy, dz
# dfl.xlamds = xlamds
# dfl.filePath = filePath
# dfl.to_domain('sf')

# fieldname_MC = ''
# approximation = "far_field"
# # approximation = "near_field"

# dfl = undulator_field_dfl_MP(dfl, z=120+6.2*(35-19-2), L_w=L_w, E_ph=E_ph, N_e=N_e, N_b=N_b,
#                                             sig_x=ebeam_sigma_x, sig_y=ebeam_sigma_y, sig_xp=ebeam_sigma_xp, sig_yp=ebeam_sigma_yp, C=0,
#                                             approximation=approximation, mode='incoh')

# plot_dfl(dfl, domains='sf', phase=True, fig_name = fieldname_MC, cbar=True)
# plot_dfl(dfl, domains='kf', phase=True, fig_name = fieldname_MC)
#%%
def plot_dfl_2Dsf(dfl, scale='mm', domains='sf',
                  savefig=False, showfig=True, filePath=None, fig_name=None):
    figsize=2
    fig = plt.figure(fig_name)
    plt.clf()
    fig.set_size_inches((4*figsize, 4*figsize))#, forward=True)
    # cmap_ph = 'gray_r'
    cmap_ph = 'viridis'
    ax = fig.add_subplot(111)
    
    domains_orig=dfl.domains()
    dfl.to_domain(domains)
    
    if domains=='sf':
        # if scale=='mm':
        #     xlabel = 'x, [мм]'
        #     ylabel = 'y, [мм]'
        #     x = dfl.scale_kx()*1e3
        #     y = dfl.scale_ky()*1e3
        # elif scale=='um':
        xlabel = 'x, [мкм]'
        ylabel = 'y, [мкм]'
        x = dfl.scale_kx()*1e6
        y = dfl.scale_ky()*1e6
    # else:
        # print('domains and scales must match each other')
            
    elif domains=='kf':
        # if scale=='мрад':
        #     xlabel = 'x, [мрад]'
        #     ylabel = 'y, [мрад]'
        #     x = dfl.scale_kx()*1e3
        #     y = dfl.scale_ky()*1e3
        # elif scale=='мкрад':
        xlabel = 'x, [мкрад]'
        ylabel = 'y, [мкрад]'
        x = dfl.scale_x()*1e6
        y = dfl.scale_y()*1e6
    # else:
    #     print('domains and scales must match each other')
        
    ax.pcolormesh(x, y, dfl.int_xy(), cmap=cmap_ph, 
                  vmin=np.min(dfl.int_xy()), vmax=np.max(dfl.int_xy()))
    
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    
    ax.set_aspect('equal', adjustable='box')
    # ax.set_box_aspect(1)

    # plt.tight_layout()
    dfl.to_domain(domains_orig[0] + domains_orig[1])

    if savefig != False:
        if savefig == True:
            savefig = 'png'
        fig.savefig(filePath + fig_name + '.' + str(savefig), format=savefig)

    plt.draw()

    if showfig == True:
        # rcParams["savefig.directory"] = os.path.dirname(filePath)
        plt.show()
    else:
        plt.close(fig)   
 
    
@if_plottable
def plot_two_dfls(dfl_first, dfl_second, domains='s', label_first=None, label_second=None, title=None,
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
    
    figsize = 6
    # fig = plt.figure(fig_name)
    plt.clf()
     
    fig.set_size_inches(4*figsize, figsize, forward=True)
    if title is not None: 
        fig.suptitle(title, fontsize=18, ha='right')
        
    ax_x = fig.add_subplot(1, 4, 1)
    ax_y = fig.add_subplot(1, 4, 2)
    
    ax_x2 = ax_x.twinx()  # instantiate a second axes that shares the same x-axis
    ax_y2 = ax_y.twinx()

    _logger.info(ind_str + "domains={}".format(str(domains)))

    if domains == 'k':
        label_x = r'$\theta_y, мкрад$'
        label_y = r'$\theta_x, мкрад$'
    elif domains == 's':
        label_x = r'$x, мкм$'
        label_y = r'$y, мкм$'
    dfl1.to_domain(domains)
    dfl2.to_domain(domains)
    
    if None in [x_lim, y_lim]:
        x_lim = min(np.max(dfl1.scale_x())*1e6, np.max(dfl2.scale_x())*1e6)
        y_lim = min(np.max(dfl1.scale_y())*1e6, np.max(dfl2.scale_y())*1e6)

    _logger.debug(ind_str + "x_lim = {}, y_lim = {}".format(x_lim, y_lim))
    ax_y.set_xlim(-y_lim, y_lim)
    ax_y2.set_xlim(-y_lim, y_lim)
    ax_x.set_xlim(-x_lim, x_lim)
    ax_x2.set_xlim(-x_lim, x_lim)        
    
    ax_y.grid()
    ax_x.grid()
    
    if slice_xy is True:   
        fig.text(0.01, 0.01, 'x- y- срез', fontsize=18)
        I_1x = np.sum(dfl1.intensity(), axis=0)[dfl1.Ny()//2+1, :]
        I_2x = np.sum(dfl2.intensity(), axis=0)[dfl2.Ny()//2+1, :]
        I_1y = np.sum(dfl1.intensity(), axis=0)[:, dfl1.Nx()//2+1]
        I_2y = np.sum(dfl2.intensity(), axis=0)[:, dfl2.Nx()//2+1]
    elif slice_xy is False:
        fig.text(0.01, 0.01, 'x- y- проекция', fontsize=18)
        I_1x = np.sum(dfl1.intensity(), axis=(0,1))
        I_2x = np.sum(dfl2.intensity(), axis=(0,1))
        I_1y = np.sum(dfl1.intensity(), axis=(0,2))
        I_2y = np.sum(dfl2.intensity(), axis=(0,2))
    else: 
        raise AttributeError('slice_xy is a boolean type')
        _logger.error(ind_str + 'slice_xy is a boolean type')
       
    ax_x.plot(dfl1.scale_x()*1e6, I_1x, c='b', label=label_first)
    ax_x2.plot(dfl2.scale_x()*1e6, I_2x, c='green', label=label_second)
    
    ax_y.plot(dfl1.scale_y()*1e6, I_1y, c='b', label=label_first)
    ax_y2.plot(dfl2.scale_y()*1e6, I_2y, c='green', label=label_second)    
    
    if None not in [label_first, label_second]:
        ax_y2.legend(fontsize=12, bbox_to_anchor=(0, 0.92), loc='upper left')#, loc=1)
        ax_y.legend(fontsize=12, bbox_to_anchor=(0, 0.995), loc='upper left')#, loc=2)
        ax_x2.legend(fontsize=12, bbox_to_anchor=(0, 0.92), loc='upper left')#, loc=1)
        ax_x.legend(fontsize=12, bbox_to_anchor=(0, 0.995), loc='upper left')#, loc=2)
        
    ax_xy1 = fig.add_subplot(1, 4, 3)
    ax_xy2 = fig.add_subplot(1, 4, 4)

    if phase == True:
        xy_proj_ph1 = np.angle(np.sum(dfl1.fld, axis=0))
        xy_proj_ph2 = np.angle(np.sum(dfl2.fld, axis=0))
        ax_xy1.pcolormesh(dfl1.scale_x()*1e6, dfl1.scale_y()*1e6, xy_proj_ph1, cmap=cmap_ph, vmin=-np.pi, vmax=np.pi)        
        ax_xy2.pcolormesh(dfl2.scale_x()*1e6, dfl2.scale_y()*1e6, xy_proj_ph2, cmap=cmap_ph, vmin=-np.pi, vmax=np.pi)
    else:
        ax_xy1.pcolormesh(dfl1.scale_x()*1e6, dfl1.scale_y()*1e6, np.sum(dfl1.intensity(), axis=0))         
        ax_xy2.pcolormesh(dfl2.scale_x()*1e6, dfl2.scale_y()*1e6, np.sum(dfl2.intensity(), axis=0))

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
        _logger.debug(ind_str + 'saving *{:}.{:}'.format(figname, savefig))
        fig.savefig(filePath + figname + '.' + str(savefig), format=savefig)
    _logger.debug(ind_str + 'done in {:.2f} seconds'.format(time.time() - start_time))

    plt.draw()

    if showfig == True:
        _logger.debug(ind_str + 'showing two dfls')
        rcParams["savefig.directory"] = os.path.dirname(filePath)
        plt.show()
    else:
        plt.close(fig)   
    _logger.info(ind_str + 'plotting two dfls done in {:.2f} seconds'.format(time.time() - start_time))
    
# plot_dfl_2Dsf(dfl, scale='mm', fig_name='field')
# plot_two_dfls(dfl, dfl, phase=False, slice_xy=True)








