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

ocelog.setLevel(logging.INFO)
_logger = logging.getLogger(__name__)

#%%
def plot_dfl_2Dsf(dfl, scale='mm', domains='sf',
                  x_lim=None, y_lim=None, savefig=False, showfig=True, filePath=None, fig_name=None, show_fig=1):
    figsize=2
    fig = plt.figure(fig_name)
    plt.clf()
    fig.set_size_inches((4*figsize, 4*figsize))#, forward=True)
    
    cmap_ph = 'gray_r'
    # cmap_ph = 'viridis'
    
    ax = fig.add_subplot(111)
    
    domains_orig=dfl.domains()
    dfl.to_domain(domains)
    
    if domains=='sf':
        if scale=='mm':
            scale_order = 1e3
            xlabel = 'x, [мм]'
            ylabel = 'y, [мм]'
            x = dfl.scale_kx()*scale_order
            y = dfl.scale_ky()*scale_order
        elif scale=='um':
            scale_order= 1e6
            xlabel = 'x, [мкм]'
            ylabel = 'y, [мкм]'
            x = dfl.scale_kx()*scale_order
            y = dfl.scale_ky()*scale_order
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
        scale_order= 1e6
        x = dfl.scale_x()*scale_order
        y = dfl.scale_y()*scale_order
    # else:
    #     print('domains and scales must match each other')

    if None in [x_lim, y_lim]:
        x_lim = min(np.max(dfl.scale_x())*scale_order, np.max(dfl.scale_x())*scale_order)
        y_lim = min(np.max(dfl.scale_y())*scale_order, np.max(dfl.scale_y())*scale_order)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlim(-x_lim, x_lim)
                  
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
     
    if show_fig==1:
        plt.draw()
        plt.show()
    else:
        return fig
    
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

    cmap_ph = 'gray_r'   
    cmap_ph = plt.get_cmap('gray_r')
    
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
        ax_xy1.pcolormesh(dfl1.scale_x()*scale_order, dfl1.scale_y()*scale_order, np.sum(dfl1.intensity(), axis=0), cmap=cmap_ph)         
        ax_xy2.pcolormesh(dfl2.scale_x()*scale_order, dfl2.scale_y()*scale_order, np.sum(dfl2.intensity(), axis=0), cmap=cmap_ph)

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
        fig.savefig(filePath + fig_name + '.' + str(savefig), format=savefig)
    _logger.debug(ind_str + 'done in {:.2f} seconds'.format(time.time() - start_time))

    plt.draw()

    if showfig == True:
        _logger.debug(ind_str + 'showing two dfls')
        rcParams["savefig.directory"] = os.path.dirname(filePath)
        plt.show()
    else:
        plt.close(fig)   
    _logger.info(ind_str + 'plotting two dfls done in {:.2f} seconds'.format(time.time() - start_time))
    
def plot_dfls(dfls, dfls_labels, colors, domains='s', scale='mm', title=None, norm='unity',
            x_lim=None, y_lim=None, slice_xy=True, phase=False, savefig=False, showfig=True, filePath=None, fig_name=None, show_fig=1, **kwargs):
 
    _logger.info('Plotting two dfls in process...')
    start_time = time.time()
    fig = plt.figure(fig_name)

    cmap_ph = plt.get_cmap('hsv')
        
    figsize = 6
    # fig = plt.figure(fig_name)
    plt.clf()
     
    fig.set_size_inches(2*figsize, figsize, forward=True)
    if title is not None: 
        fig.suptitle(title, fontsize=18, ha='right')
        
    ax_x = fig.add_subplot(1, 2, 1)
    ax_y = fig.add_subplot(1, 2, 2)
    
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
        
    _logger.debug(ind_str + "x_lim = {}, y_lim = {}".format(x_lim, y_lim))
    
    if None in [x_lim, y_lim]:
        dfl = dfls[0]
        x_lim = min(np.max(dfl.scale_x())*scale_order, np.max(dfl.scale_x())*scale_order)
        y_lim = min(np.max(dfl.scale_y())*scale_order, np.max(dfl.scale_y())*scale_order)
    ax_y.set_xlim(-y_lim, y_lim)
    ax_x.set_xlim(-x_lim, x_lim)
    
    ax_y.grid()
    ax_x.grid()
    print(dfls_labels)
    for dfl, i in zip(dfls, range(len(dfls))):
        I_x = 0 
        I_y = 0
        if norm=='unity':
            I0 =  np.mean(dfl.intensity(), axis=0)
            I0 = I0/np.max(I0)
        else:    
            I0 =  np.mean(dfl.intensity()/(np.mean(dfl.intensity(), axis=(1,2))[:, np.newaxis, np.newaxis]), axis=0)
        if slice_xy is True:   
            fig.text(0.01, 0.01, 'x- y- срез', fontsize=18)
            I_x = I0[dfl.Ny()//2, :]
            I_y = I0[:, dfl.Nx()//2]
        elif slice_xy is False:
            fig.text(0.01, 0.01, 'x- y- проекция', fontsize=18)
            I_x = np.mean(I0, axis=0)
            I_y = np.mean(I0, axis=1)
            # I_x = np.mean(dfl.intensity(), axis=(0,1))
            # I_y = np.mean(dfl.intensity(), axis=(0,2))
        else: 
            raise AttributeError('slice_xy is a boolean type')
            _logger.error(ind_str + 'slice_xy is a boolean type')
        # ax_x.plot(dfl.scale_x()*scale_order, I_x/np.max(I_x), label=dfls_labels[i], c='blue')
        # ax_y.plot(dfl.scale_y()*scale_order, I_y/np.max(I_y), label=dfls_labels[i], c='blue')

        ax_x.plot(dfl.scale_x()*scale_order, I_x, label=dfls_labels[i], c=colors[i])
        ax_y.plot(dfl.scale_y()*scale_order, I_y, label=dfls_labels[i], c=colors[i])
    
    if None not in dfls_labels:
        ax_x.legend(fontsize=12, bbox_to_anchor=(0, 0.98), loc='upper left')#, loc=1)
        # ax_y.legend(fontsize=12, bbox_to_anchor=(0, 0.995), loc='upper left')#, loc=2)
        
    
    # ax_xy1.set_title(label_first, fontsize=14, color='b')      
    # ax_xy2.set_title(label_second, fontsize=14, color='green')
          
    ax_x.set_xlabel(label_x, fontsize=16)
    ax_y.set_xlabel(label_y, fontsize=16)
    ax_x.set_ylabel('пр.е', fontsize=16)
    ax_y.set_ylabel('пр.е', fontsize=16)
    ax_x.set_ylim(0)
    ax_y.set_ylim(0)

    
    # ax_x.set_box_aspect(1)
    # ax_y.set_box_aspect(1)
    ax_x.set_box_aspect(1)
    ax_y.set_box_aspect(1)
    
    plt.tight_layout()
    
    if savefig != False:
        if savefig == True:
            savefig = 'png'
        _logger.debug(ind_str + 'saving *{:}.{:}'.format(fig_name, savefig))
        fig.savefig(filePath + fig_name + '.' + str(savefig), format=savefig, dpi=300)
    _logger.debug(ind_str + 'done in {:.2f} seconds'.format(time.time() - start_time))
    
    if show_fig==1:
        plt.draw()
        plt.show()
    else:
        return fig

def plot_2D_1D(dfl_orig, domains='s', scale='mm', label_first=None, label_second=None, title=None,
            x_lim=None, y_lim=None, slice_over = 'x', slice_xy=False, phase=False, 
            savefig=False, showfig=True, filePath=None, fig_name=None, **kwargs):
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

    # cmap_ph = plt.get_cmap('hsv')
    cmap_ph = plt.get_cmap('gray_r')

    dfl = deepcopy(dfl_orig)
    
    if filePath is None:
        filePath = dfl.filePath
    
    if fig_name is None:
        if dfl.fileName() == '':
            fig = plt.figure('Radiation comparison' + domains)
        else:
            fig = plt.figure('Radiation comparison ' + domains + ' ' + dfl.fileName())
    else:
        fig = plt.figure(fig_name + domains)
    
    figsize = 12
    # fig = plt.figure(fig_name)
    plt.clf()
     
    fig.set_size_inches(figsize/2 + 0.5, figsize, forward=True)
    if title is not None: 
        fig.suptitle(title, fontsize=18, ha='right')
        
    ax_x = fig.add_subplot(2, 1, 2)    

    _logger.info(ind_str + "domains={}".format(str(domains)))

    if domains == 'k':
        if scale=='urad':
            label_x = r'$\theta_x, мкрад$'
            label_y = r'$\theta_y, мкрад$'
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
    dfl.to_domain(domains)
    
    if None in [x_lim, y_lim]:
        x_lim = min(np.max(dfl.scale_x())*scale_order, np.max(dfl.scale_x())*scale_order)
        y_lim = min(np.max(dfl.scale_y())*scale_order, np.max(dfl.scale_y())*scale_order)
        x_lim = [-x_lim, x_lim]
        y_lim = [-y_lim, y_lim]

    _logger.debug(ind_str + "x_lim = {}, y_lim = {}".format(x_lim, y_lim))
    ax_x.set_xlim(x_lim[0], x_lim[1])
    
    ax_x.grid()
    
    if slice_xy is True:   
        fig.text(0.01, 0.01, 'x- y- срез', fontsize=18)
        if slice_over == 'x':
            I_x = np.sqrt(dfl.Nx() * dfl.Ny()) * np.mean(dfl.intensity(), axis=0)[dfl.Ny()//2, :]
        elif slice_over == 'y':
            I_x = np.sqrt(dfl.Nx() * dfl.Ny()) * np.mean(dfl.intensity(), axis=0)[:, dfl.Nx()//2]
    elif slice_xy is False:
        fig.text(0.01, 0.01, 'x- y- проекция', fontsize=18)
        if slice_over == 'x':
            I_x = np.mean(dfl.intensity(), axis=(0,1))
        elif slice_over == 'y':
            I_x = np.mean(dfl.intensity(), axis=(0,2))
    else: 
        raise AttributeError('slice_xy is a boolean type')
        _logger.error(ind_str + 'slice_xy is a boolean type')

    ax_x.plot(dfl.scale_x()*scale_order, I_x, label=label_first, **kwargs)
    
    if None not in [label_first, label_second]:
        ax_y.legend(fontsize=12, bbox_to_anchor=(0, 0.995), loc='upper left')#, loc=2)
        
    ax_xy= fig.add_subplot(2, 1, 1)

    if slice_over == 'x':
        ax_xy.pcolormesh(dfl.scale_x()*scale_order, dfl.scale_y()*scale_order, np.mean(dfl.intensity(), axis=0), cmap=cmap_ph)         
        ax_xy.set_ylabel(label_y, fontsize=16)
        ax_xy.set_xlabel(label_x, fontsize=16)
        ax_x.set_xlabel(label_x, fontsize=16)
        ax_xy.set_xlim(x_lim[0], x_lim[1])
        ax_xy.set_ylim(y_lim[0], y_lim[1])
    elif slice_over == 'y':
        ax_xy.pcolormesh(dfl.scale_y()*scale_order, dfl.scale_x()*scale_order, (np.mean(dfl.intensity(), axis=0)).T, cmap=cmap_ph)         
        ax_xy.set_ylabel(label_x, fontsize=16)
        ax_xy.set_xlabel(label_y, fontsize=16)
        ax_x.set_xlabel(label_y, fontsize=16)    
        ax_xy.set_xlim(x_lim[0], x_lim[1])
        ax_xy.set_ylim(y_lim[0], y_lim[1])

    ax_xy.set_title(label_first, fontsize=14, color='b')      
          
    ax_x.set_ylabel('пр.е', fontsize=16)
    ax_x.set_ylim(0)

    ax_x.set_box_aspect(1)
    ax_xy.set_box_aspect(1)
    ax_xy.set_box_aspect(1)
    
    plt.tight_layout()
    
    if savefig != False:
        if savefig == True:
            savefig = 'png'
        _logger.debug(ind_str + 'saving *{:}.{:}'.format(fig_name, savefig))
        fig.savefig(filePath + fig_name + '.' + str(savefig), format=savefig)
    _logger.debug(ind_str + 'done in {:.2f} seconds'.format(time.time() - start_time))

    plt.draw()

    if showfig == True:
        _logger.debug(ind_str + 'showing two dfls')
        rcParams["savefig.directory"] = os.path.dirname(filePath)
        plt.show()
    else:
        plt.close(fig)   
    _logger.info(ind_str + 'plotting two dfls done in {:.2f} seconds'.format(time.time() - start_time))
    






