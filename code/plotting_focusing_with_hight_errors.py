#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:48:45 2021

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
#%%
pathname = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_focusing_with_hight_errors'
dirr = 'y'
hrms=0
filePath_SERVAL_0 = pathname + '/' + dirr+'_SERVAL_radiaiton_in_focus_{}_rms'.format(hrms*1e9)
dfl_0 = read_field_file(filePath_SERVAL_0)

hrms=3e-9
filePath_SERVAL_3 = pathname + '/' + dirr+'_SERVAL_radiaiton_in_focus_{}_rms'.format(hrms*1e9)
dfl_3 = read_field_file(filePath_SERVAL_3)

hrms=6e-9
filePath_SERVAL_6 = pathname + '/' + dirr+'_SERVAL_radiaiton_in_focus_{}_rms'.format(hrms*1e9)
dfl_6 = read_field_file(filePath_SERVAL_6)

#%%
from drawing_routine import *
import matplotlib.lines as lines

dfls = [dfl_0, dfl_3, dfl_6]
dfls_labels = ['идеальное заркало', '0.3 нм', '0.6 нм']
colors = ['red', 'blue', 'green']
plot_dfls(dfls, dfls_labels, colors = colors, domains='s', scale='mm', title=None,
          x_lim=None, y_lim=None, slice_xy=True, phase=False, savefig=True, showfig=True, filePath=pathname + '/', fig_name=dirr+'_'+'SERVAL_radiaiton_in_focus', show_fig=1)

# corr_dfl_0 = dfl_xy_corr(dfl_0, norm=0)
# corr_dfl_3 = dfl_xy_corr(dfl_3, norm=0)
# corr_dfl_6 = dfl_xy_corr(dfl_6, norm=0)
# corr_dfls = [corr_dfl_0, corr_dfl_3, corr_dfl_6]
# plot_dfls(corr_dfls, dfls_labels, colors = ['red', 'orange', 'green'], domains='s', scale='mm', title=None,
#           x_lim=None, y_lim=None, slice_xy=True, phase=False, savefig=False, showfig=True, filePath=None, fig_name='coor_SERVAL_radiaiton_in_focus', show_fig=1)
#%%
# plot_dfl_2Dsf(dfl_3, scale='mm', domains='sf',
#               x_lim=None, y_lim=None, savefig=True, 
#               showfig=True, filePath=pathname + '/', fig_name=dirr+'_'+'SERVAL_radiaiton_in_focus_2d_{}_A'.format(hrms*1e9), show_fig=1)

#%%
pathname = '/home/andrei/Documents/XFEL/SERVAL/fields/far_field/3.80E-05_um_4.68E-06_um_2.50E-05_urad_2.00E-05_urad_focusing_with_hight_errors'
dirr = 'x'
hrms=0
filePath_SERVAL_0 = pathname + '/' + dirr+'_SERVAL_radiaiton_after_reflection_and_12_5_of_free_space_{}_rms'.format(hrms*1e9)
dfl_0 = read_field_file(filePath_SERVAL_0)

hrms=3e-9
filePath_SERVAL_3 = pathname + '/' + dirr+'_SERVAL_radiaiton_after_reflection_and_12_5_of_free_space_{}_rms'.format(hrms*1e9)
dfl_3 = read_field_file(filePath_SERVAL_3)

hrms=6e-9
filePath_SERVAL_6 = pathname + '/' + dirr+'_SERVAL_radiaiton_after_reflection_and_12_5_of_free_space_{}_rms'.format(hrms*1e9)
dfl_6 = read_field_file(filePath_SERVAL_6)

#%%
from drawing_routine import *
import matplotlib.lines as lines

dfls = [dfl_0, dfl_3, dfl_6]
dfls_labels = ['идеальное заркало', '0.3 нм', '0.6 нм']
colors = ['red', 'blue', 'green']
plot_dfls(dfls, dfls_labels, colors = colors, domains='s', scale='mm', title=None,
          x_lim=None, y_lim=None, slice_xy=True, phase=False, savefig=True, showfig=True, filePath=pathname + '/', fig_name=dirr+'_'+'SERVAL_radiaiton_after_reflection', show_fig=1)
 #%%
plot_dfl_2Dsf(dfl_3, scale='mm', domains='sf',
              x_lim=None, y_lim=None, savefig=True, 
              showfig=True, filePath=pathname + '/', fig_name=dirr+'_'+'SERVAL_radiaiton_after_reflection_2d_{}_A'.format(hrms*1e9), show_fig=1)
# corr_dfl_0 = dfl_xy_corr(dfl_0, norm=0)
# corr_dfl_3 = dfl_xy_corr(dfl_3, norm=0)
# corr_dfl_6 = dfl_xy_corr(dfl_6, norm=0)
# corr_dfls = [corr_dfl_0, corr_dfl_3, corr_dfl_6]

# plot_dfls(corr_dfls, dfls_labels, colors = ['red', 'orange', 'green'], domains='s', scale='mm', title=None,
#           x_lim=None, y_lim=None, slice_xy=True, phase=False, savefig=False, showfig=True, filePath=None, fig_name='coor_SERVAL_radiaiton_in_focus', show_fig=1)













