import getopt
import os
import re
import sys
import glob
import shutil
import time
import numpy as np
import multiprocessing as multi
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
import mosiac_images as mi
from PIL import Image
import pylab as plt 
import SCM



def invert_for_plane(data_path,width,length):
    
    cmap_wrap = SCM.romaO
    # unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
    data = io_lib.read_img(data_path, length, width)
    plot_lib.make_im_png(data, 'Before_ramp_removed.png', cmap_wrap, 'Before_ramp_removed.png',cbar=False)
    Afit, m = tools_lib.fit2d(data,w=None,deg="2")
    data = data - Afit
    data.tofile('ramp_removed.unw')
    plot_lib.make_im_png(data, 'ramp_removed.png', cmap_wrap, 'ramp_removed.png',cbar=False)
    plot_lib.make_im_png(Afit, 'ramp.png', cmap_wrap, 'ramp.png', cbar=False)



if __name__ =='__main__':

    mlipar = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000htbb_insar_processing/GEOC_121A_08800_131419_floatml/slc.mli.par'
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    data_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000htbb_insar_processing/GEOC_121A_08800_131419_floatml/20220703_20220808/20220703_20220808.unw'
    
    invert_for_plane(data_path,width,length)
