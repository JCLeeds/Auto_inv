#!/usr/bin/env python3
"""
v1.2.5 20210105 Yu Morishita, GSI

This script clips a specified rectangular area of interest from unw and cc data. The clipping can make the data size smaller and processing faster, and improve the result of Step 1-2 (loop closure). Existing files are not re-created to save time, i.e., only the newly available data will be processed. This step is optional.

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw
   - yyyymmdd_yyyymmdd.cc
 - slc.mli[.par|.png]
 - EQA.dem.par
 - Others (baselines, [E|N|U].geo, hgt[.png])
 
Outputs in GEOCml*clip/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png] (clipped)
   - yyyymmdd_yyyymmdd.cc (clipped)
 - slc.mli[.par|.png] (clipped)
 - EQA.dem.par (clipped)
 - Other files in input directory if exist (copy or clipped)
 - cliparea.txt

=====
Usage
=====
LiCSBAS05op_clip_unw.py -i in_dir -o out_dir [-r x1:x2/y1:y2] [-g lon1/lon2/lat1/lat2] [--n_para int]

 -i  Path to the GEOCml* dir containing stack of unw data.
 -o  Path to the output dir.
 -r  Range to be clipped. Index starts from 0.
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 -g  Range to be clipped in geographical coordinates (deg).
 --n_para  Number of parallel processing (Default: # of usable CPU)

"""
#%% Change log
'''
v1.2.5 20210105 Yu Morishita, GSI
 - Fill 0 by nan in unw
v1.2.4 20201119 Yu Morishita, GSI
 - Change default cmap for wrapped phase from insar to SCM.romaO
v1.2.3 20201118 Yu Morishita, GSI
 - Again Bug fix of multiprocessing
v1.2.2 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.2.1 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.2 20200909 Yu Morishita, GSI
 - Parallel processing
v1.1 20200302 Yu Morishita, Uni of Leeds and GSI
 - Bag fix for hgt.png and glob
 - Deal with cc file in uint8 format
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
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

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%%
def main(argv=None,auto=None,index_clip=False):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver="1.2.5"; date=20210105; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    ### For parallel processing
    global ifgdates2, in_dir, out_dir, length, width, x1, x2, y1, y2,cycle, cmap_wrap


    #%% Set default
    in_dir = []
    out_dir = []
    range_str = []
    range_geo_str = []
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    q = multi.get_context('fork')
    cmap_wrap = SCM.romaO

    if auto:
        in_dir = auto[0]
        out_dir = auto[1]
        if index_clip  == False:
            range_geo_str = auto[2]
        elif index_clip == True:
            range_str = auto[2]
    else:
    #%% Read options
        try:
            try:
                opts, args = getopt.getopt(argv[1:], "hi:o:r:g:", ["help", "n_para="])
            except getopt.error as msg:
                raise Usage(msg)
            for o, a in opts:
                if o == '-h' or o == '--help':
                    print(__doc__)
                    return 0
                elif o == '-i':
                    in_dir = a
                elif o == '-o':
                    out_dir = a
                elif o == '-r':
                    range_str = a
                elif o == '-g':
                    range_geo_str = a
                elif o == '--n_para':
                    n_para = int(a)

            if not in_dir:
                raise Usage('No input directory given, -i is not optional!')
            if not out_dir:
                raise Usage('No output directory given, -o is not optional!')
            if not range_str and not range_geo_str:
                raise Usage('No clip area given, use either -r or -g!')
            if range_str and range_geo_str:
                raise Usage('Both -r and -g given, use either -r or -g not both!')
            elif not os.path.isdir(in_dir):
                raise Usage('No {} dir exists!'.format(in_dir))
            elif not os.path.exists(os.path.join(in_dir, 'slc.mli.par')):
                raise Usage('No slc.mli.par file exists in {}!'.format(in_dir))

        except Usage as err:
            print("\nERROR:", file=sys.stderr, end='')
            print("  "+str(err.msg), file=sys.stderr)
            print("\nFor help, use -h or --help.\n", file=sys.stderr)
            return 2


    #%% Read info and make dir
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    ifgdates = tools_lib.get_ifgdates(in_dir)
    n_ifg = len(ifgdates)

    mlipar = os.path.join(in_dir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    if wavelength > 0.2: ## L-band
        cycle = 1.5  # 2pi/cycle for png
    else: ## C-band
        cycle = 3  # 2pi*3/cycle for png

    dempar = os.path.join(in_dir, 'EQA.dem_par')
    lat1 = float(io_lib.get_param_par(dempar, 'corner_lat')) # north
    lon1 = float(io_lib.get_param_par(dempar, 'corner_lon')) # west
    postlat = float(io_lib.get_param_par(dempar, 'post_lat')) # negative
    postlon = float(io_lib.get_param_par(dempar, 'post_lon')) # positive
    # dlat = float(LiCS_lib.get_param_par(dempar, 'post_lat')) #negative
    # dlon = float(LiCS_lib.get_param_par(dempar, 'post_lon')) #positive
    lat2 = lat1+postlat*(length-1) # south
    lon2 = lon1+postlon*(width-1) # east
    # centerlat = lat1+dlat*(length/2)
    # ra = float(LiCS_lib.get_param_par(dempar, 'ellipsoid_ra'))
    # recip_f = float(LiCS_lib.get_param_par(dempar, 'ellipsoid_reciprocal_flattening'))
    # rb = ra*(1-1/recip_f) ## polar radius
    # pixsp_a = 2*np.pi*rb/360*abs(dlat)
    # pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
    # # print("PIXSP_A===== " + str(pixsp_a))

    # Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
    # Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)
    # Lat = Lat[:length]
    # Lon = Lon[:width]
    # Lon, Lat = np.meshgrid(Lon,Lat)
    # ll = [lons.flatten(),lats.flatten()]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    #%% Check and set range to be clipped
    ### Read -r or -g option
    if range_str: ## -r
        if not tools_lib.read_range(range_str, width, length):
            print('\nERROR in {}\n'.format(range_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range(range_str, width, length)
    else: ## -g
        if not tools_lib.read_range_geo(range_geo_str, width, length, lat1, postlat, lon1, postlon):
            print('\nERROR in {}\n'.format(range_geo_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range_geo(range_geo_str, width, length, lat1, postlat, lon1, postlon)
            range_str = '{}:{}/{}:{}'.format(x1, x2, y1, y2)
        
    ### Calc clipped  info
    width_c = x2-x1
    length_c = y2-y1
    lat1_c = lat1+postlat*y1 # north
    lon1_c = lon1+postlon*x1 # west
    lat2_c = lat1_c+postlat*(length_c-1) # south
    lon2_c = lon1_c+postlon*(width_c-1) # east

    print("\nArea to be clipped:", flush=True)
    print("  0:{}/0:{} -> {}:{}/{}:{}".format(width, length, x1, x2, y1, y2))
    print("  {:.7f}/{:.7f}/{:.7f}/{:.7f} ->".format(lon1, lon2, lat2, lat1))
    print("  {:.7f}/{:.7f}/{:.7f}/{:.7f}".format(lon1_c, lon2_c, lat2_c, lat1_c))
    print("  Width/Length: {}/{} -> {}/{}".format(width, length, width_c, length_c))
    print("", flush=True)

    clipareafile = os.path.join(out_dir, 'cliparea.txt')
    with open(clipareafile, 'w') as f: f.write(range_str)
    f.close()


    #%% Make clipped par files
    mlipar_c = os.path.join(out_dir, 'slc.mli.par')
    dempar_c = os.path.join(out_dir, 'EQA.dem_par')
    
    ### slc.mli.par
    with open(mlipar, 'r') as f: file = f.read() 
    file = re.sub(r'range_samples:\s*{}'.format(width), 'range_samples: {}'.format(width_c), file)
    file = re.sub(r'azimuth_lines:\s*{}'.format(length), 'azimuth_lines: {}'.format(length_c), file)
    f.close()
    with open(mlipar_c, 'w') as f: f.write(file)
    f.close()
    ### EQA.dem_par
    with open(dempar, 'r') as f: file = f.read()
    file = re.sub(r'width:\s*{}'.format(width), 'width: {}'.format(width_c), file)
    file = re.sub(r'nlines:\s*{}'.format(length), 'nlines: {}'.format(length_c), file)
    file = re.sub(r'corner_lat:\s*{}'.format(lat1), 'corner_lat: {}'.format(lat1_c), file)
    file = re.sub(r'corner_lon:\s*{}'.format(lon1), 'corner_lon: {}'.format(lon1_c), file)
    with open(dempar_c, 'w') as f: f.write(file)
    f.close()


    #%% Clip or copy other files than unw and cc
    files = sorted(glob.glob(os.path.join(in_dir, '*')))
    for file in files:
        if os.path.isdir(file):
            continue  #not copy directory
        elif file==mlipar or file==dempar:
            continue  #not copy 
        elif os.path.getsize(file) == width*length*4: ##float file
            print('Clip {}'.format(os.path.basename(file)), flush=True)
            print('CHEKER 1 ')
            data = io_lib.read_img(file, length, width)
            data = data[y1:y2, x1:x2]
            filename = os.path.basename(file)
            outfile = os.path.join(out_dir, filename)
            data.tofile(outfile)
        elif file==os.path.join(in_dir, 'slc.mli.png'):
            print('CHEKER 2 ')
            print('Recreate slc.mli.png', flush=True)
            mli = io_lib.read_img(os.path.join(out_dir, 'slc.mli'), length_c, width_c)
            pngfile = os.path.join(out_dir, 'slc.mli.png')
            plot_lib.make_im_png(mli, pngfile, 'gray', 'MLI', cbar=False)
        elif file==os.path.join(in_dir, 'hgt.png'):
            print('Recreate hgt.png', flush=True)
            print('CHEKER 3 ')
            hgt = io_lib.read_img(os.path.join(out_dir, 'hgt'), length_c, width_c)
            vmax = np.nanpercentile(hgt, 99)
            vmin = -vmax/3 ## bnecause 1/4 of terrain is blue
            pngfile = os.path.join(out_dir, 'hgt.png')
            plot_lib.make_im_png(hgt, pngfile, 'terrain', 'DEM (m)', vmin, vmax, cbar=True)
        else:
            print('Checker 4 ')
            print(file)
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, out_dir)


    #%% Clip unw and cc
    print('\nClip unw and cc', flush=True)
    ### First, check if already exist
    ifgdates2 = []
    for ifgix, ifgd in enumerate(ifgdates): 
        out_dir1 = os.path.join(out_dir, ifgd)
        unwfile_c = os.path.join(out_dir1, ifgd+'.unw')
        ccfile_c = os.path.join(out_dir1, ifgd+'.cc')
        if not (os.path.exists(unwfile_c) and os.path.exists(ccfile_c)):
            ifgdates2.append(ifgd)

    n_ifg2 = len(ifgdates2)
    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} clipped unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    if n_ifg2 > 0:
        ### Clip with parallel processing
        if n_para > n_ifg2:
            n_para = n_ifg2
            
        print('  {} parallel processing...'.format(n_para), flush=True)
        p = q.Pool(n_para)
        p.map(clip_wrapper, range(n_ifg2))
        p.close()

    image_list = []
    dates_list_title = [] 
    # J.C addition make mosiac for frame 
    for ifgix, ifgd in enumerate(ifgdates): 
        out_dir1 = os.path.join(out_dir, ifgd)
        unwfile_c = os.path.join(out_dir1, ifgd+'.unw')
        image_list.append(np.asarray(Image.open(unwfile_c + '.png')))
        dates_list_title.append(ifgd)

    if len(image_list) > 3:
        num_cols = 3
    else:
        num_cols = len(image_list)

    figure = mi.show_image_list(list_images=image_list, 
                list_titles=None,
                num_cols=num_cols,
                figsize=(50, 50),
                grid=False,
                title_fontsize=10)
    figure.savefig(os.path.join(out_dir,'All_ifgms_easy_look_up_clipped.png'),bbox_inches='tight')
    plt.close('all')

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(out_dir)))


#%%

def clip_wrapper(ifgix):
    if np.mod(ifgix, 100) == 0:
        print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)

    ifgd = ifgdates2[ifgix]
    unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
    ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
    
    unw = io_lib.read_img(unwfile, length, width)
    unw[unw==0] = np.nan
    if os.path.getsize(ccfile) == length*width:
        ccformat = np.uint8
    elif os.path.getsize(ccfile) == length*width*4:
        ccformat = np.float32
    else:
        print("ERROR: unkown file format for {}".format(ccfile), file=sys.stderr)
        return
    coh = io_lib.read_img(ccfile, length, width, dtype=ccformat)

  
    ### Clip and remove ramp 
    unw = unw[y1:y2, x1:x2]
    coh = coh[y1:y2, x1:x2]

    # # REMOVE RAMP DOES THIS HERE SO THAT IT IS NOT NESSESARY IN THE GBIS INVERSION AND SEMIVARIAGRAM CALC DONE AFTER CLIP TO SAVE COMPUTE
    # Afit, m = tools_lib.fit2d(unw,w=None,deg="2")
    # unw = np.subtract(unw, Afit)

    ### Output
    out_dir1 = os.path.join(out_dir, ifgd)
    if not os.path.exists(out_dir1): os.mkdir(out_dir1)
    
    unw.tofile(os.path.join(out_dir1, ifgd+'.unw'))
    coh.tofile(os.path.join(out_dir1, ifgd+'.cc'))

    ## Output png for corrected unw
    pngfile = os.path.join(out_dir1, ifgd+'.unw.png')
    title = '{} ({}pi/cycle)'.format(ifgd, cycle*2)
    plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), pngfile, cmap_wrap, title, -np.pi, np.pi, cbar=False)


#%% main
if __name__ == "__main__":
    sys.exit(main())
