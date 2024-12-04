#!/usr/bin/env python3
"""
v1.3.3 20210402 Yu Morishita, GSI

========
Overview
========
This script checks quality of unw data and identifies bad interferograms based on average coherence and coverage of the unw data. This also prepares a time series working directory (overwrite if already exists).

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png]
   - yyyymmdd_yyyymmdd.cc
 - slc.mli[.par|.png]
 - baselines (can be dummy)
 - EQA.dem_par

 Outputs in TS_GEOCml*/ :
 - info/
   - 11bad_ifg.txt    : List of bad ifgs discarded from further processing
   - 11ifg_stats.txt  : Statistics of interferograms
   - EQA.dem_par (copy)
   - slc.mli.par (copy)
 - results/
   - slc.mli[.png] (copy, if exist)
   - hgt[.png] (copy, if exist)
 - 11bad_ifg_ras/yyyymmdd_yyyymmdd.unw.png : png of bad ifgs
 - 11ifg_ras/yyyymmdd_yyyymmdd.unw.png     : png of good ifgs
 - network/network11*.png  : Figures of baseline configuration

=====
Usage
=====
LiCSBAS11_check_unw.py -d ifgdir [-t tsadir] [-c coh_thre] [-u unw_thre]

 -d  Path to the GEOCml* dir containing stack of unw data.
 -t  Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
 -c  Threshold of average coherence (Default: 0.05)
 -u  Threshold of coverage of unw data (Default: 0.3)

"""
#%% Change log
'''
v1.3.4 20240528 John Condon,
- edited this version to give custom results for automatic inversion 
bad ifgm defined on coverage of each ifgm, avrg coherence, edge of frame effects and baselines in future.

v1.3.3 20210402 Yu Morioshita, GSI
 - Treat all nan as bad ifg
 - Raise error if all ifgs are bad
v1.3.2 20201116 Yu Morioshita, GSI
 - Exit if suffix is not set
v1.3.1 20200911 Yu Morioshita, GSI
 - Change default to -c 0.05 -u 0.3
v1.3 20200703 Yu Morioshita, GSI
 - Replace problematic terms
v1.2 20200225 Yu Morishita, Uni of Leeds and GSI
 - Not output network pdf
 - Deal with cc file in uint8 format
v1.1 20191115 Yu Morishita, Uni of Leeds and GSI
 - Add hgt
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import time
import shutil
import numpy as np
import datetime as dt
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
import pylab as plt 
import glob
import multiprocessing as multi
import mosiac_images as mi
from PIL import Image
import pylab as plt 
class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%% Main
def main(argv=None,auto=None):
    global good_ifgdates, length, width, in_dir, dates_at_the_edge, clipped_dir
    #%% Check argv
    if argv == None:
        argv = sys.argv
    if auto:
        print(len(auto))
        in_dir=auto[0]
        clipped_dir = auto[1]
        out_dir=auto[2]
        cc_ifg_thre = auto[3]
        coverage = auto[4]
    
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()


   
    
    
    q = multi.get_context('fork')

    

    start = time.time()
    ver="1.3.3"; date=20210402; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    ifgdir = clipped_dir
    tsadir = out_dir
    coh_thre = cc_ifg_thre
    unw_cov_thre = coverage


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:c:u:", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-d':
                ifgdir = a
            elif o == '-t':
                tsadir = a
            elif o == '-c':
                coh_thre = float(a)
            elif o == '-u':
                unw_cov_thre = float(a)

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
                raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    print("\ncoh_thre     : {}".format(coh_thre), flush=True)
    print("unw_cov_thre : {}".format(unw_cov_thre), flush=True)


    #%% Directory setting
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_'+os.path.basename(ifgdir))

    if not os.path.exists(tsadir): os.mkdir(tsadir)

    ifg_rasdir = os.path.join(tsadir, '11ifg_ras')
    bad_ifg_rasdir = os.path.join(tsadir, '11bad_ifg_ras')

    if os.path.exists(ifg_rasdir): shutil.rmtree(ifg_rasdir)
    if os.path.exists(bad_ifg_rasdir): shutil.rmtree(bad_ifg_rasdir)
    os.mkdir(ifg_rasdir)
    os.mkdir(bad_ifg_rasdir)

    netdir = os.path.join(tsadir, 'network')
    if not os.path.exists(netdir): os.mkdir(netdir)

    infodir = os.path.join(tsadir, 'info')
    if not os.path.exists(infodir): os.mkdir(infodir)

    resultsdir = os.path.join(tsadir, 'results')
    if not os.path.exists(resultsdir): os.mkdir(resultsdir)


    #%% Read date, network information and size
    ### Get dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)

    n_ifg = len(ifgdates)
    n_im = len(imdates)

    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    print("\nSize         : {} x {}".format(width, length), flush=True)

    ### Copy dempar and mli[png|par]
    # for file in ['slc.mli.par', 'EQA.dem_par']:
    #     if os.path.exists(os.path.join(ifgdir, file)):
    #         shutil.copy(os.path.join(ifgdir, file), infodir)

    # for file in ['slc.mli', 'slc.mli.png', 'hgt', 'hgt.png']:
    #     if os.path.exists(os.path.join(ifgdir, file)):
    #         shutil.copy(os.path.join(ifgdir, file), resultsdir)


    #%% Read data
    ### Allocate memory
    n_unw = np.zeros((length, width), dtype=np.float32)
    coh_avg_ifg = []
    n_unw_ifg = []

    ### Read data and calculate
    print('\nReading unw and cc data...', flush=True)
    # First, identify valid area (n_unw>im)
    for ifgix, ifgd in enumerate(ifgdates):
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th unw to identify valid area...".format(ifgix, n_ifg), flush=True)
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)
        # print('max unwrap value')
        # print(np.nanmax(unw))
        # print(unw[~np.isnan(unw)])
        unw[unw == 0] = np.nan # Fill 0 with nan
        # print(~np.isnan(unw))
        # print('every 1000 unw value')
        # print(unw[0::1000])
        n_unw += ~np.isnan(unw) # Summing number of unnan unw

    ## Identify valid area and calc rate_cov
    print(n_im)
    bool_valid = (n_unw>=1) # JC edit changed from percentage of each ifgm that has more non nan pixels in that spot than nimages.
    n_unw_valid = bool_valid.sum()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~checker~~~~~~~~~~~~~~~~~~')
    # print('imdates=' + str(imdates))
    # print('nim=' + str(n_im))
    # print('bool_valid=' + str(bool_valid))
    # print('n_unw_valid=' + str(n_unw_valid))
    # print('n_unw=' +str(n_unw) )

    ## Read cc and unw data
    for ifgix, ifgd in enumerate(ifgdates):
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th cc and unw...".format(ifgix, n_ifg), flush=True)
        ## unw
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        unw[unw == 0] = np.nan # Fill 0 with nan
        unw[~bool_valid] = np.nan # Fill sea area with nan
        n_unw_ifg.append((~np.isnan(unw)).sum())

        ## cc
        ccfile = os.path.join(ifgdir, ifgd, ifgd+'.cc')
        if os.path.getsize(ccfile) == length*width:
            coh = io_lib.read_img(ccfile, length, width, np.uint8)
            coh = coh.astype(np.float32)/255
            coh[coh==0] = np.nan
        else:
            coh = io_lib.read_img(ccfile, length, width)

        coh_avg_ifg.append(np.nanmean(coh[bool_valid])) # Use valid area only
        

    rate_cov = np.array(n_unw_ifg)/n_unw_valid

    ## Read bperp data or dummy
    bperp_file = os.path.join(ifgdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp = np.random.random(n_im).tolist()


    #%% Identify bad ifgs, link ras and output stats information
    bad_ifgdates = []
    ixs_bad_ifgdates = []

    ### Header of stats file
    ifg_statsfile = os.path.join(infodir, '11ifg_stats.txt')
    fstats = open(ifg_statsfile, 'w')
    print('# Size: {0}({1}x{2}), n_valid: {3}'.format(width*length, width, length, n_unw_valid), file=fstats)
    print('# unw_cov_thre: {0}, coh_thre: {1}'.format(unw_cov_thre, coh_thre), file=fstats)
    print('# ifg dates         bperp   dt unw_cov  coh_av', file=fstats)

    ### Identify suffix of raster image (png, ras or bmp?)
    unwfile = os.path.join(ifgdir, ifgdates[0], ifgdates[0]+'.unw')
    if os.path.exists(unwfile+'.ras'):
        suffix = '.ras'
    elif os.path.exists(unwfile+'.bmp'):
        suffix = '.bmp'
    elif os.path.exists(unwfile+'.png'):
        suffix = '.png'
    else:
        print('\nERROR: No browse image available for {}!\n'
              .format(unwfile), file=sys.stderr)
        return 2

    for i, ifgd in enumerate(ifgdates):
        rasname = ifgdates[i]+'.unw'+suffix
        rasorg = os.path.join(ifgdir, ifgdates[i], rasname)

        if not os.path.exists(rasorg):
            print('\nERROR: No browse image {} available!\n'
                  .format(rasorg), file=sys.stderr)
            return 2

        ### For stats file
        ix_primary = imdates.index(ifgd[:8])
        ix_secondary = imdates.index(ifgd[-8:])
        bperp_ifg = bperp[ix_secondary]-bperp[ix_primary]
        mday = dt.datetime.strptime(ifgd[:8], '%Y%m%d').toordinal()
        sday = dt.datetime.strptime(ifgd[-8:], '%Y%m%d').toordinal()
        dt_ifg = sday-mday

        ### Identify bad ifgs and link ras
        if rate_cov[i] < unw_cov_thre or coh_avg_ifg[i] < coh_thre or \
           np.isnan(rate_cov[i]) or np.isnan(coh_avg_ifg[i]):
            bad_ifgdates.append(ifgdates[i])
            ixs_bad_ifgdates.append(i)
            rm_flag = '*'
            os.symlink(os.path.relpath(rasorg, bad_ifg_rasdir), os.path.join(bad_ifg_rasdir, rasname))
        else:
            os.symlink(os.path.relpath(rasorg, ifg_rasdir), os.path.join(ifg_rasdir, rasname))
            rm_flag = ''

  

        print('{0}  {1:6.1f}  {2:3}   {3:5.3f}   {4:5.3f} {5}'.format(ifgd, bperp_ifg, dt_ifg, rate_cov[i],  coh_avg_ifg[i], rm_flag), file=fstats)

    fstats.close()

    ### Output list of bad ifg
    print('\n{0}/{1} ifgs are discarded from further processing.'.format(len(bad_ifgdates), n_ifg))
    print('ifg dates        unw_cov coh_av')
    bad_ifgfile = os.path.join(infodir, '11bad_ifg.txt')
    with open(bad_ifgfile, 'w') as f:
        for i, ifgd in enumerate(bad_ifgdates):
            print('{}'.format(ifgd), file=f)
            print('{}  {:5.3f}  {:5.3f}'.format(ifgd, rate_cov[ixs_bad_ifgdates[i]],  coh_avg_ifg[ixs_bad_ifgdates[i]]), flush=True)
    f.close()
    ### Raise error if all ifgs are bad    # files = glob.glob(os.path.join(in_dir, '*'))
        # for file in files:
        #     if not os.path.isdir(file): #not copy directory, only file
        #         print('Copy {}'.format(os.path.basename(file)), flush=True)
        #         shutil.copy(file, output_geoc_ml_path)
    all_bad = False
    if len(bad_ifgdates) == n_ifg:
        all_bad = True 
      
    #     raise ValueError('All ifgs are regarded as bad!\nChange the parameters or check the input ifgs.\n')

    

    if all_bad is False:
        #%% Identify removed image and output file
        good_ifgdates = list(set(ifgdates)-set(bad_ifgdates))
        good_ifgdates.sort()
        n_ifg2 = len(good_ifgdates)
        dates_at_the_edge = []
        if n_ifg2 > 0:
            ### Mask with parallel processing
            if n_para > n_ifg2:
                n_para = n_ifg2         
                print('  {} parallel processing...'.format(n_para), flush=True)
                p = q.Pool(n_para)
                dates_at_the_edge.append(p.map(check_edge_of_frame, range(n_ifg2)))
                p.close()
    
        print("~~~~~~~~~~~~~~~~~~~list of ifgms with signal at edge~~~~~~~~~~~~~~~~~~")
        print(dates_at_the_edge)
        print("~~~~~~~~~~~~~~~~~~~list of ifgms with signal at edge~~~~~~~~~~~~~~~~~~")
        if len(dates_at_the_edge) == 0:
            good_ifgdates = list(set(good_ifgdates))
        else:
            good_ifgdates = list(set(good_ifgdates)-set(dates_at_the_edge[0]))
        good_ifgdates.sort()
        print('~~~~~~~~~~~~~~~~~~~list of good ifgms~~~~~~~~~~~~~~~')
        print(good_ifgdates)
        print('~~~~~~~~~~~~~~~~~~~list of good ifgms~~~~~~~~~~~~~~~')
        good_imdates = tools_lib.ifgdates2imdates(good_ifgdates)
        bad_imdates = list(set(imdates)-set(good_imdates))
        bad_imdates.sort()
        ### Output list of removed image
        bad_imfile = os.path.join(tsadir, 'removed_dates.txt')
        removed_dates = list(set(ifgdates)-set(good_ifgdates))
        with open(bad_imfile, 'w') as f:
            for date in removed_dates:
                f.write(date + '\n')
        f.close()

        
                

        #%% Plot network
        pngfile = os.path.join(netdir, 'network11_all.png')
        plot_lib.plot_network(ifgdates, bperp, [], pngfile)

        pngfile = os.path.join(netdir, 'network11.png')
        plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile)

        pngfile = os.path.join(netdir, 'network11_nobad.png')
        plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile, plot_bad=False)
    
        elapsed_time = time.time()-start
        hour = int(elapsed_time/3600)
        minite = int(np.mod((elapsed_time/60),60))
        sec = int(np.mod(elapsed_time,60))
        print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

        print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
        print('Output directory: {}\n'.format(os.path.relpath(tsadir)))

        for i, ifgd in enumerate(good_ifgdates):
            rasname = ifgdates[i]+'.unw'+suffix
            rasorg = os.path.join(ifgdir, ifgdates[i], rasname)
            
        # ifgdates2 = []
        print("~~~~~~~~~IFGMS PASSING COHERANCE THRESHOLD~~~~~~~~~~~")
        for ifgix, ifgd in enumerate(good_ifgdates):
                print(ifgd)
                out_dir1 = os.path.join(tsadir, ifgd)
                in_dir1 = os.path.join(in_dir,ifgd)
                shutil.copytree(os.path.join(in_dir1), os.path.join(out_dir1))
        print("~~~~~~~~~IFGMS PASSING COHERANCE THRESHOLD~~~~~~~~~~~")      

        # if len(good_ifgdates) > 0:
        #     image_list = []
        #     dates_list_title = [] 
        #     for ifgix, ifgd in enumerate(good_ifgdates): 
        #         out_dir1 = os.path.join(tsadir, ifgd)
        #         unwfile_c = os.path.join(out_dir1, ifgd+'_signal_masked.unw')
        #         if os.path.isfile(unwfile_c + '.png'):
        #             image_list.append(np.asarray(Image.open(unwfile_c + '.png')))
        #             dates_list_title.append(ifgd)

        #     if len(image_list) > 3:
        #         num_cols = 3
        #     else:
        #         num_cols = len(image_list)

        #     figure = mi.show_image_list(list_images=image_list, 
        #                 list_titles=None,
        #                 num_cols=num_cols,
        #                 figsize=(50, 50),
        #                 grid=False,
        #                 title_fontsize=10)
        #     figure.savefig(os.path.join(out_dir,'All_ifgms_easy_look_up_dates_used.png'),bbox_inches='tight')
        #     plt.close('all')
        
        # if len(removed_dates) > 0:
        #     image_list = []
        #     dates_list_title = [] 
        #     for ifgix, ifgd in enumerate(removed_dates): 
        #         unw_dir1 = os.path.join(ifgdir, ifgd)
        #         unwfile_c = os.path.join(unw_dir1, ifgd+'_signal_masked.unw')
        #         if os.path.isfile(unwfile_c +'.png'):
        #             image_list.append(np.asarray(Image.open(unwfile_c + '.png')))
        #             dates_list_title.append(ifgd)

        #     if len(image_list) > 3:
        #         num_cols = 3
        #     else:
        #         num_cols = len(image_list)

        #     figure = mi.show_image_list(list_images=image_list, 
        #                 list_titles=None,
        #                 num_cols=num_cols,
        #                 figsize=(50, 50),
        #                 grid=False,
        #                 title_fontsize=10)
        #     figure.savefig(os.path.join(out_dir,'All_ifgms_easy_look_up_dates_removed.png'),bbox_inches='tight')
        #     plt.close('all')



    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        print(file)
        if not os.path.isdir(file): #not copy directory, only file
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, tsadir)

def check_edge_of_frame(ifgix):
    ifgd = good_ifgdates[ifgix] 
    unw_path = os.path.join(os.path.join(in_dir,ifgd),ifgd+".unw") 
    ifgm = io_lib.read_img(unw_path,length,width)
    mask_sig = io_lib.read_img(os.path.join(in_dir,"signal_mask"),length,width)
    masked_pixels = np.where(mask_sig==0)
    nan_in_circle = len(np.where(np.isnan(ifgm[masked_pixels]) == True)[0])
    none_nan_in_circle = len(np.where(np.isnan(ifgm[masked_pixels]) == False)[0])
    if none_nan_in_circle == 0:
        percentage_nans = 1
    else:
        # ratio = nan_in_circle/none_nan_in_circle
        percentage_nans = nan_in_circle/(none_nan_in_circle+nan_in_circle) 
        print(percentage_nans)
    # print(nan_in_circle)
    # print(none_nan_in_circle)
    if percentage_nans >=0.75:
        bad_dates  = (ifgd)
        print('removing this date')
        print(ifgd)
        return bad_dates
    else:
        return 


#%% main
if __name__ == "__main__":
    sys.exit(main())

