import scrape_USGS as sUSGS
import data_ingestion as DI 
import os 
import numpy as np
import LiCSBAS03op_GACOS as gacos
import LiCSBAS05op_clip_unw as clip
import LiCSBAS04op_mask_unw as mask 
import LiCSBAS_io_lib as LiCS_lib
import LiCSBAS_tools_lib as LiCS_tools
from lmfit.model import *
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib import pyplot as plt
import matplotlib.path as path
import obspy as obspy
import re
import scipy 
import os
import sys
import glob
import shutil
import multiprocessing as multi
import SCM
import LiCSBAS_plot_lib as plot_lib
from skimage.measure import block_reduce
import xarray as xr 
import matplotlib.pyplot as plt
import shapely 
from shapely.geometry import Polygon
import time 
import downsamp as ds 
import mosiac_images as mi
from PIL import Image
import pylab as plt 

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


def main(geoc_ml_path,output_geoc_ml_path,rad,center,new_points,argv=None,stacked=False,cov=None):
        
        if argv == None:
            argv = sys.argv

        global ifgdates2, in_dir, out_dir, length, width, bool_mask, cmap_wrap, cmap_noise, cmap_unwrap, cycle, radius, cent, nmpoints, stack_data, return_cov
        
        try:
            n_para = len(os.sched_getaffinity(0))
        except:
            n_para = multi.cpu_count()

        cmap_noise = 'viridis'
        # cmap_wrap = SCM.romaO
        cmap_wrap = 'insar'
        cmap_unwrap = 'bwr'
        q = multi.get_context('fork')

        # length, width = float_ifgm.shape 
        signal_mask = os.path.join(geoc_ml_path, 'signal_mask')
        try:
            bool_mask = np.fromfile(signal_mask,dtype=np.float32).astype(int)
        except ValueError as err:
            print("Signal mask file missing from input directory please signal mask step first: ", err)

        # print(bool_mask[0:10])
        in_dir = geoc_ml_path
        out_dir = output_geoc_ml_path
        # down_inner = step_inner
        # down_outer = step_outer
        radius = rad 
        cent = center
        nmpoints = new_points
        stack_data = stacked
        return_cov = cov


        mlipar = os.path.join(in_dir, 'slc.mli.par')
        width = int(LiCS_lib.get_param_par(mlipar, 'range_samples'))
        length = int(LiCS_lib.get_param_par(mlipar, 'azimuth_lines'))

        speed_of_light = 299792458 #m/s
        radar_frequency = float(LiCS_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
        wavelength = speed_of_light/radar_frequency #meter
        if wavelength > 0.2: ## L-band
            cycle = 1.5  # 2pi/cycle for png
        else: ## C-band
            cycle = 3  # 2pi*3/cycle for png

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
       
        EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
        width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
        length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
        dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
        dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
        lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
        lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
 
        centerlat = lat1+dlat*(length/2)
        ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
        recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
        rb = ra*(1-1/recip_f) ## polar radius
        pixsp_a = 2*np.pi*rb/360*abs(dlat)
        pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
        # print("PIXSP_A===== " + str(pixsp_a))
        # print("PIXELSP_R==== " + str(pixsp_r))
        
        Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
        Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)
      
        Lat = Lat[:length]
        Lon = Lon[:width]
      
        ifgdates = LiCS_tools.get_ifgdates(in_dir)
        n_ifg = len(ifgdates)
       
        files = glob.glob(os.path.join(in_dir, '*'))
        for file in files:
            if not os.path.isdir(file): #not copy directory, only file
                print('Copy {}'.format(os.path.basename(file)), flush=True)
                shutil.copy(file, out_dir)

          #%% Mask unw
        print('\nMask unw and link cc', flush=True)
        ### First, check if already exist
        ifgdates2 = []
        for ifgix, ifgd in enumerate(ifgdates): 
            out_dir1 = os.path.join(out_dir, ifgd)
            unwfile_m = os.path.join(out_dir1, ifgd+'.unw')
            ccfile_m = os.path.join(out_dir1, ifgd+'.cc')
            if not (os.path.exists(unwfile_m) and os.path.exists(ccfile_m)):
                ifgdates2.append(ifgd)

        if stack_data:
            n_ifg2 = 1 
        else:
            n_ifg2 = len(ifgdates2)

        if n_ifg-n_ifg2 > 0:
            print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

        if n_ifg2 > 0:
            ### Mask with parallel processing
            if n_para > n_ifg2:
                n_para = n_ifg2

                
            print('  {} parallel processing...'.format(n_para), flush=True)
            p = q.Pool(n_para)
            p.map(mask_downsampler, range(n_ifg2))
            p.close()

        print("", flush=True)

    #%% Copy other files
        # files = glob.glob(os.path.join(in_dir, '*'))
        # for file in files:
        #     if not os.path.isdir(file): #not copy directory, only file
        #         print('Copy {}'.format(os.path.basename(file)), flush=True)
        #         shutil.copy(file, output_geoc_ml_path)

        print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
        print('Output directory: {}\n'.format(os.path.relpath(out_dir)))
        image_list = []
        for ifgix, ifgd in enumerate(ifgdates2): 
            out_dir1 = os.path.join(out_dir, ifgd)
            png_regular_unw = os.path.join(out_dir1, ifgd+ '.regular_unw.png')

            if os.path.isfile(png_regular_unw):
                image_list.append(np.asarray(Image.open(png_regular_unw)))
                print(image_list)
        # dates_list_title.append(ifgd)

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

        figure.savefig(os.path.join(out_dir,'All_ifgms_easy_look_up_downsamp.png'),bbox_inches='tight')
        plt.close('all')

        
def mask_downsampler(ifgix):
    ifgd = ifgdates2[ifgix]
    if np.mod(ifgix,100) == 0:
        print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)

    
    if stack_data:
        out_dir1 = os.path.join(out_dir,'Single_Deformation_field')
        if not os.path.exists(out_dir1): os.mkdir(out_dir1)
        unwfile = os.path.join(in_dir, 'stacked_data.unw')
        dtype = np.float64
        unw = LiCS_lib.read_img(unwfile, length, width, dtype=dtype)
        pngfile_unw = os.path.join(out_dir,'ds_stacked.unw.png')
    else:
        out_dir1 = os.path.join(out_dir, ifgd)
        unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
        dtype = np.float32
        unw = LiCS_lib.read_img(unwfile, length, width, dtype=dtype)
        pngfile_unw = os.path.join(out_dir1, ifgd+'.unw.png')

    thetafile = os.path.join(in_dir,"theta.geo")
    phifile = os.path.join(in_dir,"phi.geo")

    phi = LiCS_lib.read_img(phifile, length, width)
    theta = LiCS_lib.read_img(thetafile, length, width)
    unw[unw==0] = np.nan
   

    print('at average')
    if return_cov:
        # print(return_cov)
        sill_nugget_range = return_cov[ifgd] 
        print(return_cov[ifgd])
        unw,Lon,Lat,Heading,Inc, cov, Lon_m, Lat_m = poly_average_opti(in_dir,unw,phi,theta,bool_mask,radius,cent,nmpoints,sill_nugget_range)
        # unw, Lat , Lon = LiCS_tools.invert_plane(unw, Lat, Lon)
    else:
        unw,Lon,Lat,Heading,Inc,Lon_m, Lat_m = poly_average_opti(in_dir,unw,phi,theta,bool_mask,radius,cent,nmpoints)
        # unw, Lat , Lon = LiCS_tools.invert_plane(unw, Lat, Lon)

    # Convert from rad to meters 
    # unw = -unw*(0.0555/(4*np.pi)) # ------> rad to m conversion is -lamba/4*pi posative is -LOS
     
    # print("2nd mask")
    # print("shape scaled ===== " + str(np.shape(unw)))
    
    ### Output
   
    # print("output_directory1########### " + out_dir1)
    if not os.path.exists(out_dir1): os.mkdir(out_dir1)

    # if stack_data:
    #     unw.tofile(os.path.join(out_dir,'stacked_ds.unw'))
    # else:
    unw.tofile(os.path.join(out_dir1, ifgd+'.unw'))

    if not os.path.exists(os.path.join(out_dir1, ifgd+'.cc')):
        ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
        os.symlink(os.path.relpath(ccfile, out_dir1), os.path.join(out_dir1, ifgd+'.cc'))

    ## Output png for masked unw
    
    pngfile_Inc = os.path.join(out_dir1, ifgd+ '.Inc.png')
    pngfile_Head = os.path.join(out_dir1, ifgd+ '.Head.png')
    png_regular_unw = os.path.join(out_dir1, ifgd+ '.regular_unw.png')


    # title = '{} ({}pi/cycle)'.format(ifgd, cycle*2)
    # plot_lib.make_scatter_png(Lon,Lat,np.angle(np.exp(1j*unw/cycle)*cycle), pngfile_unw, cmap_wrap, title, -np.pi, np.pi, cbar=True)
    print("done 1 ")
    plot_lib.make_scatter_png(Lon,Lat,Inc, pngfile_Inc, cmap_noise, "Incidence", cbar=True)
    print("done 2 ")
    plot_lib.make_scatter_png(Lon,Lat,Heading, pngfile_Head, cmap_noise, "Heading",cbar=True)
    print("done 3")
    plot_lib.make_scatter_png(Lon,Lat,unw, png_regular_unw, cmap_unwrap, "Displacement (radians)",cbar=True,downsamp=True)
    Frame_identifier = in_dir.split('/')[-1]

    
    npzdownout = os.path.join(os.path.join(out_dir),Frame_identifier+'_'+ifgd+'.ds_unw_Lon_Lat_Inc_Heading.npz')
    # unw = -unw/4/np.pi*0.0555 * 1000 # Times by 1000 to get into mm for kite £ double check this shit 
    # np.vstack((tp, fp)).T
    day=[737063]
    if return_cov:
        sill_nugget_range = return_cov[ifgd] 
        np.savez(npzdownout, ph_disp=unw, lonlat=np.vstack((Lon,Lat)).T,la=Inc,heading=Heading,cov=cov,day=day,lonlat_m=np.vstack((Lon_m,Lat_m)).T,sill_nugget_range=[sill_nugget_range[0],sill_nugget_range[1],sill_nugget_range[2]])
    else:
        np.savez(npzdownout, ph_disp=unw, lonlat=np.vstack((Lon,Lat)).T,la=Inc,heading=Heading,day=day, lonlat_m=np.vstack((Lon_m,Lat_m)).T)
    
    LiCS_lib.npz2mat(npzdownout)
    # os.rename(npzdownout[:-3]+'mat',os.path.join('./us6000jk0t_grond_area/data',npzdownout[:-3]+'mat'))
    # print('THIS IS THE AREA TO LOOOK JOHN 33333333333333333333333333333333333333')
    # print(npzdownout)
    # print(Frame_identifier)

    # print((npzdownout[:-3]+'mat'))
    # print(npzdownout)
  


def poly_average_opti(geoc_ml_path,unw,theta,phi,mask,radius,cent,nmpoints,sill_nugget_range):

    EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
    width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    lat2 = lat1+dlat*(length-1) # south # Remove
    lon2 = lon1+dlon*(width-1) # east # Remove
    #lon, lat = np.arange(lon1, lon2+postlon, postlon), np.arange(lat1, lat2+postlat, postlat)
    
    
    centerlat = lat1+dlat*(length/2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    pixsp_a = 2*np.pi*rb/360*abs(dlat)
    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))


    Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
    Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a) 
    Lat = Lat[:length]
    Lon = Lon[:width]
    # Lon, Lat = np.linspace(lon1, lon2, width), np.linspace(lat1, lat2, length) # Remove
    total_points = length*width
    print('TOTAL Points === ' + str(total_points))
  
    # generate coordinate mesh
    lon_grid, lat_grid = np.meshgrid(Lon, Lat)

    radius = radius*1.5
    # radius = 0.25 # Remove
    x_usgs,y_usgs = LiCS_tools.bl2xy(cent[1],cent[0],width,length,lat1,dlat,lon1,dlon)
    usgs_Lon = x_usgs * pixsp_a
    usgs_Lat = y_usgs * pixsp_r
    m_cent = [usgs_Lon,usgs_Lat]
    t = time.time()
    # usgs_Lon = cent[1] # Remove
    # usgs_Lat = cent[0] # Remove
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ max Disp before downsamp " + str(np.nanmax(unw)) + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' )
    points = ds.resample_all(unw,theta,phi, lon_grid, lat_grid,m_cent,radius,nmpoints,dlon,dlat,pixsp_a,pixsp_r,lon1,lat1)
    # Changed to output in lat long hopefully
   
    if len(points[:,2][~np.isnan(points[:,2])]) < nmpoints/2: 
        points = ds.resample_all(unw,theta,phi, lon_grid, lat_grid,m_cent,radius,nmpoints*1.5,dlon,dlat,pixsp_a,pixsp_r,lon1,lat1)
    else:
        pass

    ifgm = points[:,2]
    Lon_m = points[:,0]
    Lat_m = points[:,1]
    Inc = points[:,4]
    Head = points[:,3]
    Lon = points[:,5]
    Lat = points[:,6]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ max Disp after downsamp " + str(np.nanmax(ifgm)) + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' )

    # Lon_m = Lon_m[np.nonzero(Lon_m)]
    # Lat_m = Lat_m[np.nonzero(Lat_m)]
    # print(points[:,5])
    # print(points[:,6])

    # print("size ifgm= "+ str(np.shape(ifgm))+
    #       " size Lon= " + str(np.shape(Lon))+
    #     " size Lat= " + str(np.shape(Lat))+ 
    #     " size Inc= " + str(np.shape(Inc))+ 
    #     " size Head = " + str(np.shape(Head)))
    
    # Remove nans
    Lon = Lon[~np.isnan(ifgm)]
    Lat = Lat[~np.isnan(ifgm)]
    Lon_m = Lon_m[~np.isnan(ifgm)]
    Lat_m = Lat_m[~np.isnan(ifgm)]
    Inc = Inc[~np.isnan(ifgm)]
    Head = Head[~np.isnan(ifgm)]
    ifgm = ifgm[~np.isnan(ifgm)]
   

    # print("size ifgm= "+ str(len(ifgm))+
    #       " size Lon= " + str(len(Lon))+
    #     " size Lat= " + str(len(Lat))+ 
    #     " size Inc= " + str(len(Inc))+ 
    #     " size Head = " + str(len(Head)))
    
    print("DOWN_SAMPLED NUMBER OF POINTS ==== " + str(len(points[:,2])))
    # points = ds.resample_all(unw,theta,phi, lon_grid, lat_grid, scaler_a_outer, scaler_r_outer, scaler_a_inner,scaler_r_inner,m_cent,radius)
    t2 = time.time()
    print("time taken to downsample {:10.4f} seconds".format((t2-t)))
     # distances = np.array(list(map(list, zip(xdist, ydist))))
        # all_norm_dist = np.linalg.norm((distances-distances[:,None]),axis=-1)
        # cov=sill*np.exp(-all_norm_dist/result.best_values['r'])+result.best_values['n']*np.eye(np.shape(Lon))

  
  

    if return_cov:
        # if len(return_cov) == 3:
        cov = LiCS_tools.calc_cov(Lon_m,Lat_m,ifgm,sill_nugget_range[0],sill_nugget_range[1],sill_nugget_range[2])
        return ifgm, Lon, Lat, Head, Inc, cov, Lon_m, Lat_m
        # else:
        #     print("Error Please Enter Cov in format [sill,range,nugget]")
    else:
        return ifgm, Lon, Lat, Head, Inc, Lon_m, Lat_m 



def checker(bl,tr,p):
    points_mask = np.where((p[:,0] > bl[0]) & (p[:,0] < tr[0]) & (p[:,1] > bl[1]) &(p[:,1] < tr[1]),True,False)
    return points_mask

# %%
