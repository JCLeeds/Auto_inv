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
import LiCSBASJC_downsample as ds 
import LiCSBAS_plot_lib as LiCS_plot
import LiCSBAS11_check_unw as check
import time 
import multiprocessing as multi 
import forward_model as fm
import calc_semivariograms as cs 
import time 

class deformation_and_noise:
    """
    Class for processing of data_block fromed in the data_ingension stage. 
    """
    def __init__(self,event_id,
                 date_primary=20230108,
                 date_secondary=20230201,
                 frame=None,
                 single_ifgm=True,
                 all_coseis=False,
                 stack=False,
                 scale_factor_mag=0.075,
                 scale_factor_depth=0.075,
                 scale_factor_clip_mag=0.45,
                 scale_factor_clip_depth=0.0075,
                 coherence_mask=0.3,
                 min_unw_coverage=0.01, # removed
                 target_down_samp=2000,
                 inv_soft='GROND',
                 look_for_gacos=True,
                 NP=1): 
        self.event_id = event_id
        self.date_primary = date_primary
        self.date_secondary = date_secondary
        self.frame = frame 
        self.single_ifgm = single_ifgm 
        self.all_coseis = all_coseis
        self.dostack = stack
        self.scale_factor_mag = scale_factor_mag
        self.scale_factor_depth = scale_factor_depth
        self.scale_factor_clip_mag = scale_factor_clip_mag 
        self.scale_factor_clip_depth = scale_factor_clip_depth
        self.coherence_mask_thresh = coherence_mask
        self.min_unw_coverage = min_unw_coverage
        self.target_down_samp = target_down_samp
        self.inv_soft = inv_soft
        self.NP = NP
        self.geoc_ml_path = None 
        self.geoc_gacos_corr_path = None 
        self.geoc_masked_path = None
        self.geoc_clipped_path = None
        self.geoc_masked_signal = None
        self.geoc_ds_path = None 
        self.event_object = sUSGS.USGS_event(self.event_id)
        self.data_block = DI.DataBlock(self.event_object)
        Flag_jasmin_down = False
        t = time.time()
    
        if all_coseis == False:
            if frame is None: 
                print("No Frame specified using frames present in LiCS EQ catalog page")
            self.geoc_path,self.gacos_path = self.data_block.pull_data_frame_dates(date_primary,
                                                                    date_secondary,
                                                                    frame=frame,
                                                                    single_ifgm=single_ifgm)
        else:
            self.geoc_path, self.gacos_path = self.data_block.pull_frame_coseis()
        self.run_processing_flow(self.geoc_path,self.gacos_path,look_for_gacos)
        self.geoc_final_path = self.geoc_ds_path
        print(" THIS IS THE FINAL PATH BEFORE HANDING TO GBIS_run.py")
        print(self.geoc_final_path)
        self.move_final_output()
        t2 = time.time() 

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPROC COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("time taken to PREPROCESS {:10.4f} seconds".format((t2-t)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~  Moving On ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    def run_processing_flow(self,geoc_path,gacos_path,look_for_gacos):
        print("################### Looky here ####################")
        print(geoc_path)
        self.geoc_ml_path = self.data_block.create_geoc_ml(geoc_path)
   
        # # Full mask, gacos, clip
        self.geoc_masked_path = self.coherence_mask(self.geoc_ml_path,self.coherence_mask_thresh)
        # geoc_masked_path = geoc_ml_path
        if look_for_gacos is True:
            try:
                self.geoc_gacos_corr_path = self.apply_gacos(self.geoc_masked_path,gacos_path)
            except: 
                self.geoc_gacos_corr_path = self.geoc_masked_path
        else:
            self.geoc_gacos_corr_path = self.geoc_masked_path

            print("No GACOS availble for this frame")
        self.geoc_clipped_path = self.usgs_clip(self.geoc_gacos_corr_path,
                                           scale_factor_mag=self.scale_factor_clip_mag,
                                           scale_factor_depth=self.scale_factor_clip_depth)
        
        # self.geoc_QA_path = self.remove_poor_ifgms(self.geoc_clipped_path,self.coherence_mask_thresh,self.min_unw_coverage)
        
        self.forward_model(self.geoc_clipped_path,self.NP)
        self.geoc_masked_signal = self.signal_mask(self.geoc_clipped_path,
                                              scale_factor_mag=self.scale_factor_mag,
                                              scale_factor_depth=self.scale_factor_depth)
        dirs_with_ifgms, meta_file_paths = self.data_block.get_path_names(self.geoc_masked_signal)
        if isinstance(self.geoc_masked_signal,list):
            noise_dict = self.semi_variogram(self.geoc_masked_signal)
            print(noise_dict)
            for ii in range(len(self.geoc_masked_signal)):
                self.stack(self.geoc_masked_signal[ii]) 
            self.geoc_ds_path = self.nested_uniform_down_sample(self.geoc_masked_signal,
                                                                    self.target_down_samp,
                                                                    scale_factor_mag=self.scale_factor_mag,
                                                                    scale_factor_depth=self.scale_factor_depth,
                                                                    stacked=self.dostack,
                                                                    cov=noise_dict)
        else:
            noise_dict = self.semi_variogram(self.geoc_masked_signal)
            self.stack(self.geoc_masked_signal)
            starttime = time.time()
            self.geoc_ds_path = self.nested_uniform_down_sample(self.geoc_masked_signal,
                                                            self.target_down_samp,
                                                            scale_factor_mag=self.scale_factor_mag,
                                                            scale_factor_depth=self.scale_factor_depth,
                                                            stacked=self.dostack,
                                                            cov=noise_dict)
            endtime = time.time()
            print("Time elasped on downsampling = " + str(endtime-starttime))
            print(noise_dict)   
        return 


    def read_binary_img(self,path_unw,slc_mli_par_path):
        """
        Reads in ifgm into numpy array 
        """
        width = int(LiCS_lib.get_param_par(slc_mli_par_path, 'range_samples'))
        length = int(LiCS_lib.get_param_par(slc_mli_par_path, 'azimuth_lines'))
        ifgm = LiCS_lib.read_img(path_unw,length,width)
        return ifgm, length, width
    
    def remove_poor_ifgms(self,geoc_ml_path,co_thresh,coverage):
        """
        Applies ifgm checks based on LiCSBAS step 11
        """
        if isinstance(geoc_ml_path,list):
            geoc_QA_output = []
            for ii in range(len(geoc_ml_path)):
                if os.path.isdir(geoc_ml_path[ii] + "_QAed"):
                    geoc_QA_output_tmp = geoc_ml_path[ii] + "_QAed"
                    geoc_QA_output.append(geoc_QA_output_tmp)
                    print('DATA FOR ' + geoc_ml_path[ii] + "_QAed" + "   already present moving on using stored data" )
                else:
                    geoc_QA_output_tmp = geoc_ml_path[ii] + "_QAed"
                    check.main(auto=[geoc_ml_path[ii],geoc_QA_output_tmp,co_thresh,coverage])
                    geoc_QA_output.append(geoc_QA_output_tmp)
        else:
            if os.path.isdir(geoc_ml_path + "_masked"):
                geoc_QA_output = geoc_ml_path + "_masked"
                print('DATA FOR ' + geoc_ml_path + "_masked" + "   already present moving on using stored data" )
            else:
                geoc_QA_output = geoc_ml_path + "_masked"
                check.main(auto=[geoc_ml_path,geoc_QA_output,co_thresh,coverage]) 
        return geoc_QA_output
    
    def coherence_mask(self,geoc_ml_path,co_thresh):
        """
        Applies coherence mask based on J. McGrath LiCSBAS step 3
        """
        if isinstance(geoc_ml_path,list):
            geoc_mask_output = []
            for ii in range(len(geoc_ml_path)):
                if os.path.isdir(geoc_ml_path[ii] + "_masked"):
                    geoc_mask_output_tmp = geoc_ml_path[ii] + "_masked"
                    geoc_mask_output.append(geoc_mask_output_tmp)
                    print('DATA FOR ' + geoc_ml_path[ii] + "_masked" + "   already present moving on using stored data" )
                else:
                    geoc_mask_output_tmp = geoc_ml_path[ii] + "_masked"
                    mask.main(auto=[geoc_ml_path[ii],geoc_mask_output_tmp,co_thresh])
                    geoc_mask_output.append(geoc_mask_output_tmp)
        else:
            if os.path.isdir(geoc_ml_path + "_masked"):
                geoc_mask_output = geoc_ml_path + "_masked"
                print('DATA FOR ' + geoc_ml_path + "_masked" + "   already present moving on using stored data" )
            else:
                geoc_mask_output = geoc_ml_path + "_masked"
                mask.main(auto=[geoc_ml_path,geoc_mask_output,co_thresh]) 
        return geoc_mask_output

    def apply_gacos(self,geoc_ml_path,gacos_path):
        """
        Applied GACOS corrections to LiCSBAS interferograms 
        """
        if isinstance(geoc_ml_path,list):
            geoc_gacos_corr_path = []
            for ii in range(len(geoc_ml_path)):
                if os.path.isdir(geoc_ml_path[ii] + "_GACOS_Corrected"):
                    outputdir = geoc_ml_path[ii] + "_GACOS_Corrected"
                    geoc_gacos_corr_path.append(outputdir)
                    print('DATA FOR ' + geoc_ml_path[ii] + "_GACOS_Corrected" + "   already present moving on using stored data" )
                else:
                    outputdir = geoc_ml_path[ii] + "_" + "GACOS_Corrected"
                    gacos.main(auto=[geoc_ml_path[ii],outputdir,gacos_path[ii]])
                    geoc_gacos_corr_path.append(outputdir)
        else:
            if os.path.isdir(geoc_ml_path + "_GACOS_Corrected"):
                geoc_gacos_corr_path = geoc_ml_path + "_GACOS_Corrected"
                print('DATA FOR ' + geoc_ml_path + "_GACOS_Corrected" + "   already present moving on using stored data" )
            else:
                outputdir = geoc_ml_path + "_" + "GACOS_Corrected"
                gacos.main(auto=[geoc_ml_path,outputdir,gacos_path])
                geoc_gacos_corr_path = outputdir
        return geoc_gacos_corr_path
    
    def usgs_clip(self,geoc_ml_path,scale_factor_mag=0.75,scale_factor_depth=0.055):
        """
        Clips frame around USGS locations scaled inversly to depth and linearly with Mag 
        Needs changing to work in local domain not degrees 
        """
        #lon1/lon2/lat1/lat2
        print(self.event_object.time_pos_depth['Position'])
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        print(cent)
        clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)
        print(clip_width)
        lat1 = cent[0] - clip_width/2 
        lon1 = cent[1] - clip_width/2
        lat2 = cent[0] + clip_width/2 
        lon2 = cent[1] + clip_width/2 
        # geoc_clipped_path = geoc_ml_path + "_clipped"
        geo_string = str(lon1) +"/" + str(lon2) + "/" + str(lat1) + "/" + str(lat2)
        print(geo_string)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~CLIP INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Depth used = "  + str(self.event_object.time_pos_depth['Depth']))
        print("Depth scaler = " + str(scale_factor_depth) )
        print('Mag used = ' + str(self.event_object.MTdict['magnitude']))
        print("mag scaler used = " + str(scale_factor_mag) )
        print("Clip width in degrees = " + str(clip_width))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~CLIP INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if isinstance(geoc_ml_path,list):
            geoc_clipped_path = []
            for ii in range(len(geoc_ml_path)):
                if os.path.isdir(geoc_ml_path[ii] + "_clipped"):
                    path = geoc_ml_path[ii] + "_clipped"
                    print('DATA FOR ' + geoc_ml_path[ii] + "_clipped" + "   already present moving on using stored data" )
                    geoc_clipped_path.append(path)
                else:
                    path = geoc_ml_path[ii] + "_clipped"
                    # print(path)
                    clip.main(auto=[geoc_ml_path[ii],path,geo_string])
                    geoc_clipped_path.append(path)
        else:
            if os.path.isdir(geoc_ml_path + "_clipped"):
                print('DATA FOR ' + geoc_ml_path + "_clipped" + "   already present moving on using stored data" ) 
                geoc_clipped_path = geoc_ml_path + "_clipped"
            else:
                geoc_clipped_path = geoc_ml_path + "_clipped"
                clip.main(auto=[geoc_ml_path,geoc_clipped_path,geo_string])
        return  geoc_clipped_path
    
    def forward_model(self,geoc_ml_path,NP):
        mu = 3.2e10
        slip_rate=5.5e-5
        L = np.cbrt(float(self.event_object.MTdict['moment'])/(slip_rate*mu))
        if NP == 1:
            strike = float(self.event_object.strike_dip_rake['strike'][0])
            dip = float(self.event_object.strike_dip_rake['dip'][0])
            rake = float(self.event_object.strike_dip_rake['rake'][0])
        elif NP == 2:
            strike = float(self.event_object.strike_dip_rake['strike'][1])
            dip = float(self.event_object.strike_dip_rake['dip'][1])
            rake = float(self.event_object.strike_dip_rake['rake'][1])

        slip = L * slip_rate
        depth = float(self.event_object.MTdict['Depth_MT'])*1000
        width = L
        length = L
        location = [float(self.event_object.time_pos_depth['Position'][0]),
                    float(self.event_object.time_pos_depth['Position'][1])]
       
        if isinstance(geoc_ml_path,list):
            for geoc in geoc_ml_path:
                dates = [name for name in os.listdir(geoc) if os.path.isdir(os.path.join(geoc, name))]
            # dates = '20230108_20230213'
                dont_process = 0
                for jj in range(len(dates)):
                    if os.path.exists(os.path.join(os.path.join(geoc),dates[jj]+"forward_model_comp.png")):
                        print(os.path.join(os.path.join(geoc),dates[jj]+"forward_model_comp.png") + 'already made moving on')
                        dont_process += 1
                    else:
                        pass 
                if dont_process == len(dates):
                    pass 
                else:
                    fig = fm.forward_model(geoc,
                                strike,
                                dip,
                                rake,
                                slip,
                                depth,
                                length,
                                width,
                                location) 
                
        else:
            dates = [name for name in os.listdir(geoc_ml_path) if os.path.isdir(os.path.join(geoc_ml_path, name))]
            # dates = '20230108_20230213'
            dont_process = 0 
            for ii in range(len(dates)):
                if os.path.exists(os.path.join(os.path.join(geoc_ml_path),dates[ii]+"forward_model_comp.png")):
                        print(os.path.join(os.path.join(geoc_ml_path),dates[ii]+"forward_model_comp.png") + 'already made moving on')
                        dont_process += 1
                else:
                    pass 
            if dont_process == len(dates):
                pass 
            else:
                fig = fm.forward_model(geoc_ml_path,
                                    strike,
                                    dip,
                                    rake,
                                    slip,
                                    depth,
                                    length,
                                    width,
                                    location) 
                   

    def signal_mask(self,geoc_ml_path,scale_factor_mag=0.75,scale_factor_depth=0.055): 
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)*(113.11*10**3)


        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~MASK INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Depth used = "  + str(self.event_object.time_pos_depth['Depth']))
        print("Depth scaler = " + str(scale_factor_depth) )
        print('Mag used = ' + str(self.event_object.MTdict['magnitude']))
        print("mag scaler used = " + str(scale_factor_mag) )
        print("Mask Diameter in meters = " + str(clip_width))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~MASK INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if isinstance(geoc_ml_path,list):
            geoc_mask_signal = []
            for ii in range(len(geoc_ml_path)):

                if os.path.isdir(geoc_ml_path[ii] + "_signal_masked"):
                    geoc_mask_signal_tmp = geoc_ml_path[ii] + "_signal_masked"
                    geoc_mask_signal.append(geoc_mask_signal_tmp)
                    print('DATA FOR ' + geoc_ml_path[ii] + "_signal_masked" + "   already present moving on using stored data" ) 
                else:
                    EQA_dem_par = os.path.join(geoc_ml_path[ii],"EQA.dem_par")
                    width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
                    length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
                    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
                    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
                    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
                    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
                    #lon, lat, width, length, lat1, postlat, lon1, postlon
                    x_usgs,y_usgs = LiCS_tools.bl2xy(cent[1],cent[0],width,length,lat1,dlat,lon1,dlon)
                    # Generate circle mask around USGS_point for tuesday
                    print(x_usgs)
                    print(y_usgs)

                    centerlat = lat1+dlat*(length/2)
                    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
                    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
                    rb = ra*(1-1/recip_f) ## polar radius
                    pixsp_a = 2*np.pi*rb/360*abs(dlat)
                    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
                    Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
                    Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)
                    usgs_Lon = x_usgs * pixsp_a
                    usgs_Lat = y_usgs * pixsp_r
                    Lat = Lat[:length]
                    Lon = Lon[:width]
                    lon_grid,lat_grid = np.meshgrid(Lon,Lat)
                    poly = path.Path.circle(center=(usgs_Lon,usgs_Lat),radius=clip_width/2)
                    poly_mask = poly.contains_points(np.transpose([lon_grid.flatten(), lat_grid.flatten()])).reshape(np.shape(lon_grid))
                    True_mask = np.where(poly_mask)[0]
                
                    geoc_mask_signal_tmp = geoc_ml_path[ii] + "_signal_masked"
                    mask.main(auto=[geoc_ml_path[ii],geoc_mask_signal_tmp,False,poly_mask])
                    geoc_mask_signal.append(geoc_mask_signal_tmp)
        else:
            if os.path.isdir(geoc_ml_path + "_signal_masked"):
                geoc_mask_signal = geoc_ml_path + "_signal_masked"
                # geoc_mask_signal.append(geoc_mask_signal_tmp)
                print('DATA FOR ' + geoc_ml_path + "_signal_masked" + "   already present moving on using stored data" )
            else:
                EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
                width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
                length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
                dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
                dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
                lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
                lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
                #lon, lat, width, length, lat1, postlat, lon1, postlon
                x_usgs,y_usgs = LiCS_tools.bl2xy(cent[1],cent[0],width,length,lat1,dlat,lon1,dlon)
                # Generate circle mask around USGS_point for tuesday
                print(x_usgs)
                print(y_usgs)
                centerlat = lat1+dlat*(length/2)
                ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
                recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
                rb = ra*(1-1/recip_f) ## polar radius
                pixsp_a = 2*np.pi*rb/360*abs(dlat)
                pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
                Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
                Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)
                usgs_Lon = x_usgs * pixsp_a
                usgs_Lat = y_usgs * pixsp_r
                Lat = Lat[:length]
                Lon = Lon[:width]
                lon_grid,lat_grid = np.meshgrid(Lon,Lat)
                poly = path.Path.circle(center=(usgs_Lon,usgs_Lat),radius=clip_width/2)
                poly_mask = poly.contains_points(np.transpose([lon_grid.flatten(), lat_grid.flatten()])).reshape(np.shape(lon_grid))
                True_mask = np.where(poly_mask)[0]
                geoc_mask_signal = geoc_ml_path + "_signal_masked"
                mask.main(auto=[geoc_ml_path,geoc_mask_signal,False,poly_mask]) 
    
        return geoc_mask_signal

    def semi_variogram(self,geoc_ml_path):
        if isinstance(geoc_ml_path,list):
            final_dict = {}
            process = 0
            for ii in range(len(geoc_ml_path)):
                    # dates = LiCS_tools.get_ifgdates(geoc_ml_path)
                    output_dict = cs.calculate_semivarigrams(geoc_ml_path[ii])
                    final_dict = final_dict | output_dict
        else:
                    final_dict = cs.calculate_semivarigrams(geoc_ml_path)


        return final_dict

    def nested_uniform_down_sample(self,geoc_ml_path,nmpoints,scale_factor_mag=0.75,scale_factor_depth=0.055,stacked=False,cov=None):
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)*(113.11*10**3)
        if isinstance(geoc_ml_path,list):
            geoc_downsampled_path = []
            # mat_paths = [] 
            # npz_paths = []
            for ii in range(len(geoc_ml_path)):
                if os.path.isdir(geoc_ml_path[ii] + "_down_sampled"):
                    path = geoc_ml_path[ii] + "_down_sampled"
                    print('DATA FOR ' + geoc_ml_path[ii] + "_down_sampled" + "   already present moving on using stored data")
                    geoc_downsampled_path.append(path)
                else:
                    path = geoc_ml_path[ii] + "_down_sampled"
                    ds.main(geoc_ml_path[ii],path,clip_width/2,cent,nmpoints,stacked=stacked,cov=cov)
                    geoc_downsampled_path.append(path)

            return geoc_downsampled_path
        else:
            if os.path.isdir(geoc_ml_path + "_down_sampled"):
                geoc_downsampled_path = geoc_ml_path + "_down_sampled"
                print('DATA FOR ' + geoc_ml_path + "_down_sampled" + "   already present moving on using stored data")
            else:
                geoc_downsampled_path = geoc_ml_path + "_down_sampled"
                ds.main(geoc_ml_path,geoc_downsampled_path,clip_width/2,cent,nmpoints,stacked=stacked,cov=cov)
            return geoc_downsampled_path

    def stack(self,geoc_ml_path):

        EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
        width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
        length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
        dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
        dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
        lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
        lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
        #lon, lat, width, length, lat1, postlat, lon1, postlon
        
        # Generate circle mask around USGS_point for tuesday
        centerlat = lat1+dlat*(length/2)
        ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
        recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
        rb = ra*(1-1/recip_f) ## polar radius
        pixsp_a = 2*np.pi*rb/360*abs(dlat)
        pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
        Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
        Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)

        stacked_data = LiCS_lib.stack_ifgms(geoc_ml_path,length,width)
        outfile = os.path.join(geoc_ml_path, 'stacked_data.unw')
        stacked_data.tofile(outfile)
        print("stacked_data to file dims ==== " + str(np.shape(stacked_data)))
        pngfile = os.path.join(geoc_ml_path,"stacked.png")
        LiCS_plot.make_im_png(np.angle(np.exp(1j*stacked_data/3)*3), pngfile, 'insar', 'stacked.unw', vmin=-np.pi, vmax=np.pi, cbar=False)
        
        return 

    def move_final_output(self):
        if isinstance(self.geoc_final_path,list):
            for geoc in self.geoc_final_path:
                onlyfiles = [f for f in os.listdir(geoc) if os.path.isfile(os.path.join(geoc, f))]
                matfiles = []
                npz_files = [] 
                if self.inv_soft == 'GROND':
                    for file in onlyfiles:
                        if ".mat" in file:
                            full_path = os.path.join(geoc,file)
                            os.rename(full_path,os.path.join(self.event_object.Grond_insar,file.split('/')[-1]))
                            # matfiles.append(file)
                        else:
                            pass 
                elif self.inv_soft == "GBIS":
                    for file in onlyfiles:
                        if ".npz" in file:
                            full_path = os.path.join(geoc,file)
                            os.rename(full_path,os.path.join(self.event_object.GBIS_location,file.split('/')[-1]))
                        else:
                            pass 
            
            return 
        else:
            onlyfiles = [f for f in os.listdir(self.geoc_final_path) if os.path.isfile(os.path.join(self.geoc_final_path, f))]
            matfiles = []
            npz_files = [] 
            if self.inv_soft == 'GROND':
                for file in onlyfiles:
                    if ".mat" in file:
                        full_path = os.path.join(self.geoc_final_path,file)
                        os.rename(full_path,os.path.join(self.event_object.Grond_insar,file.split('/')[-1]))
                        # matfiles.append(file)
                    else:
                        pass 
            elif self.inv_soft == "GBIS":
                for file in onlyfiles:
                    if ".npz" in file:
                        full_path = os.path.join(self.geoc_final_path,file)
                        os.rename(full_path,os.path.join(self.event_object.GBIS_location,file.split('/')[-1]))
                    else:
                        pass 
            return 

if __name__ == "__main__":
    DaN = deformation_and_noise("us6000jk0t")

        
