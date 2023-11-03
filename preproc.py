import scrape_USGS as sUSGS
import data_ingestion_legacy as DI 
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
import time 
import multiprocessing as multi 

class deformation_and_noise:
    """
    Class for processing of data_block fromed in the data_ingension stage. 
    """
    def __init__(self,event_id,
                 date_primary=20220110,
                 date_secondary=20220201,
                 frame='072A_05090_131313',
                 single_ifgm=True,
                 all_coseis=False,
                 stack=False,
                 scale_factor_mag=0.05,
                 scale_factor_depth=0.030,
                 scale_factor_clip_mag=0.05,
                 scale_factor_clip_depth=0.075,
                 coherence_mask=0.1,
                 target_down_samp=2000): 
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
        self.target_down_samp = target_down_samp

        self.event_object = sUSGS.USGS_event(self.event_id)
        self.data_block = DI.DataBlock(self.event_object)
        Flag_jasmin_down = False 
        if Flag_jasmin_down == False:
            if all_coseis == False:
                geoc_path,gacos_path = self.data_block.pull_data_frame_dates(date_primary,
                                                                        date_secondary,
                                                                        frame=frame,
                                                                        single_ifgm=single_ifgm)
            else:
                geoc_path, gacos_path = self.data_block.pull_frame_coseis()
        else:
            # While Jasmine down 
            geoc_path = "/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313"
            gacos_path = "/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GACOS_072A_05090_131313"

        self.geoc_final_path, self.sill_semi, self.nugget_semi, self.range_semi = self.run_processing_flow(geoc_path,gacos_path)
        onlyfiles = [f for f in os.listdir(self.geoc_final_path) if os.path.isfile(os.path.join(self.geoc_final_path, f))]
        matfiles = []
        for file in onlyfiles:
            if ".mat" in file:
                full_path = os.path.join(self.geoc_final_path,file)
                os.rename(full_path,os.path.join(self.event_object.Grond_insar,file.split('/')[-1]))
                # matfiles.append(file)
            else:
                pass 
        



    def run_processing_flow(self,geoc_path,gacos_path):
        geoc_ml_path = self.data_block.create_geoc_ml(geoc_path)
        # # Full mask, gacos, clip
        geoc_masked_path = self.coherence_mask(geoc_ml_path,self.coherence_mask_thresh)
        # geoc_masked_path = geoc_ml_path
        try:
            geoc_gacos_corr_path = self.apply_gacos(geoc_masked_path,gacos_path)
        except: 
            geoc_gacos_corr_path = geoc_masked_path
            print("No GACOS availble for this frame")
        geoc_clipped_path = self.usgs_clip(geoc_gacos_corr_path,
                                           scale_factor_mag=self.scale_factor_clip_mag,
                                           scale_factor_depth=self.scale_factor_clip_depth)
        geoc_masked_signal = self.signal_mask(geoc_clipped_path,
                                              scale_factor_mag=self.scale_factor_mag,
                                              scale_factor_depth=self.scale_factor_depth)
        dirs_with_ifgms, meta_file_paths = self.data_block.get_path_names(geoc_masked_signal)
        if isinstance(geoc_masked_signal,list):
            for ii in range(len(geoc_masked_signal)):
                for dir in dirs_with_ifgms[ii]:
                    print(dir)
                    try:
                        sill_semi, nugget_semi, range_semi= self.calc_semivariogram(geoc_masked_signal[ii],
                                                                                    dir,
                                                                                    signal_mask=True,
                                                                                    mask=False,
                                                                                    plot_semi=True,
                                                                                    semi_mask_thresh=30.6,
                                                                                    max_lag=150)  
                    except:
                        pass
        else:
            for ii in range(len(dirs_with_ifgms)):
                sill_semi, nugget_semi, range_semi = self.calc_semivariogram(geoc_masked_signal,
                                                                             dirs_with_ifgms[ii],
                                                                             signal_mask=True,
                                                                             mask=False,
                                                                             plot_semi=True,
                                                                             semi_mask_thresh=30.6,
                                                                             max_lag=150)
        if isinstance(geoc_masked_signal,list):
            for ii in range(len(geoc_masked_signal)):
                self.stack(geoc_masked_signal[ii])
        else:
            self.stack(geoc_masked_signal)
        starttime = time.time()
        geoc_ds_path = self.nested_uniform_down_sample(geoc_masked_signal,
                                                        self.target_down_samp,
                                                        scale_factor_mag=self.scale_factor_mag,
                                                        scale_factor_depth=self.scale_factor_depth,
                                                        stacked=self.dostack,
                                                        cov=[sill_semi,range_semi,nugget_semi])
        endtime = time.time()
        print("Time elasped on downsampling = " + str(endtime-starttime))
        return  geoc_ds_path, sill_semi, nugget_semi, range_semi


    def read_binary_img(self,path_unw,slc_mli_par_path):
        """
        Reads in ifgm into numpy array 
        """
        width = int(LiCS_lib.get_param_par(slc_mli_par_path, 'range_samples'))
        length = int(LiCS_lib.get_param_par(slc_mli_par_path, 'azimuth_lines'))
        ifgm = LiCS_lib.read_img(path_unw,length,width)
        return ifgm, length, width
    
    def coherence_mask(self,geoc_ml_path,co_thresh):
        """
        Applies coherence mask based on J. McGrath LiCSBAS step 3
        """
        if isinstance(geoc_ml_path,list):
            geoc_mask_output = []
            for ii in range(len(geoc_ml_path)):
                geoc_mask_output_tmp = geoc_ml_path[ii] + "_masked"
                mask.main(auto=[geoc_ml_path[ii],geoc_mask_output_tmp,co_thresh])
                geoc_mask_output.append(geoc_mask_output_tmp)
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
                outputdir = geoc_ml_path[ii] + "_" + "GACOS_Corrected"
                gacos.main(auto=[geoc_ml_path[ii],outputdir,gacos_path[ii]])
                geoc_gacos_corr_path.append(outputdir)
        else:
            outputdir = geoc_ml_path + "_" + "GACOS_Corrected"
            gacos.main(auto=[geoc_ml_path,outputdir,gacos_path])
            geoc_gacos_corr_path = outputdir
        return geoc_gacos_corr_path
    
    def usgs_clip(self,geoc_ml_path,scale_factor_mag=0.75,scale_factor_depth=0.05):
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
        if isinstance(geoc_ml_path,list):
            geoc_clipped_path = []
            for ii in range(len(geoc_ml_path)):
                path = geoc_ml_path[ii] + "_clipped"
                print(path)
                clip.main(auto=[geoc_ml_path[ii],path,geo_string])
                geoc_clipped_path.append(path)
        else:
            geoc_clipped_path = geoc_ml_path + "_clipped"
            clip.main(auto=[geoc_ml_path,geoc_clipped_path,geo_string])
        return  geoc_clipped_path
    
    def signal_mask(self,geoc_ml_path,scale_factor_mag=0.75,scale_factor_depth=0.05): 
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)*(113.11*10**3)
        if isinstance(geoc_ml_path,list):
            geoc_mask_signal = []
            for ii in range(len(geoc_ml_path)):
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

    def nested_uniform_down_sample(self,geoc_ml_path,nmpoints,scale_factor_mag=0.75,scale_factor_depth=0.05,stacked=False,cov=None):
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)*(113.11*10**3)
        if isinstance(geoc_ml_path,list):
            geoc_downsampled_path = []
            # mat_paths = [] 
            # npz_paths = []
            for ii in range(len(geoc_ml_path)):
                path = geoc_ml_path[ii] + "_down_sampled"
                ds.main(geoc_ml_path[ii],path,clip_width/2,cent,nmpoints,stacked=stacked,cov=cov)
                # npz_paths.append(path_to_npz)
                # mat_paths.append(mat_paths)
                geoc_downsampled_path.append(path)
            return geoc_downsampled_path
        else:
            geoc_downsampled_path = geoc_ml_path + "_down_sampled"
            ds.main(geoc_ml_path,geoc_downsampled_path,clip_width/2,cent,nmpoints,stacked=stacked,cov=cov)
            return geoc_downsampled_path

    def calc_semivariogram(self,geoc_ml_path,dates_path,mask=False,signal_mask=False,plot_semi=False,semi_mask_thresh=55.6,max_lag=100):
        outdir = geoc_ml_path
        slc_mli_par_path = os.path.join(geoc_ml_path,"slc.mli.par")
        try:
            dates_dates = dates_path.split("/")[-1].split("_")[0] +"_"+dates_path.split("/")[-1].split("_")[1]
            print(dates_dates)
        except:
            print("Date directory not in format yyyymmdd_yyymmdd") 
            print("Directory given: " + str(dates_path))
            return 
        unw_path = os.path.join(os.path.join(geoc_ml_path,dates_dates),dates_dates+".unw")
        ifgm,width_slc,length_slc = self.read_binary_img(unw_path,slc_mli_par_path)
        ifgm = -ifgm/4/np.pi*0.0555 # Added by JC to convert to meters deformation.

        EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
        width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
        length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
        dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
        dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
        lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
        lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
        

        print('\nIn geographical coordinates', flush=True)

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


        XX, YY = np.meshgrid(Lon, Lat)
        XX = XX.flatten()
        YY = YY.flatten()
        if signal_mask:
            mask_sig,width_m_scl,length_m_slc = self.read_binary_img(os.path.join(geoc_ml_path,"signal_mask"),slc_mli_par_path)
            masked_pixels = np.where(mask_sig==0)
            ifgm[masked_pixels] = np.nan
            ifgm_orig = ifgm.copy()
        else:
            ifgm_orig = ifgm.copy()

        if mask: 
            mask,width_m_scl,length_m_slc = self.read_binary_img(os.path.join(geoc_ml_path,"mask"),slc_mli_par_path)
            masked_pixels = np.where(mask==0)
            ifgm[masked_pixels] = np.nan
            ifgm_orig = ifgm.copy()
        else:
            ifgm_orig = ifgm.copy()



        # ifgm[abs(ifgm) > (semi_mask_thresh)] = np.nan
        ifgm_nan = ifgm.copy()
        ifgm_deramp = ifgm.copy()

        ifgm = ifgm.flatten()

        # Drop all nan data
        xdist = XX[~np.isnan(ifgm)]
        ydist = YY[~np.isnan(ifgm)]

        maximum_dist = np.sqrt(((np.max(xdist) - np.min(xdist)) ** 2) + ((np.max(ydist) - np.min(ydist) ** 2)))
        ifgm = ifgm[~np.isnan(ifgm)]

        

        # calc from lmfit
        mod = Model(self.spherical)
        medians = np.array([])
        bincenters = np.array([])
        stds = np.array([])

        # Find random pairings of pixels to check
        # Number of random checks
        n_pix = int(1e6)

        pix_1 = np.array([])
        pix_2 = np.array([])

        # Going to look at n_pix pairs. Only iterate 5 times. Life is short
        its = 0
         # Default Value
        while pix_1.shape[0] < n_pix and its < 5:
            its += 1
            # Create n_pix random selection of data points (Random selection with replacement)
            # Work out too many in case we need to remove duplicates
            pix_1 = np.concatenate([pix_1, np.random.choice(np.arange(ifgm.shape[0]), n_pix * 2)])
            pix_2 = np.concatenate([pix_2, np.random.choice(np.arange(ifgm.shape[0]), n_pix * 2)])

            # Find where the same pixel is selected twice
            duplicate = np.where(pix_1 == pix_2)[0]
            pix_1 = np.delete(pix_1, duplicate)
            pix_2 = np.delete(pix_2, duplicate)

            # Drop duplicate pairings
            unique_pix = np.unique(np.vstack([pix_1, pix_2]).T, axis=0)
            pix_1 = unique_pix[:, 0].astype('int')
            pix_2 = unique_pix[:, 1].astype('int')

            # Remove pixels with a seperation of more than 225 km 
            dists = np.sqrt(((xdist[pix_1] - xdist[pix_2]) ** 2) + ((ydist[pix_1] - ydist[pix_2]) ** 2))
            # # Max Lag solution to end member issue from J. McGrath
            # pix_1 = np.delete(pix_1, np.where(dists > (max_lag * 1000))[0])
            # pix_2 = np.delete(pix_2, np.where(dists > (max_lag * 1000))[0])

            # Max Dist solution to end member issue J. Condon 
            pix_1 = np.delete(pix_1, np.where(dists > (maximum_dist*0.75))[0])
            pix_2 = np.delete(pix_2, np.where(dists > (maximum_dist*0.75))[0])

        # In case of early ending
        if n_pix > len(pix_1):
            n_pix = len(pix_1)

        # Trim to n_pix, and create integer array
        pix_1 = pix_1[:n_pix].astype('int')
        pix_2 = pix_2[:n_pix].astype('int')

        # Calculate distances between random points
        dists = np.sqrt(((xdist[pix_1] - xdist[pix_2]) ** 2) + ((ydist[pix_1] - ydist[pix_2]) ** 2))
        # Calculate squared difference between random points
        vals = abs((ifgm[pix_1] - ifgm[pix_2])) ** 2

        medians, binedges = stats.binned_statistic(dists, vals, 'median', bins=100)[:-1]
        stds = stats.binned_statistic(dists, vals, 'std', bins=100)[0]
        bincenters = (binedges[0:-1] + binedges[1:]) / 2

        try:
            mod.set_param_hint('p', value=np.percentile(medians, 95))  # guess maximum variance
            mod.set_param_hint('n', value=0)  # guess 0
            mod.set_param_hint('r', value=100000)  # guess 100 km
            sigma = stds + np.power(bincenters / max(bincenters), 2)
            sigma = stds * (1 + (max(bincenters) / bincenters))
            result = mod.fit(medians, d=bincenters, weights=sigma)
        except:
            # Try smaller ranges
            n_bins = len(bincenters)
            try:
                bincenters = bincenters[:int(n_bins * 3 / 4)]
                stds = stds[:int(n_bins * 3 / 4)]
                medians = medians[:int(n_bins * 3 / 4)]
                sigma = stds + np.power(bincenters / max(bincenters), 3)
                sigma = stds * (1 + (max(bincenters) / bincenters))
                result = mod.fit(medians, d=bincenters, weights=sigma)
            except:
                try:
                    bincenters = bincenters[:int(n_bins / 2)]
                    stds = stds[:int(n_bins / 2)]
                    medians = medians[:int(n_bins / 2)]
                    sigma = stds + np.power(bincenters / max(bincenters), 3)
                    sigma = stds * (1 + (max(bincenters) / bincenters))
                    result = mod.fit(medians, d=bincenters, weights=sigma)
                except:
                    print('Ifgm  Failed to solve - setting sill to {}'.format(sill))

        try:
            # Print Sill (ie variance)
            sill = result.best_values['p']
            model_semi = (result.best_values['n'] + sill * ((3 * bincenters)/ (2 * result.best_values['r']) - 0.5*((bincenters**3) / (result.best_values['r']**3))))
            model_semi[np.where(bincenters > result.best_values['r'])[0]] = result.best_values['n'] + sill
        except:
            sill = 100
            model_semi = np.zeros(bincenters.shape) * np.nan

       
        if plot_semi:
            if not os.path.exists(os.path.join(outdir, 'semivariograms')):
                os.mkdir(os.path.join(outdir, 'semivariograms'))

            fig=plt.figure(figsize=(12,12))
            ax=fig.add_subplot(2,2,1)
            im = ax.imshow(ifgm_orig)
            plt.title('Original {}'.format(dates_dates))
            fig.colorbar(im, ax=ax)
            ax=fig.add_subplot(2,2,2)
            im = ax.imshow(ifgm_nan)
            plt.title('NaN {}'.format(dates_dates))
            fig.colorbar(im, ax=ax)
            ax=fig.add_subplot(2,2,3)
            im = ax.imshow(ifgm_deramp)
            plt.title('NaN + Deramp {}'.format(dates_dates))
            fig.colorbar(im, ax=ax)
            ax=fig.add_subplot(2,2,4)
            im = ax.scatter(bincenters, medians, c=sigma, label=dates_dates)
            ax.plot(bincenters, model_semi, label='{} model'.format(dates_dates))
            fig.colorbar(im, ax=ax)
            try:
                plt.title('Partial Sill: {:.4f}, Nugget: {:.4f}, Range: {:.4f} km'.format(sill, result.best_values['n'],result.best_values['r']/1000))
            except:
                plt.title('Semivariogram Failed')
            if sill == sill:
                plt.savefig(os.path.join(outdir, 'semivariograms', 'semivarigram{}X.png'.format(dates_dates)))
            else:
                plt.savefig(os.path.join(outdir, 'semivariograms', 'semivarigram{}.png'.format(dates_dates)))
            plt.close()

        # if np.mod(ii + 1, 10) == 0:
        #     print('\t{}/{}\tSill: {:.2f} ({:.2e}\tpairs processed in {:.1f} seconds)'.format(ii + 1, n_im, sill, n_pix, time.time() - begin_semi))
        # [X1,X2]=np.meshgrid(XX,XX)
        # [Y1,Y2]=np.meshgrid(YY,YY)
        # print('here')
        # H  = np.sqrt(((X1 - X2) ** 2) + ((Y1 - Y2) ** 2))
        # print('here')
        
        # distances = np.array(list(map(list, zip(xdist, ydist))))
        # all_norm_dist = np.linalg.norm((distances-distances[:,None]),axis=-1)
        # cov=sill*np.exp(-all_norm_dist/result.best_values['r'])+result.best_values['n']*np.eye(np.shape(Lon))
        # print('here')
        # np.savez(os.path.join(outdir,dates_dates+'/{}.sill_nugget_range_cov.npz'.format(dates_dates)))

        return sill, result.best_values['n'],result.best_values['r']
 
    def spherical(self,d, p, n, r):

        """
        Compute spherical variogram model
        @param d: 1D distance array
        @param p: partial sill
        @param n: nugget
        @param r: range
        @return: spherical variogram model
        """
        if r>d.max():
            r=d.max()-1
        return np.where(d > r, p + n, p * (3/2 * d/r - 1/2 * d**3 / r**3) + n)

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

   

if __name__ == "__main__":
    DaN = deformation_and_noise("us6000jk0t")

    # # multi.set_start_method('spawn')
    # # example event ID's us6000jk0t, us6000jqxc, us6000kynh,
    # test_event = sUSGS.USGS_event("us6000jk0t")
    # test_event.create_folder_stuct()
    # test_event.create_event_file()
 
    # #acending Turkey-Iran Boarder 
    # # test_event = sUSGS.USGS_event("us6000ldpg")
    # #Decending test
    # # test_event = sUSGS.USGS_event("us7000ki5u")
    # test_block = DI.DataBlock(test_event)
    # event_date_start = obspy.core.UTCDateTime(test_event.time_pos_depth['DateTime']) - (15*86400)
    # event_date_end = obspy.core.UTCDateTime(test_event.time_pos_depth['DateTime']) + (15*86400)
    # scale_factor_mag = 0.05 
    # scale_factor_depth = 0.030

    # scale_factor_clip_mag = 0.05
    # scale_factor_clip_depth = 0.075


    # print(event_date_start)
    # print(event_date_end)


    # # test_block.pull_frame_coseis()
    # # geoc_path, gacos_path = test_block.pull_data_frame_dates(20220110,20220201,frame="100A_05036_121313")
    # # ACENDING TEST SINGLE FRAME
    # # geoc_path,gacos_path = test_block.pull_data_frame_dates(20230108,20230201,frame="072A_05090_131313",single_ifgm=True)
    
    # geoc_path = "/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313"
    # gacos_path = "/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GACOS_072A_05090_131313"
    # # DECENDING TEST SINGLE FRAME 
    # # geoc_path,gacos_path = test_block.pull_data_frame_dates(20230716,20230728,frame="021D_05367_131313",single_ifgm=False)
    # # All Coseismic 
    # # geoc_path, gacos_path = test_block.pull_frame_coseis()
    # print(geoc_path)
    # geoc_ml_path = test_block.create_geoc_ml(geoc_path)
   
    # DaN = deformation_and_noise(test_event,test_block)

    # # # Full mask, gacos, clip
    # geoc_masked_path = DaN.coherence_mask(geoc_ml_path,0.1)
    # # geoc_masked_path = geoc_ml_path
    # try:
    #     geoc_gacos_corr_path = DaN.apply_gacos(geoc_masked_path,gacos_path)
    # except: 
    #     geoc_gacos_corr_path = geoc_masked_path
    #     print("No GACOS availble for this frame")

    # geoc_clipped_path = DaN.usgs_clip(geoc_gacos_corr_path,scale_factor_mag=scale_factor_clip_mag,scale_factor_depth=scale_factor_clip_depth)
    # geoc_masked_signal = DaN.signal_mask(geoc_clipped_path,scale_factor_mag=scale_factor_mag,scale_factor_depth=scale_factor_depth)

    # dirs_with_ifgms, meta_file_paths = test_block.get_path_names(geoc_masked_signal)
    # print(geoc_masked_signal)
    # print(dirs_with_ifgms)
    # if isinstance(geoc_masked_signal,list):
    #      for ii in range(len(geoc_masked_signal)):
    #         for dir in dirs_with_ifgms[ii]:
    #             print(dir)
    #             try:
    #                 sill_semi, nugget_semi, range_semi= DaN.calc_semivariogram(geoc_masked_signal[ii],dir,signal_mask=True,mask=False,plot_semi=True,semi_mask_thresh=30.6,max_lag=150)  
    #             except:
    #                 pass
    # else:
    #     for ii in range(len(dirs_with_ifgms)):
    #         sill_semi, nugget_semi, range_semi = DaN.calc_semivariogram(geoc_masked_signal,dirs_with_ifgms[ii],signal_mask=True,mask=False,plot_semi=True,semi_mask_thresh=30.6,max_lag=150)


    # if isinstance(geoc_masked_signal,list):
    #     for ii in range(len(geoc_masked_signal)):
    #         DaN.stack(geoc_masked_signal[ii])
    # else:
    #     DaN.stack(geoc_masked_signal)

    # starttime = time.time()
    # geoc_ds_path = DaN.nested_uniform_down_sample(geoc_masked_signal,2000,scale_factor_mag=scale_factor_mag,scale_factor_depth=scale_factor_depth,stacked=True,cov=[sill_semi,range_semi,nugget_semi])
    # endtime = time.time()
    # print("Time elasped on downsampling = " + str(endtime-starttime))


    # # No Masking, gacos, clip
    # geoc_gacos_corr_path = DaN.apply_gacos(geoc_ml_path,gacos_path)
    # geoc_clipped_path = DaN.usgs_clip(geoc_gacos_corr_path)

  
   ### to do tomorow:
    # parameter test  geoc_masked_path = DaN.coherence_mask(geoc_ml_path,0.1) 
    # parameter test semi_mask_threshold 
    # parameter test max lag 
    # fix multiple dates same frame need to save the dates somewhere and change name of top dir. 

    
    
##### preproc while jasmine down:



        
