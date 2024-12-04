import scrape_USGS as sUSGS
import data_ingestion as DI 
import os 
import numpy as np
import glob
import LiCSBAS02_ml_prep as LiCS_ml 
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
import shutil
import timeout_decorator
import Initial_location_plot as ILP
from difflib import SequenceMatcher


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
                 min_unw_coverage=0.3, # removed
                 target_down_samp=2000,
                 inv_soft='GBIS',
                 look_for_gacos=True,
                #  NP=1,
                 pygmt=True,
                 loop_processing_flow=True): 
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
        # self.NP = NP
        self.geoc_ml_path = None 
        self.geoc_gacos_corr_path = None 
        self.geoc_masked_path = None
        self.geoc_clipped_path = None
        self.geoc_masked_signal = None
        self.geoc_ds_path = None 
        self.geoc_QA_path = None 
        self.geoc_final_path = None
        self.pygmt = pygmt
        self.event_object = sUSGS.USGS_event(self.event_id)
        self.data_block = DI.DataBlock(self.event_object)
        self.loop_processing_flow = loop_processing_flow
        self.geoc_path = None 
        self.gacos_path = None 
        self.top_level_dir = os.getcwd()
        Flag_jasmin_down = False
        t = time.time()

        # TO KNOW diameter of mask set at 2 times 0.01m deformation, and clip set at 5 * diameter mask. 
        
        # old_stdout = sys.stdout
        # log_file = open("auto_GBIS.log","w")
        # sys.stdout = log_file
    
        if all_coseis == False:
            if frame is None: 
                print("No Frame specified using frames present in LiCS EQ catalog page")
            self.geoc_path,self.gacos_path = self.data_block.pull_data_frame_dates(date_primary,
                                                                    date_secondary,
                                                                    frame=frame,
                                                                    single_ifgm=single_ifgm)
        else:
            
            pull_attempt = 0
            if self.loop_processing_flow == True:
                while pull_attempt < 4:
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PULL ATTEMPT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" + str(pull_attempt))
                    try:
                        self.geoc_path, self.gacos_path = self.data_block.pull_frame_coseis()
                        pull_attempt = 5 
                    except Exception as e:
                        print(e)
                        self.flush_all_processing()
                        pull_attempt +=1
                        if pull_attempt > 4:
                            print('I cant seem to pull this data?')
            else:
                self.geoc_path, self.gacos_path = self.data_block.pull_frame_coseis()
        
        # try:
      
        # except:
        #     pass

        print('INITIAL DATA PULL HAS BEEN COMPLETED')
        self.check_data_pull()
        self.geoc_path = self.check_geoc_has_data(self.geoc_path) # gacos path editing is handled in function dirty but works. only if you use gacos functianlity here
        attempt = 0 

        if self.loop_processing_flow == True:
            while attempt < 4: # incase data is pulled incorrectly or copied wrong
                try: 
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS ATTEMPT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" + str(attempt))
                    edge = self.run_processing_flow(look_for_gacos)
                    attempt = 5
                except Exception as e:
                    self.flush_all_processing()
                    # self.flush_processing()
                    self.geoc_path, self.gacos_path = self.data_block.pull_frame_coseis()
                    self.check_data_pull()
                    self.geoc_path = self.check_geoc_has_data(self.geoc_path)
                    attempt+=1
                    print(e)
                    print('processing flow failed trying again with attempt number ' + str(attempt))
                    if attempt > 4:
                        print('Failed 5 times on processing flow check errors')
        else:
            edge = self.run_processing_flow(look_for_gacos)
        print('FINAL PROCESSING HAS FINISHED')
                
        self.geoc_final_path = self.geoc_ds_path
        print(" THIS IS THE FINAL PATH BEFORE HANDING TO GBIS_run.py")
        if os.path.exists(os.path.join(self.event_object.LiCS_locations,'frames_sent_to_gbis.txt')):
            os.remove(os.path.join(self.event_object.LiCS_locations,'frames_sent_to_gbis.txt'))
        with open(os.path.join(self.event_object.LiCS_locations,'frames_sent_to_gbis.txt'), 'w') as file:
            for ii in range(len(self.geoc_final_path)):
                file.writelines(self.geoc_final_path[ii].split('/')[-1] + '\n')
        file.close()        

        print(self.geoc_final_path)
        self.move_final_output()
        t2 = time.time() 

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPROC COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("time taken to PREPROCESS {:10.4f} seconds".format((t2-t)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~          ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # old_stdout = sys.stdout
        # log_file = open("auto_GBIS.log","w")
        # sys.stdout = log_file
       
    
    def run_processing_flow(self,look_for_gacos,good_location=False):
        if good_location == True:
            self.event_object.diameter_mask_in_m = self.event_object.diameter_mask_in_m/1.75
        print("################### HERE WE GO! ####################")
        print(self.geoc_path)
        print("################### CONVERTING LICS TIFS TO BINARY! ####################")
        ml_attempt = 0 # adding in due to error in pull causing this to break in 1/10
        try:
           edge_frames = self.plot_location_and_frames()
        #    self.data_block.pull_data_frame_dates(startdate,enddate,frame=None,single_ifgm=False)
        except Exception as e:
            print(e)
            
        self.geoc_ml_path = self.create_geoc_ml()
        ml_attempt = 10 
   
            # self.geoc_ml_path = self.data_block.create_geoc_ml(self.geoc_path)

        print("################### FINISHED CONVERTING LICS TIFS TO BINARY! ####################")
   
        # # Full mask, gacos, clip
        print("################### LETS APPLY THAT COHERENCE MASK SHALL WE? ####################")
        self.geoc_masked_path = self.coherence_mask(self.geoc_ml_path,self.coherence_mask_thresh)
        print("################### ALL NICE AND MASKED ####################")
        # geoc_masked_path = geoc_ml_path
        print("################### APPLY GACOS CORRECTOS ####################")
        if look_for_gacos is True:
            try:
                self.geoc_gacos_corr_path = self.apply_gacos(self.geoc_masked_path,self.gacos_path)
            except Exception as e:
                print(e) 
                self.geoc_gacos_corr_path = self.geoc_masked_path.copy()
                print('GACOS WAS NOT APPLIED')

        print('####################BEGIN CLIPPING#################################')    
        self.geoc_clipped_path = self.usgs_clip(self.geoc_gacos_corr_path,
                                              scale_factor_mag=self.scale_factor_mag,
                                              scale_factor_depth=self.scale_factor_depth)    
        print("################### CLIPPING COMPLETE  ####################")
        # self.geoc_QA_path = self.remove_poor_ifgms(self.geoc_clipped_path,self.coherence_mask_thresh,self.min_unw_coverage)

     

        print("################### APPLYING SIGNAL MASK  ####################")
        self.geoc_masked_signal = self.signal_mask(self.geoc_clipped_path,
                                              scale_factor_mag=self.scale_factor_mag,
                                              scale_factor_depth=self.scale_factor_depth)
        print("################### FINISHED APPLYING SIGNAL MASK  ####################")
        print("################### QC COHERENCE, COVERANGE AND EDGE CHECKS  ####################")
        self.geoc_QA_path = self.remove_poor_ifgms(self.geoc_masked_signal,self.geoc_clipped_path,self.coherence_mask_thresh,self.min_unw_coverage)
        self.geoc_QA_path =  self.check_geoc_has_data(self.geoc_QA_path,gacos=False)
        dirs_with_ifgms, meta_file_paths = self.data_block.get_path_names(self.geoc_QA_path)
        if self.pygmt is True:
            print("################### LETS SEE A USGS FORWARD MODEL  ####################")  
            self.forward_model(self.geoc_QA_path,1)
            self.forward_model(self.geoc_QA_path,2)
        print(len(self.geoc_QA_path))
        if len(self.geoc_QA_path) == 0:
            print('EQ is at the edge of ALL frames as defined by 75\% of the mask being NaN values, terminating process')
            return  
            # raise Exception("EQ is at the edge of ALL frames as defined by 75\% of the mask being NaN values, terminating process") 
        else:
            print("################### QC COHERENCE, COVERANGE AND EDGE CHECKS FINISHED  ####################")
            print("################### LETS SEMIVARY THAT GRAM  ####################")
            if isinstance(self.geoc_QA_path,list):
                noise_dict = self.semi_variogram(self.geoc_QA_path)
                print(noise_dict)
                print("################### FINISHED SEMIVARYING THAT GRAM  ####################")
                for ii in range(len(self.geoc_masked_signal)):
                    try:
                        self.stack(self.geoc_masked_signal[ii])
                    except:
                        pass
                print("################### START CIRC DOWN SAMP  ####################")
                starttime = time.time()
                self.geoc_ds_path = self.nested_uniform_down_sample(self.geoc_QA_path,
                                                                        self.target_down_samp,
                                                                        scale_factor_mag=self.scale_factor_mag,
                                                                        scale_factor_depth=self.scale_factor_depth,
                                                                        stacked=self.dostack,
                                                                        cov=noise_dict)
                endtime = time.time()
                print("Time elasped on downsampling = " + str(endtime-starttime))                                                   
                print("################### FINISH CIRC DOWN SAMP  ####################")
                print(self.geoc_ds_path)                                                      
            else:
                print("################### QC STEPS WITH EDGE DETECTION  ####################")
                self.geoc_QA_path = self.remove_poor_ifgms(self.geoc_masked_signal,self.coherence_mask_thresh,self.min_unw_coverage)
                self.geoc_QA_path =  self.check_geoc_has_data(self.geoc_QA_path,gacos=False)
                dirs_with_ifgms, meta_file_paths = self.data_block.get_path_names(self.geoc_QA_path)
                if len(self.geoc_QA_path) == 0:
                    raise Exception("EQ is at the edge of ALL frames as defined by 75\% of the mask being NaN values, terminating process") 
                print("################### QC STEPS WITH EDGE DETECTION FINISHED  ####################")
                print("################### LETS SEMIVARI THAT GRAM  ####################")
                noise_dict = self.semi_variogram(self.geoc_QA_PATH)
                print("################### FINISHED SEMIVARING THAT GRAM  ####################")
                self.stack(self.geoc_masked_signal)
                starttime = time.time()
                print("################### START CIRC DOWN SAMP  ####################")
                self.geoc_ds_path = self.nested_uniform_down_sample(self.geoc_QA_path,
                                                                self.target_down_samp,
                                                                scale_factor_mag=self.scale_factor_mag,
                                                                scale_factor_depth=self.scale_factor_depth,
                                                                stacked=self.dostack,
                                                                cov=noise_dict)
                endtime = time.time()
                print("Time elasped on downsampling = " + str(endtime-starttime))
                print("################### FINISH CIRC DOWN SAMP  ####################")
            return

    def create_geoc_ml(self):
        """
        Converting TIFS to Float32 using LiCS step two
        """
        if isinstance(self.geoc_path,list):
            geoc_ml_path = []
            for ii in range(len(self.geoc_path)):
                output_geoc_gacos = str(self.geoc_path[ii]+"_floatml")
                geoc_ml_path.append(output_geoc_gacos)
                if os.path.isdir(output_geoc_gacos): 
                    shutil.rmtree(output_geoc_gacos)
                    print(output_geoc_gacos + '  path in use, removing data')
                else:
                    LiCS_ml.main(auto=[self.geoc_path[ii],output_geoc_gacos])

        else:
            output_geoc_gacos = str(self.geoc_path+"_floatml")
            if os.path.isdir(output_geoc_gacos): 
                print(output_geoc_gacos + '  path in use using data already populated')
            else:
                LiCS_ml.main(auto=[self.geoc_path,output_geoc_gacos])
            geoc_ml_path = output_geoc_gacos

        return geoc_ml_path
    def read_binary_img(self,path_unw,slc_mli_par_path):
        """
        Reads in ifgm into numpy array 
        """
        width = int(LiCS_lib.get_param_par(slc_mli_par_path, 'range_samples'))
        length = int(LiCS_lib.get_param_par(slc_mli_par_path, 'azimuth_lines'))
        ifgm = LiCS_lib.read_img(path_unw,length,width)
        return ifgm, length, width
    def plot_location_and_frames(self):
        poly_list = [] 
        for geoc in self.geoc_path:
            print(geoc)
            poly_list.append(glob.glob(geoc+'/*-poly.txt'))  
        ILP.location_plot(self.event_object.event_file_path,poly_list,os.path.join(self.event_object.GBIS_location,'location_and_active_frames_plot.png'))
    def remove_poor_ifgms(self,geoc_ml_path,geoc_clipped_path,co_thresh,coverage):
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
                    check.main(auto=[geoc_ml_path[ii],geoc_clipped_path[ii],geoc_QA_output_tmp,co_thresh,coverage])
                    geoc_QA_output.append(geoc_QA_output_tmp)
        else:
            if os.path.isdir(geoc_ml_path + "_QAed"):
                geoc_QA_output = geoc_ml_path + "_QAed"
                print('DATA FOR ' + geoc_ml_path + "_QAed" + "   already present moving on using stored data" )
            else:
                geoc_QA_output = geoc_ml_path + "_QAed"
                check.main(auto=[geoc_ml_path,geoc_clipped_path,geoc_QA_output,co_thresh,coverage]) 
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
    @timeout_decorator.timeout(4600)
    def apply_gacos(self,geoc_ml_path,gacos_path):
        """
        Applied GACOS corrections to LiCSBAS interferograms 
        """
        if isinstance(geoc_ml_path,list):
            geoc_gacos_corr_path = []
            for ii in range(len(geoc_ml_path)):
                if os.path.isdir(geoc_ml_path[ii] + "_GACOS_Corrected"):
                    dir = os.listdir(gacos_path[ii]) 
                    if len(dir) == 0: 
                        #checker to not do correction if gacos folder empty
                        outputdir = geoc_ml_path[ii] 
                    else: 
                        outputdir = geoc_ml_path[ii] + "_GACOS_Corrected"

                    geoc_gacos_corr_path.append(outputdir)
                    print('DATA FOR ' + geoc_ml_path[ii] + "_GACOS_Corrected" + "   already present moving on using stored data" )
                else:
                    outputdir = geoc_ml_path[ii] + "_" + "GACOS_Corrected"
                    dir = os.listdir(gacos_path[ii]) 
                    # Checking if the list is empty or not 
                    if len(dir) == 0: 
                        outputdir = geoc_ml_path[ii] 
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
    @timeout_decorator.timeout(4600)
    def usgs_clip(self,geoc_ml_path,scale_factor_mag=0.75,scale_factor_depth=0.055):
        """
        Clips frame around USGS locations scaled inversly to depth and linearly with Mag 
        Needs changing to work in local domain not degrees 
        """
        #lon1/lon2/lat1/lat2
        print(self.event_object.time_pos_depth['Position'])
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        print(cent)
        # clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)
        clip_width = (self.event_object.diameter_mask_in_m * 3)/(111.13 * 1e3) # Not tested

        print(clip_width)
        lat1 = cent[0] - clip_width/2 
        lon1 = cent[1] - clip_width/2
        lat2 = cent[0] + clip_width/2 
        lon2 = cent[1] + clip_width/2 
        # if lat1 > np.min(lat)
        # geoc_clipped_path = geoc_ml_path + "_clipped"
        geo_string = str(lon1) +"/" + str(lon2) + "/" + str(lat1) + "/" + str(lat2)
        print(geo_string)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~CLIP INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Depth used = "  + str(self.event_object.time_pos_depth['Depth']))
        # print("Depth scaler = " + str(scale_factor_depth) )
        print('Mag used = ' + str(self.event_object.MTdict['magnitude']))
        # print("mag scaler used = " + str(scale_factor_mag) )
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
    
    @timeout_decorator.timeout(4600)
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
        # depth = float(self.event_object.time_pos_depth['Depth'])
        width = L
        length = L
        # if self.event_object.time_pos_depth['Position_USGS']:
        #     location = [float(self.event_object.time_pos_depth['Position_USGS'][0]),
        #             float(self.event_object.time_pos_depth['Position_USGS'][1])]
        # else:
        location = [float(self.event_object.time_pos_depth['Position'][0]),
                        float(self.event_object.time_pos_depth['Position'][1])]
       
        if isinstance(geoc_ml_path,list):
            for geoc in geoc_ml_path:
                dates = [name for name in os.listdir(geoc) if os.path.isdir(os.path.join(geoc, name))]
            # dates = '20230108_20230213'
                dont_process = 0
                for jj in range(len(dates)):
                    if os.path.exists(os.path.join(os.path.join(geoc),dates[jj]+"forward_model_comp_" + str(NP) + ".png")):
                        print(os.path.join(os.path.join(geoc),dates[jj]+"forward_model_comp_" + str(NP) + ".png") + 'already made moving on')
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
                                location,
                                NP) 
                
        else:
            dates = [name for name in os.listdir(geoc_ml_path) if os.path.isdir(os.path.join(geoc_ml_path, name))]
            # dates = '20230108_20230213'
            dont_process = 0 
            for ii in range(len(dates)):
                if os.path.exists(os.path.join(os.path.join(geoc_ml_path),dates[ii]+"forward_model_comp_"+ str(NP) + ".png")):
                        print(os.path.join(os.path.join(geoc_ml_path),dates[ii]+"forward_model_comp_" + str(NP) + ".png") + 'already made moving on')
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
                                    location,
                                    NP) 
    @timeout_decorator.timeout(4600)             
    def signal_mask(self,geoc_ml_path,scale_factor_mag=0.75,scale_factor_depth=0.055): 
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        # clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)*(113.11*10**3)
        clip_width =  self.event_object.diameter_mask_in_m 

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~SIGNAL MASK INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Depth used = "  + str(self.event_object.time_pos_depth['Depth']))
        print("Depth scaler = " + str(scale_factor_depth) )
        print('Mag used = ' + str(self.event_object.MTdict['magnitude']))
        print("mag scaler used = " + str(scale_factor_mag) )
        print("Mask Diameter in meters = " + str(clip_width))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~SIGNAL MASK INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
    @timeout_decorator.timeout(4600)
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
    @timeout_decorator.timeout(4600)
    def nested_uniform_down_sample(self,geoc_ml_path,nmpoints,scale_factor_mag=0.75,scale_factor_depth=0.055,stacked=False,cov=None):
        cent = [float(self.event_object.time_pos_depth['Position'][0]),float(self.event_object.time_pos_depth['Position'][1])]
        # clip_width = (float(self.event_object.MTdict['magnitude']) * scale_factor_mag)+(float(self.event_object.time_pos_depth['Depth'])*scale_factor_depth)*(113.11*10**3)
        clip_width =  self.event_object.diameter_mask_in_m 
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
                    print(geoc_ml_path[ii] + "_down_sampled" + " INITIATED")
                    geoc_downsampled_path.append(path)
                    print(path)

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

    def check_geoc_has_data(self,geoc_path,gacos=True):
        print("~~~~~~~~~~~CHECKING IF DATA AVAILBLE ~~~~~~~~~~~~~~~~~")
        if isinstance(geoc_path,list):
            list_removed_geoc = []
            list_removed_gacos = []

            for indx, geoc in enumerate(geoc_path):

                # if os.path.isdir(geoc) is False:
                #     geoc_path.remove(geoc)  
                subfolder = [ f.path for f in os.scandir(geoc) if f.is_dir()]
                removed_total = 0
                total = 0 
                print(geoc)
                for folder in subfolder:
                    if ('network' in folder or '11bad_ifg_ras' in folder or '11ifg_ras' in folder 
                    or 'info' in folder or 'results' in folder or 'semivariogram' in folder):
                        continue 
                    else:
                        print(folder)
                        total += 1
                        if len(os.listdir(folder)) == 0:
                            os.rmdir(folder)
                            removed_total +=1
                    print('remove total = ' + str(removed_total))
                    print('total = ' + str(total))
                if removed_total == total:
                    list_removed_geoc.append(geoc)
                    gacos = '/'.join(geoc.split('/')[0:-1])+'/GACOS_' + '_'.join(geoc.split('/')[-1].split('_')[1:len(geoc.split('_'))])
                    list_removed_gacos.append(gacos)
            for geoc in list_removed_geoc:
                print('I am removing this geoc === ' + geoc)
                shutil.rmtree(geoc)
                geoc_path.remove(geoc)
            if gacos is True :
                for gacos in list_removed_gacos:
                    shutil.rmtree(gacos)
                    print('I am removing this gacos === ' + gacos)
                    self.gacos_path.remove(gacos)
                return geoc_path
            else:
                return geoc_path
        else:
            subfolder = [ f.path for f in os.scandir(self.geoc_path) if f.is_dir()]
            removed_total = 0
            total = 0 
            for folder in subfolder:
                if ('network' in folder or '11bad_ifg_ras' in folder or '11ifg_ras' in folder 
                    or 'info' in folder or 'results' in folder):
                        continue 
                else:
                    total += 1
                    if len(os.listdir(folder)) == 0:
                            os.rmdir(folder)
                            removed_total +=1
                    if removed_total == total:
                        shutil.rmtree(geoc_path)
                        if gacos is True:
                            gacos = '/'.join(geoc_path.split('/')[0:-1])+'/GACOS_' + '_'.join(geoc_path.split('/')[-1].split('_')[1:len(geoc_path.split('_'))])
                            shutil.rmtree(gacos)
                            self.gacos_path.remove(gacos)
                            print('inversion failed No data pulled')    
                        else:
                            pass
            if gacos is True:
                return geoc_path
            else:
                return geoc_path
        print("~~~~~~~~~~~CHECKING IF DATA AVAILBLE ~~~~~~~~~~~~~~~~~")

    def check_data_pull(self):
        broken_data_pulls_geoc = [] 
        for indx, geoc in enumerate(self.geoc_path):
            frame_in_geoc = '_'.join(geoc.split('/')[-1].split('_')[1:len(geoc.split('_'))])
            file_checker = [frame_in_geoc +'.geo.E.tif',
                            frame_in_geoc+'.geo.hgt.tif',
                            frame_in_geoc + '.geo.mli.tif',
                            frame_in_geoc + '.geo.N.tif',
                            frame_in_geoc + '.geo.U.tif',
                            'baselines',
                            'metadata.txt',
                            'network.png',
                            frame_in_geoc + '-poly.txt']
                    
            # pulled_files = [f for f in os.listdir(geoc) if os.path.isfile(f)]
            # print(os.listdir(geoc))
            # print(sorted(file_checker))
            # print(sorted(pulled_files))
            print(os.listdir(geoc))
            print(file_checker)
            res = all(ele in os.listdir(geoc) for ele in file_checker)
            print(res)
            if  all(ele in os.listdir(geoc) for ele in file_checker) is False:
                print('Missing Files')
                print(set(os.listdir(geoc)) - set(file_checker))
                broken_data_pulls_geoc.append(geoc)
        for geoc in broken_data_pulls_geoc:
                print('I am removing this geoc as the data pull failed to get all files  ' + geoc)
                try:
                    shutil.rmtree(geoc)
                except:
                    shutil.move(geoc,geoc + '_busted_delete_when_can')
                self.geoc_path.remove(geoc)
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

    def compile_preproc_report(self):
        return




    def flush_geoc(self,geoc_path):
        for geoc in geoc_path:
            if os.path.isdir(geoc):
                shutil.rmtree(geoc)
            else:
                pass
        return 
    def flush_for_second_run(self):

        if self.geoc_clipped_path:
            self.flush_geoc(self.geoc_clipped_path)
            self.geoc_clipped_path = [] 

        if self.geoc_masked_path:
            self.flush_geoc(self.geoc_masked_path)
            self.geoc_masked_path = [] 

        if self.geoc_masked_signal:
            self.flush_geoc(self.geoc_masked_signal)
            self.geoc_masked_signal = [] 

        if self.geoc_ds_path:
            self.flush_geoc(self.geoc_ds_path)
            self.geoc_ds_path = [] 

        if self.geoc_QA_path:
            self.flush_geoc(self.geoc_QA_path)
            self.geoc_QA_path = [] 

        if self.geoc_final_path:
            self.flush_geoc(self.geoc_final_path)
            self.geoc_final_path = [] 

        files = glob.glob(self.event_object.GBIS_location + '/*')
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        
        if os.path.isfile(self.event_object.GBIS_insar_template_NP1):
            pass 
        else: 
            shutil.copy("./example_GBIS_input.inp",self.event_object.GBIS_insar_template_NP1)
        
        if os.path.isfile(self.event_object.GBIS_insar_template_NP2):
            pass 
        else: 
            shutil.copy("./example_GBIS_input.inp",self.event_object.GBIS_insar_template_NP2)

        return

    def flush_processing(self):
        # if self.geoc_path:
        #     self.flush_geoc(self.geoc_path,remove_data=False)
        # if self.gacos_path:
        #     self.flush_geoc(self.gacos_path,remove_data=False)

        if self.geoc_ml_path:
            self.flush_geoc(self.geoc_ml_path)
            self.geoc_ml_path = [] 

        if self.geoc_gacos_corr_path:
            self.flush_geoc(self.geoc_gacos_corr_path)
            self.geoc_gacos_corr_path = []

        if self.geoc_clipped_path:
            self.flush_geoc(self.geoc_clipped_path)
            self.geoc_clipped_path = [] 

        if self.geoc_masked_path:
            self.flush_geoc(self.geoc_masked_path)
            self.geoc_masked_path = [] 

        if self.geoc_masked_signal:
            self.flush_geoc(self.geoc_masked_signal)
            self.geoc_masked_signal = [] 

        if self.geoc_ds_path:
            self.flush_geoc(self.geoc_ds_path)
            self.geoc_ds_path = [] 

        if self.geoc_QA_path:
            self.flush_geoc(self.geoc_QA_path)
            self.geoc_QA_path = [] 

        if self.geoc_final_path:
            self.flush_geoc(self.geoc_final_path)
            self.geoc_final_path = [] 

        files = glob.glob(self.event_object.GBIS_location + '/*')
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        
        if os.path.isfile(self.event_object.GBIS_insar_template_NP1):
            pass 
        else: 
            shutil.copy("./example_GBIS_input.inp",self.event_object.GBIS_insar_template_NP1)
        
        if os.path.isfile(self.event_object.GBIS_insar_template_NP2):
            pass 
        else: 
            shutil.copy("./example_GBIS_input.inp",self.event_object.GBIS_insar_template_NP2)

        return

    def flush_all_processing(self):
        directories = os.listdir(self.event_object.LiCS_locations)
        
        for directory in directories:
            # print(directory)
            if 'GEOC' in directory or 'GACOS' in directory: 
                shutil.rmtree(os.path.join(self.event_object.LiCS_locations,directory),ignore_errors=True)
        self.geoc_QA_path = [] 
        self.geoc_ds_path = [] 
        self.geoc_masked_signal = [] 
        self.geoc_masked_path = [] 
        self.geoc_clipped_path = [] 
        self.geoc_ml_path = [] 
        self.geoc_gacos_corr_path = []
        self.geoc_path = []
        self.gacos_path = [] 
      
        print(os.chdir(self.top_level_dir))
        files = glob.glob(self.event_object.GBIS_location + '/*')
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        
        if os.path.isfile(self.event_object.GBIS_insar_template_NP1):
            pass 
        else: 
            # os.chdir(cwd)
            os.chdir(self.top_level_dir)
            shutil.copy("./example_GBIS_input.inp",self.event_object.GBIS_insar_template_NP1)
        
        if os.path.isfile(self.event_object.GBIS_insar_template_NP2):
            pass 
        else: 
            # os.chdir(cwd)
            os.chdir(self.top_level_dir)
            shutil.copy("./example_GBIS_input.inp",self.event_object.GBIS_insar_template_NP2)

        return

if __name__ == "__main__":
    DaN = deformation_and_noise("us6000jk0t")

        
