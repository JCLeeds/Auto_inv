#import matlab.engine
import sys 
from scipy.io import loadmat
import pandas as pd
import numpy as np
import shutil
# import simple_plot as sp
import subprocess as sp
import os
import matlab.engine 
import scrape_USGS as sUSGS
# import misc_scripts.data_ingestion_legacy as DI
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
import time 
import multiprocessing as multi 
import preproc as DaN
import subprocess as sp 
import os
import h5py
from scipy import io
from scipy.io import loadmat
import LiCSBASJC_output as op 
import pandas as pd
import time 
import output_model as om
import pygmt
import timeout_decorator
import pickle
import datetime 
import local2llh as l2llh
import glob
import output_location_comp as olc
import gmt_profile_plot_auto as profile
import llh2local as llh2l
import GBIS_output_clean as GOC



class auto_GBIS:
    def __init__(self,deformation_and_noise_object,GBIS_loc,NP=1,number_trials=1e6,pygmt=True,generateReport=True,location_search=False,limit_trials=False):
        # old_stdout = sys.stdout
        # log_file = open("auto_GBIS.log","w")
        # sys.stdout = log_file

        self.DaN_object = deformation_and_noise_object
        self.path_to_data = self.DaN_object.event_object.GBIS_location
        self.number_trials = number_trials
        self.pygmt = pygmt 
        self.NP = NP
        self.generateReport = generateReport
        self.limit_trials = limit_trials

        if self.NP == 1: 
            self.GBIS_input_loc = self.DaN_object.event_object.GBIS_insar_template_NP1
        elif self.NP == 2:  
            self.GBIS_input_loc = self.DaN_object.event_object.GBIS_insar_template_NP2
        else:
            raise Exception("And just how many Nodal Planes do you think there are? NP has to be either 1 or 2") 

        t = time.time()
        print("~~~~~~~~~ GBIS_RUN Starting ~~~~~~~~~~~~~")
        print("~~~~~~~~~ DATA LOADED FOR INVERSION ~~~~~~~~~~~~~")
        self.npzfiles = self.get_data_location()
        print(self.npzfiles)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.GBIS_mat_file_location, self.sill_nug_range = self.convert_GBIS_Mat_format()
        # if limit_trials == True and len(self.GBIS_mat_file_location) > 5:
        #     self.number_trials = number_trials/10

        self.show_noise_params()
        self.path_to_GBIS = self.read_input(GBIS_loc)
        self.eng = self.start_matlab_set_path()

        self.estimate_length = self.calc_square_start()
        self.max_dist = self.maximum_dist()
        self.boundingbox = self.calc_boundingbox()
        self.y_lower,self.y_upper,self.x_lower,self.x_upper = self.boundingBox_meters()
        self.create_insar_input()
        if location_search:
            self.edit_input_priors_wide_seach(NP=NP)
        else:
            self.edit_input_priors(NP=NP)
      
        self.opt_model, self.GBIS_lon, self.GBIS_lat = self.gbisrun()
 
        # self.plot_locations()
        self.create_catalog_entry(NP)
        self.create_beachball_InSAR(NP)
       

        if pygmt is False:
            pass 
        else:
            try:
                self.strip_outputs(NP)
            except Exception as e:
                    pass
            try:
                self.gmt_output(NP)
            except Exception as e:
                print(e)
            try:
                self.location_output(NP)
            except Exception as e:
                print(e)
        print("~~~~~~~~~ GBIS_RUN Finished ~~~~~~~~~~~~~")
        t2 = time.time()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~ GBIS RUN SINGLE EVENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("time taken to INVERT {:10.4f} seconds".format((t2-t)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~          ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # sys.stdout = old_stdout
        # log_file.close()
       
    
    
    def convert_GBIS_Mat_format(self):
        #Add a check in for number of points 
        GBIS_mat_file_location = []
        sill_nug_range = []
        self.data = [] 
        for ii in range(len(self.npzfiles)):
            one_field = np.load(self.npzfiles[ii])
            print(list(one_field.keys()))
            # print(self.data.files)
            # LiCS_lib.npz2mat(self.npzfiles[0])
            Phase = one_field['ph_disp'].T 
            Lat = one_field['lonlat'][:,1].T
            Lon = one_field['lonlat'][:,0].T
            Inc = one_field['la'].T
            Heading = one_field['heading'].T
            print(np.shape(Phase))
            sill_nug_range.append(list(one_field['sill_nugget_range'].T))
            tmpnpz_gbis_format =self.npzfiles[ii][:-3] + 'GBIS.npz'
            np.savez(tmpnpz_gbis_format, Phase=Phase,Lat=Lat,Lon=Lon,Inc=Inc,Heading=Heading)
            LiCS_lib.npz2mat(tmpnpz_gbis_format)
            GBIS_mat_file_location.append(tmpnpz_gbis_format[:-3] +'mat')
            self.data.append(one_field)
        return GBIS_mat_file_location , sill_nug_range
     

    def read_input(self,GBIS_loc):
        # alltext = [] 
        # f = open(GBIS_loc)
        # for line in f:
        #     if line.startswith('#'):
        #         continue 
        #     else:
        #         alltext.append(line)
        # path = alltext[0].strip('\n')
        return GBIS_loc
    
    def start_matlab_set_path(self):
        eng = matlab.engine.start_matlab()
        s = eng.genpath(self.path_to_GBIS)
        eng.addpath(s, nargout=0)
        return eng 
    

    def calc_square_start(self):
        mu = 3.2e10
        slip_rate=5.5e-5
        L = np.cbrt(float(self.DaN_object.event_object.MTdict['moment'])/(slip_rate*mu))
        return L
    
    def calc_boundingbox(self):
        max_lats = [] 
        min_lats = [] 
        max_lons = [] 
        min_lons = [] 
        for ii in range(len(self.data)):
            max_lats.append(np.max(self.data[ii]['lonlat'][:,1]))
            min_lats.append(np.min(self.data[ii]['lonlat'][:,1]))
            max_lons.append(np.max(self.data[ii]['lonlat'][:,0]))
            min_lons.append(np.min(self.data[ii]['lonlat'][:,0]))
        
        # boundingbox = [round(np.max(min_lons),4)+0.0001,round(np.min(max_lats),4)-0.0001,round(np.min(max_lons),4)-0.0001,round(np.max(min_lats),4)+0.0001]
        # boundingbox = [np.max(min_lons),np.min(max_lats),np.min(max_lons),np.max(min_lats)]
        boundingbox = [np.min(min_lons),np.max(max_lats),np.max(max_lons),np.min(min_lats)]
        # ll = [lons.flatten(),lats.flatten()]
        # ll = np.array(ll,dtype=float)
        # xy = llh.llh2local(ll,np.array([locations[1],locations[0]],dtype=float))
        #    referance = [np.median(lics_mat['Lon']), np.median(lics_mat['Lat'])]
        return boundingbox

 
    def get_data_location(self):
        onlyfiles = [f for f in os.listdir(self.path_to_data) if os.path.isfile(os.path.join(self.path_to_data, f))]
        npzfiles = []
        for file in onlyfiles:
            if ".npz" in file and '.GBIS' not in file:
                full_path = os.path.join(self.path_to_data,file)
                npzfiles.append(full_path)
            else:
                pass 
        # if len(npzfiles) > 1:

        #     print("Two mat files presenent please remove one")
        return npzfiles

    def maximum_dist(self):
        max_dists = [] 
        for ii in range(len(self.data)):
            max_dists.append((np.sqrt((np.max(self.data[ii]['lonlat_m'][:,0]) - np.min(self.data[ii]['lonlat_m'][:,0])) ** 2
                                + (np.max(self.data[ii]['lonlat_m'][:,1]) - np.max(self.data[ii]['lonlat_m'][:,1])) ** 2)))
        max_dist = np.min(max_dists) # edited from mean to min 
           
        return max_dist

    def boundingBox_meters(self):
        for ii in range(len(self.data)):
            ref_location = [float(self.DaN_object.event_object.time_pos_depth['Position'][1]),float(self.DaN_object.event_object.time_pos_depth['Position'][0])]
            ll = [self.data[ii]['lonlat'][:,0].flatten(),self.data[ii]['lonlat'][:,1].flatten()]
            ll = np.array(ll,dtype=float)
            xy = llh2l.llh2local(ll,np.array([ref_location[0],ref_location[1]],dtype=float)) # CHANGE BACK
            xx_vec = xy[0,:]
            yy_vec = xy[1,:]
            
            max_lats = [] 
            min_lats = [] 
            max_lons = [] 
            min_lons = [] 
        
            max_lats.append(np.max(yy_vec))
            min_lats.append(np.min(yy_vec))
            max_lons.append(np.max(xx_vec))
            min_lons.append(np.min(xx_vec))
            # name = 'testnpzformetergrid' + str(ii) +'.npz'
            # np.savez(tmpnpz_gbis_format, xy=xy,ll=ll)

        boundingbox = [np.max(min_lons),np.min(max_lats),np.min(max_lons),np.max(min_lats)]
        y_lower,y_upper,x_lower,x_upper = np.max(min_lats) + 5000, np.min(max_lats)-5000, np.max(min_lons)+5000, np.min(max_lons)-5000
        print('~~~~~~~~~~~~~~~~~~~~~~~~~LOOKY LOOOKY--------------------------')
        print(y_lower)
        print(y_upper)
        print(x_lower)
        print(x_upper)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~LOOKY LOOOKY--------------------------')
        # print(xy)
        # # break 
        # y_lower,y_upper,x_lower,x_upper = [1000,1000,1000,1000]
        # boundingbox = [np.max(min_lons),np.min(max_lats),np.min(max_lons),np.max(min_lats)]
        #    referance = [np.median(lics_mat['Lon']), np.median(lics_mat['Lat'])]
        return y_lower,y_upper,x_lower,x_upper

    
    def show_noise_params(self):
        label = []
        for ii in range(len(self.npzfiles)):
            label.append(self.npzfiles[ii].split('/')[-1])
        sills = [] 
        ranges = [] 
        nugs = []
        for array in self.sill_nug_range:
            sills.append(array[0])
            nugs.append(array[1])
            ranges.append(array[2])
        print(sills)
        print(ranges)
        print(nugs)
        print(self.sill_nug_range)

        print(np.shape(self.sill_nug_range))
        plt.figure()
        plt.scatter(label,sills)
        plt.title('Sill comparisons')
        plt.savefig(os.path.join(self.path_to_data,'File_sill_comp.png'))

        plt.figure()
        plt.title('Range comparison ')
        plt.scatter(label,ranges)
        plt.savefig(os.path.join(self.path_to_data,'File_range_comp.png'))

        plt.figure()
        plt.title('Nugget comparison ')
        plt.scatter(label,nugs)
        plt.savefig(os.path.join(self.path_to_data,'File_nug_comp.png'))
        return 
    # def nan_across_frame_checker(self):
    #     ''' If for a frame with over 3 images there is a nan in 2/3 change the final frame value to nan, fixes rouge pixels in water issue, may introduce errors at center of EQs'''
    #     total_frames = [] 
    #     for ii in range(len(self.GBIS_mat_file_location)):
    #         frame = self.GBIS_mat_file_location[ii].split('/')[-1].split('_floatml_')[0].replace('GEOC_', '')
    #         total_frames.append(frame)
    #     total_frames = list(set(total_frames))
    #     for frame in total_frames:

    #     return 
    def create_insar_input(self):
        input_loc = self.GBIS_input_loc
        
        # insarID = 1;                            % InSAR dataset unique identifier
        # insar{insarID}.dataPath ='input_unwrapped_interferogram.mat'
        # insar{insarID}.wavelength = 0.056;      % Wavelength in m (e.g., Envisat/ERS/Sentinel: 0.056; CSK/TSX/TDX: 0.031)
        # insar{insarID}.constOffset = 'y';       % Remove constant offset? 'y' or 'n'
        # insar{insarID}.rampFlag = 'y';          % Remove linear ramp? 'y' or 'n'
        # insar{insarID}.sillExp =2.2925882414169513e-05
        # insar{insarID}.range =10185.22042427217
        # insar{insarID}.nugget =2.991728653751577e-06
        # insar{insarID}.quadtreeThresh = 0; % Quadtree threshold variance (e.g., 0.01^2 m or 1e-04 m)
        # with open(input_loc,'r') as file:
        # if len(self.GBIS_mat_file_location) > 10:
        frame_info = [] 
        tbaselines = [] 
        index = []
        primary_dates = [] 
        secondary_dates = []
        matfile_locations = [] 
        data_df = {'frame':None,
                    'tbl':None,
                    'index':None}
        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        file.close()
        time = params[1].split('=')[-1]
        for ii in range(len(self.GBIS_mat_file_location)):
            frame = self.GBIS_mat_file_location[ii].split('/')[-1].split('_floatml_')[0].replace('GEOC_', '')
            date_primary, date_secondary = self.GBIS_mat_file_location[ii].split('/')[-1].split('_QAed_')[-1].split('.')[0].split('_')[0],self.GBIS_mat_file_location[ii].split('/')[-1].split('_QAed_')[-1].split('.')[0].split('_')[1]
            print(date_primary)
            print(date_secondary)
            print(frame)
            print(date_primary[0:4], date_primary[4:6], date_primary[6:8])
            date_primary = datetime.datetime(int(date_primary[0:4]), int(date_primary[4:6]), int(date_primary[6:8]))
            print(date_secondary[0:4], date_secondary[4:6], date_secondary[6:8])
            date_secondary = datetime.datetime(int(date_secondary[0:4]), int(date_secondary[4:6]), int(date_secondary[6:8]))
            
            print(date_secondary - date_primary)
            tbaselines.append(date_secondary - date_primary)
            frame_info.append(frame)
            primary_dates.append(date_primary)
            secondary_dates.append(secondary_dates)
            index.append(ii) 

        data_df['frame'] = frame_info
        data_df['primary_date'] = primary_dates
        data_df['secondary_date'] = secondary_dates
        data_df['tbl'] = tbaselines
        data_df['index'] = index
        print(data_df)
        data_df = pd.DataFrame.from_dict(data_df)

        total_frames =  list(set(frame_info))

        index_to_keep = [] 
        for frame in total_frames: 
            frame_df = data_df[data_df['frame'] == frame]
            if len(frame_df) > 2 and len(list(set(total_frames))) > 3:
                index_to_keep.append(frame_df[frame_df.tbl == frame_df.tbl.min()].index.values)
            elif len(frame_df) > 2 and len(list(set(total_frames))) > 1:
                index_to_keep.append(frame_df[frame_df.tbl == frame_df.tbl.min()].index.values)
                index_to_keep.append(frame_df[frame_df.tbl == frame_df.tbl.nsmallest(2).iloc[-1]].index.values)
                print(frame_df[frame_df.tbl == frame_df.tbl.nsmallest(2).iloc[-2]])
                print('IM IN THIS ELIF STATEMENT')
                # index_to_keep.append(frame_df[frame_df.tbl == frame_df.tbl.nsmallest(3).iloc[-1]].index.values)
                # index_to_keep.append(frame_df[frame_df.tbl == frame_df.tbl.nsmallest(4).iloc[-1]].index.values)
            else:
                index_to_keep.append(frame_df.index.values)

        print(index_to_keep)
        final_indexs = []
        for index_list in index_to_keep:
            for ii in range(len(index_list)):
                final_indexs.append(int(index_list[ii]))
        final_indexs = np.unique(np.array(final_indexs,dtype=int))
        print(final_indexs)
        # self.GBIS_mat_file_location = np.array(self.GBIS_mat_file_location)[final_indexs]
        # self.sill_nug_range = np.array(self.sill_nug_range)[final_indexs][:]

        if os.path.isfile(self.path_to_data + '/ifgms_used_in_inversion.txt'):
            os.remove(self.path_to_data + '/ifgms_used_in_inversion.txt')
        else:
            pass
        with open(self.path_to_data + '/ifgms_used_in_inversion.txt','w') as file:
            for index in final_indexs:
                file.write( str(index) + 'file: ' +  self.GBIS_mat_file_location[index] + '\n')
        file.close()
            
        with open(input_loc,'r') as file:
            lines = file.readlines() 
        file.close()
        #     file.readlines() 
        # write = [] 
        for ii in range(len(lines)):
            if any('insarID' in x for x in lines[ii]):
                lines[ii] = ' '
            else:
                pass 
        
          
        strings_to_write = [] 
        for ii in range(len(self.GBIS_mat_file_location)):
            strings_to_write.append('insarID = ' + str(ii + 1) + ';' + '\n' +
                                'insar{insarID}.dataPath = ' +' \'' + self.GBIS_mat_file_location[ii] + '\'' +';' + '\n' + 
                                'insar{insarID}.wavelength = 0.056;' + '\n' +
                                'insar{insarID}.constOffset = \'y\';' + '\n' +
                                'insar{insarID}.rampFlag = \'n\';' +   '\n' +
                                'insar{insarID}.sillExp =' + str(self.sill_nug_range[ii][0]) +';' + '\n' +
                                'insar{insarID}.range =' + str(self.sill_nug_range[ii][2]) +';' + '\n' +
                                'insar{insarID}.nugget=' + str(np.abs(self.sill_nug_range[ii][1])) +';' + '\n'+ 
                                'insar{insarID}.quadtreeThresh = 0;' +'\n')
            
        with open(input_loc, 'w') as file:
            file.writelines(lines)
            file.writelines(strings_to_write)
        file.close()
        return 

    def edit_input_priors(self,NP=1):
        # if NP == 1:
        #     input_loc = self.DaN_object.event_object.GBIS_insar_template_NP1
        # elif NP == 2:
        #     input_loc = self.DaN_object.event_object.GBIS_insar_template_NP2
        #  else:
        #     raise Exception("And just how many Nodal Planes do you think there are? NP has to be either 1 or 2") 
        input_loc = self.GBIS_input_loc

        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        file.close()
  
        name = params[0].split('=')[-1]
        time = params[1].split('=')[-1]
        latitude = float(params[2].split('=')[-1])
        longitude = float(params[3].split('=')[-1])
        magnitude = float(params[4].split('=')[-1])
        magnitude_type = params[5].split('=')[-1]
        moment = float(params[6].split('=')[-1])
        depth = float(params[7].split('=')[-1])
        catalog = params[8].split('=')[-1]
        strike1 = float(params[9].split('=')[-1])
        dip1 = float(params[10].split('=')[-1])
        rake1 = float(params[11].split('=')[-1])
        strike2 = float(params[12].split('=')[-1])
        dip2 = float(params[13].split('=')[-1])
        rake2 = float(params[14].split('=')[-1]) 

        slip_rate=5.5e-5
        mu = 3.2e10
        L = self.estimate_length
    
        # L = 8000
        slip = L*slip_rate

        DS1 = np.sin(rake1*np.pi/180) * slip * -1
        SS1 = np.cos(rake1*np.pi/180) * slip * -1

        DS2 = np.sin(rake2*np.pi/180) * slip * -1
        SS2 = np.cos(rake2*np.pi/180) * slip * -1


        
        # if -dip1 - 30 < -90:
        #     dip1 = 0 
        dip1_lower = -dip1 - 30
        dip2_lower = -dip2 - 30
        if dip1_lower < -89.9:
            dip1_lower = -89.9 
        if dip2_lower < -89.9:
            dip2_lower = -89.9

        
                # Capping dip at 15 deg this removes convergance to a non physically shallow dipping source, This will need revisiting.
        dip1_upper = -dip1 + 30
        dip2_upper = -dip2 + 30
        if dip1_upper > -1:
            dip1_upper = -5 
        if dip2_upper > -1:
            dip2_upper = -5  

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  " + str(strike2))
        #Convert from USGS to GBIS convention 
        strike1 = (strike1 + 180) % 360
        strike2 = (strike2 + 180) % 360

        if strike1 == 0: 
            strike1 = strike1 + 360 
        if strike2 == 0:
            strike2 = strike2 + 360 
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  " + str(strike2))

        strike_upper1 = (strike1 + 40)  % 360 
        strike_upper2 = (strike2 + 40) % 360 
        strike_lower1 = (strike1 - 40) # needs to allow for a negative lower bound so that 3rd to 4th quadrant can be scanned over 
        strike_lower2 = (strike2 - 40) # needs to allow for a negative lower bound so that 3rd to 4th quadrant can be scanned over 

        if strike_upper1 == 0:
            strike_upper1 = strike_upper1 + 360 
        if strike_upper2 == 0:
            strike_upper2 = strike_upper2 + 360

        if strike_lower1 == 0:
            strike_lower1 = strike_lower1 + 360 
        if strike_lower2 == 0:
            strike_lower2 = strike_lower2 + 360
        
        if strike_upper1 < strike_lower1:
            strike_upper1, strike_lower1 = strike_lower1, strike_upper1
        
        if strike_upper2 < strike_lower2:
            strike_upper2, strike_lower2 = strike_lower2, strike_upper2

        if strike1 > strike_lower1 and strike1 < strike_upper1:
            pass 
        else:
            strike1 = (strike_lower1 + strike_upper1) /2
        
        if strike2 > strike_lower2 and strike2 < strike_upper2:
            pass 
        else:
            strike2 = (strike_lower2 + strike_upper2) /2

        if depth*0.25 <= 2500:
            depth_lower = depth*0.1
        else:
            depth_lower = depth*0.25

        print("################################################")
        print(self.GBIS_mat_file_location)
        print(input_loc)
        print('     L       W      Z     Dip     Str      X       Y      SS       DS ')
        with open(input_loc,'r') as file:
            lines = file.readlines() 
        file.close()
        for ii in range(len(lines)):
            # print(lines[ii])
            if 'geo.referencePoint' in lines[ii]:
                lines[ii] = ('geo.referencePoint =['
                    + str(self.DaN_object.event_object.time_pos_depth['Position'][1]) + ";"
                    + str(self.DaN_object.event_object.time_pos_depth['Position'][0]) + "];" + '\n')
            elif 'geo.boundingBox' in lines[ii]:
                lines[ii] = ('geo.boundingBox =[' 
                            + str(self.boundingbox[0]) + ";" 
                            + str(self.boundingbox[1]) + ";"
                            + str(self.boundingbox[2]) + ";" 
                            + str(self.boundingbox[3]) + "];" '\n')
                            
            elif 'modelInput.fault.start' in lines[ii] and NP==1:
                lines[ii] = ('modelInput.fault.start=['
                            + str(int(self.estimate_length)) + ';  ' 
                            + str(int(self.estimate_length)) + ';  '
                            + str(int(depth)) + ';  '
                            + str(int(-dip1)) + ';  '
                            + str(int(strike1)) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(SS1) + ';  '
                            + str(DS1) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.start' in lines[ii] and NP==2:
                lines[ii] = ('modelInput.fault.start=['
                            + str(int(self.estimate_length)) + ';  ' 
                            + str(int(self.estimate_length)) + ';  '
                            + str(int(depth)) + ';  '
                            + str(int(-dip2)) + ';  '
                            + str(int(strike2)) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(SS2) + ';  '
                            + str(DS2) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.step' in lines[ii] and NP==1:
                pass
            elif 'modelInput.fault.step' in lines[ii] and NP==2:
                pass 
            elif 'modelInput.fault.lower' in lines[ii] and NP==1:
                # if strike_lower1 < strike_upper1:
                #     strike_bound = strike_lower1
                # else:
                #     strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.5)) + ';  ' 
                            + str(int(self.estimate_length*0.15)) + ';  '
                            + str(int(depth_lower)) + ';  '
                            + str(dip1_lower) + ';  '
                            + str(int(strike_lower1)) + ';  '
                            + str(int(-self.max_dist/5)) + ';  '
                            + str(int(-self.max_dist/5)) + ';  '
                            # + str(self.x_lower) + ';  '
                            # + str(self.y_lower) + ';  '
                            + str(SS1 - 4) + ';  '
                            + str(DS1 - 4) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.lower' in lines[ii] and NP==2:
                # if strike_lower2 < strike_upper2:
                #     strike_bound = strike_lower2
                # else:
                #     strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.5)) + ';  ' 
                            + str(int(self.estimate_length*0.15)) + ';  '
                            + str(int(depth_lower)) + ';  '
                            + str(dip2_lower) + ';  '
                            + str(int(strike_lower2)) + ';  '
                            + str(int(-self.max_dist/5)) + ';  '
                            + str(int(-self.max_dist/5)) + ';  '
                            # + str(self.x_lower) + ';  '
                            # + str(self.y_lower) + ';  '
                            + str(SS2 - 5) + ';  '
                            + str(DS2 - 5) + '];'
                            '\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==1:
                # if strike_lower1 > strike_upper1:
                #     strike_bound = strike_lower1
                # else:
                #     strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*5)) + ';  ' 
                            + str(int(self.estimate_length*2.5)) + ';  '
                            + str(int(depth*2)) + ';  '
                            + str(int(dip1_upper)) + ';  '
                            + str(int(strike_upper1)) + ';  '
                            + str(int(self.max_dist/5)) + ';  '
                            + str(int(self.max_dist/5)) + ';  '
                            # + str(self.x_upper) + ';  '
                            # + str(self.y_upper) + ';  '
                            + str(SS1 + 5) + ';  '
                            + str(DS1 + 5) + '];'
                            '\n'
                            ) 
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==2:
                # if strike_lower2 > strike_upper2:
                #     strike_bound = strike_lower2
                # else:
                #     strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*5)) + ';  ' 
                            + str(int(self.estimate_length*2.5)) + ';  '
                            + str(int(depth*2)) + ';  '
                            + str(int(dip2_upper)) + ';  '
                            + str(int(strike_upper2)) + ';  '
                            + str(int(self.max_dist/5)) + ';  '
                            + str(int(self.max_dist/5)) + ';  '
                            # + str(self.x_upper) + ';  '
                            # + str(self.y_upper) + ';  '
                            + str(SS2 + 5) + ';  '
                            + str(DS2 + 5) + '];'
                            '\n'
                            ) 
                print(lines[ii])
        with open(input_loc, 'w') as file:
            file.writelines(lines)
        file.close()
        return
    def edit_input_priors_location_finding_run(self,NP=1):
        """

        For initial location run increased distance search and reduced mechanism variance

        """
        # if NP == 1:
        #     input_loc = self.DaN_object.event_object.GBIS_insar_template_NP1
        # elif NP == 2:
        #     input_loc = self.DaN_object.event_object.GBIS_insar_template_NP2
        #  else:
        #     raise Exception("And just how many Nodal Planes do you think there are? NP has to be either 1 or 2") 
        input_loc = self.GBIS_input_loc

        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        file.close()
        name = params[0].split('=')[-1]
        time = params[1].split('=')[-1]
        latitude = float(params[2].split('=')[-1])
        longitude = float(params[3].split('=')[-1])
        magnitude = float(params[4].split('=')[-1])
        magnitude_type = params[5].split('=')[-1]
        moment = float(params[6].split('=')[-1])
        depth = float(params[7].split('=')[-1])
        catalog = params[8].split('=')[-1]
        strike1 = float(params[9].split('=')[-1])
        dip1 = float(params[10].split('=')[-1])
        rake1 = float(params[11].split('=')[-1])
        strike2 = float(params[12].split('=')[-1])
        dip2 = float(params[13].split('=')[-1])
        rake2 = float(params[14].split('=')[-1])   
        # if -dip1 - 30 < -90:
        #     dip1 = 0 
        dip1_lower = -dip1 - 5
        dip2_lower = -dip2 - 5
        if dip1_lower < -89.9:
            dip1_lower = -89.9 
        if dip2_lower < -89.9:
            dip2_lower = -89.9


        # Capping dip at 15 deg this removes convergance to a non physically shallow dipping source, This will need revisiting.
        dip1_upper = -dip1 + 5
        dip2_upper = -dip2 + 5
        if dip1_upper > -1:
            dip1_upper = -1 
        if dip2_upper > -1:
            dip2_upper = -1  

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  " + str(strike2))
        #Convert from USGS to GBIS convention 
        strike1 = (strike1 + 180) % 360 
        strike2 = (strike2 + 180) % 360

        if strike1 == 0: 
            strike1 = strike1 + 360 
        if strike2 == 0:
            strike2 = strike2 + 360 
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  " + str(strike2))

        strike_upper1 = (strike1 + 10)  % 360 
        strike_upper2 = (strike2 + 10) % 360 
        strike_lower1 = (strike1 - 10) % 360 
        strike_lower2 = (strike2 - 10) % 360 

        if strike_upper1 == 0:
            strike_upper1 = strike_upper1 + 360 
        if strike_upper2 == 0:
            strike_upper2 = strike_upper2 + 360

        if strike_lower1 == 0:
            strike_lower1 = strike_lower1 + 360 
        if strike_lower2 == 0:
            strike_lower2 = strike_lower2 + 360
        
        if strike_upper1 < strike_lower1:
            strike_upper1, strike_lower1 = strike_lower1, strike_upper1
        
        if strike_upper2 < strike_lower2:
            strike_upper2, strike_lower2 = strike_lower2, strike_upper2

        

   
                
        print("################################################")
        print(self.GBIS_mat_file_location)
        print(input_loc)
        print('     L       W      Z     Dip     Str      X       Y      SS       DS ')
        with open(input_loc,'r') as file:
            lines = file.readlines() 
        file.close()
        for ii in range(len(lines)):
            # print(lines[ii])
            if 'geo.referencePoint' in lines[ii]:
                lines[ii] = ('geo.referencePoint =['
                    + str(self.DaN_object.event_object.time_pos_depth['Position'][1]) + ";"
                    + str(self.DaN_object.event_object.time_pos_depth['Position'][0]) + "];" + '\n')
            elif 'geo.boundingBox' in lines[ii]:
                lines[ii] = ('geo.boundingBox =[' 
                            + str(self.boundingbox[0]) + ";" 
                            + str(self.boundingbox[1]) + ";"
                            + str(self.boundingbox[2]) + ";" 
                            + str(self.boundingbox[3]) + "];" '\n')
                            
            elif 'modelInput.fault.start' in lines[ii] and NP==1:
                lines[ii] = ('modelInput.fault.start=['
                            + str(int(self.estimate_length)) + ';  ' 
                            + str(int(self.estimate_length)) + ';  '
                            + str(int(depth)) + ';  '
                            + str(int(-dip1)) + ';  '
                            + str(int(strike1)) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.start' in lines[ii] and NP==2:
                lines[ii] = ('modelInput.fault.start=['
                            + str(int(self.estimate_length)) + ';  ' 
                            + str(int(self.estimate_length)) + ';  '
                            + str(int(depth)) + ';  '
                            + str(int(-dip2)) + ';  '
                            + str(int(strike2)) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.step' in lines[ii] and NP==1:
                pass 
            elif 'modelInput.fault.step' in lines[ii] and NP==2:
                pass 
            elif 'modelInput.fault.lower' in lines[ii] and NP==1:
                # if strike_lower1 < strike_upper1:
                #     strike_bound = strike_lower1
                # else:
                #     strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.95)) + ';  ' 
                            + str(int(self.estimate_length*0.95)) + ';  '
                            + str(int(depth*0.25)) + ';  '
                            + str(dip1_lower) + ';  '
                            + str(int(strike_lower1)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(-10.0) + ';  '
                            + str(-10.0) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.lower' in lines[ii] and NP==2:
                # if strike_lower2 < strike_upper2:
                #     strike_bound = strike_lower2
                # else:
                #     strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.95)) + ';  ' 
                            + str(int(self.estimate_length*0.95)) + ';  '
                            + str(int(depth*0.25)) + ';  '
                            + str(dip2_lower) + ';  '
                            + str(int(strike_lower2)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(-10.0) + ';  '
                            + str(-10.0) + '];'
                            '\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==1:
                # if strike_lower1 > strike_upper1:
                #     strike_bound = strike_lower1
                # else:
                #     strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*1.1)) + ';  ' 
                            + str(int(self.estimate_length*1.1)) + ';  '
                            + str(int(depth*3)) + ';  '
                            + str(int(dip1_upper)) + ';  '
                            + str(int(strike_upper1)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(10.0) + ';  '
                            + str(10.0) + '];'
                            '\n'
                            ) 
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==2:
                # if strike_lower2 > strike_upper2:
                #     strike_bound = strike_lower2
                # else:
                #     strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*1.1)) + ';  ' 
                            + str(int(self.estimate_length*1.1)) + ';  '
                            + str(int(depth*3)) + ';  '
                            + str(int(dip2_upper)) + ';  '
                            + str(int(strike_upper2)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(10.0) + ';  '
                            + str(10.0) + '];'
                            '\n'
                            ) 
                print(lines[ii])
        with open(input_loc, 'w') as file:
            file.writelines(lines)
        file.close()
        return
    def edit_input_priors_wide_seach(self,NP=1):
        input_loc = self.GBIS_input_loc

        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        file.close()
        name = params[0].split('=')[-1]
        time = params[1].split('=')[-1]
        latitude = float(params[2].split('=')[-1])
        longitude = float(params[3].split('=')[-1])
        magnitude = float(params[4].split('=')[-1])
        magnitude_type = params[5].split('=')[-1]
        moment = float(params[6].split('=')[-1])
        depth = float(params[7].split('=')[-1])
        catalog = params[8].split('=')[-1]
        strike1 = float(params[9].split('=')[-1])
        dip1 = float(params[10].split('=')[-1])
        rake1 = float(params[11].split('=')[-1])
        strike2 = float(params[12].split('=')[-1])
        dip2 = float(params[13].split('=')[-1])
        rake2 = float(params[14].split('=')[-1]) 

        slip_rate=5.5e-5
        mu = 3.2e10
        L = self.estimate_length
    
        # L = 8000
        slip = L*slip_rate

        DS1 = np.sin(rake1*np.pi/180) * slip 
        SS1 = np.cos(rake1*np.pi/180) * slip

        DS2 = np.sin(rake2*np.pi/180) * slip 
        SS2 = np.cos(rake2*np.pi/180) * slip

        if depth*0.25 <= 1000:
            depth_lower = 1 
        else:
            depth_lower = depth*0.1

        with open(input_loc,'r') as file:
            lines = file.readlines() 
        file.close()
        for ii in range(len(lines)):
            # print(lines[ii])
            if 'geo.referencePoint' in lines[ii]:
                lines[ii] = ('geo.referencePoint =['
                    + str(self.DaN_object.event_object.time_pos_depth['Position'][1]) + ";"
                    + str(self.DaN_object.event_object.time_pos_depth['Position'][0]) + "];" + '\n')
            elif 'geo.boundingBox' in lines[ii]:
                lines[ii] = ('geo.boundingBox =[' 
                            + str(self.boundingbox[0]) + ";" 
                            + str(self.boundingbox[1]) + ";"
                            + str(self.boundingbox[2]) + ";" 
                            + str(self.boundingbox[3]) + "];" '\n')
                            
            elif 'modelInput.fault.start' in lines[ii] and NP==1:
                lines[ii] = ('modelInput.fault.start=['
                            + str(int(self.estimate_length)) + ';  ' 
                            + str(int(self.estimate_length)) + ';  '
                            + str(int(depth)) + ';  '
                            + str(int(-dip1)) + ';  '
                            + str(int(strike1)) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(SS1) + ';  '
                            + str(DS1) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.start' in lines[ii] and NP==2:
                lines[ii] = ('modelInput.fault.start=['
                            + str(int(self.estimate_length)) + ';  ' 
                            + str(int(self.estimate_length)) + ';  '
                            + str(int(depth)) + ';  '
                            + str(int(-dip2)) + ';  '
                            + str(int(strike2)) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(SS2) + ';  '
                            + str(DS2) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.step' in lines[ii] and NP==1:
                pass
            elif 'modelInput.fault.step' in lines[ii] and NP==2:
                pass 
            elif 'modelInput.fault.lower' in lines[ii] and NP==1:
                # if strike_lower1 < strike_upper1:
                #     strike_bound = strike_lower1
                # else:
                #     strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.1)) + ';  ' 
                            + str(int(self.estimate_length*0.1)) + ';  '
                            + str(int(depth_lower)) + ';  '
                            + str(-89) + ';  '
                            + str(int(0)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(SS1 - 10) + ';  '
                            + str(DS1 - 10) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.lower' in lines[ii] and NP==2:
                # if strike_lower2 < strike_upper2:
                #     strike_bound = strike_lower2
                # else:
                #     strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.1)) + ';  ' 
                            + str(int(self.estimate_length*0.1)) + ';  '
                            + str(int(depth_lower)) + ';  '
                            + str(-89) + ';  '
                            + str(int(0)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(int(-self.max_dist/2)) + ';  '
                            + str(SS2 - 10) + ';  '
                            + str(DS2 - 10) + '];'
                            '\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==1:
                # if strike_lower1 > strike_upper1:
                #     strike_bound = strike_lower1
                # else:
                #     strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*10)) + ';  ' 
                            + str(int(self.estimate_length*10)) + ';  '
                            + str(int(depth*10)) + ';  '
                            + str(int(-1)) + ';  '
                            + str(int(360)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(SS1 + 10) + ';  '
                            + str(DS1 + 10) + '];'
                            '\n'
                            ) 
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==2:
                # if strike_lower2 > strike_upper2:
                #     strike_bound = strike_lower2
                # else:
                #     strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*10)) + ';  ' 
                            + str(int(self.estimate_length*10)) + ';  '
                            + str(int(depth*10)) + ';  '
                            + str(int(-1)) + ';  '
                            + str(int(360)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(int(self.max_dist/2)) + ';  '
                            + str(SS2 + 10) + ';  '
                            + str(DS2 + 10) + '];'
                            '\n'
                            ) 
                print(lines[ii])
        with open(input_loc, 'w') as file:
            file.writelines(lines)
        file.close()
        return


    def gbisrun(self, generateReport=True):
        cwd = os.getcwd()
        # os.chdir(self.DaN_object.event_object.GBIS_location)
        # if len(self.data) == 1:
        #     self.InSAR_codes = 1
        #     InSAR_codes_string = "invert_1_F"
        # else:
        # self.InSAR_codes = np.arange(len(self.GBIS_mat_file_location) + 1)[1:len(self.GBIS_mat_file_location)+1]
        # if len(self.GBIS_mat_file_location) > 10: 
            
        with open(self.path_to_data + '/ifgms_used_in_inversion.txt','r') as file:
            lines = file.readlines()
        file.close()
        self.InSAR_codes = []
        self.date_order = []
        for line in lines:
            self.InSAR_codes.append(int(line.split('file')[0])+1)
            self.date_order.append(line.split('_QAed_')[-1].split('.ds')[0])
        self.InSAR_codes = np.array(self.InSAR_codes)

        # self.InSAR_codes = [1]
        print("~~~~~~~~ Insar Codes ~~~~~~~~~~~~~~~")
        print(self.InSAR_codes)
        print("~~~~~~~ Insar Codes ~~~~~~~~~~~~~~~")
        print("~~~~~~~~ Data Paths ~~~~~~~~~~~~~~~")
        print(self.GBIS_mat_file_location)
        print("~~~~~~~ Data Paths ~~~~~~~~~~~~~~~")
        InSAR_codes_string = "invert_"
        print(InSAR_codes_string)
        print(self.InSAR_codes)
        for ii in range(len(self.InSAR_codes)):
            InSAR_codes_string = InSAR_codes_string + str(self.InSAR_codes[ii]) + "_"
      
        InSAR_codes_string = InSAR_codes_string + "F"
        print(InSAR_codes_string)
        self.outputdir= "./" + self.GBIS_input_loc.split('/')[-1][:-4] + "/" + InSAR_codes_string
        self.outputfile = self.outputdir + "/" + InSAR_codes_string +".mat"
        self.opt_model_vertex = self.outputdir +'/' + 'optmodel_vertex.mat'
        # if os.path.isdir(self.outputdir):
        #     print("Inversion Results already present skipping")
        # else:
        print(self.InSAR_codes)
        self.eng.GBISrun(self.GBIS_input_loc,self.InSAR_codes,'n','F',self.number_trials,'n','n','n',nargout=0)
        # self.eng.GBISrun(self.GBIS_input_loc,[1],'n','F',self.number_trials,'n','n','n',nargout=0)
        
    
        output_matrix = loadmat(self.outputfile,squeeze_me=True)
        opt_model = np.array(output_matrix['invResults'][()].item()[-1])
        self.const_offs =  np.array(output_matrix['invResults'][()].item()[2])[()].item()[10]
        self.const_offs = self.const_offs[9:len(self.const_offs)]
        
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~optimal model results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(opt_model)
        print(np.size(opt_model))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~optimal model results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # if os.path.isdir(self.outputdir): # temporary skip
        #     pass 
        # else:
        if self.generateReport:
            try:
                self.eng.generateFinalReport(self.outputfile,self.number_trials*0.3,nargout=0)
            except Exception as e:
                print(e)
                try:
                    self.eng.generateFinalReport(self.outputfile,self.number_trials*0.2,nargout=0)
                except Exception as e:
                    print(e)
                    print('Well we failed to generate a report my Friend, moving on swiftly nothing to see here')
                    try:
                        self.eng.generateFinalReport(self.outputfile,self.number_trials*0.1,nargout=0)
                    except Exception as e:
                        print(e)
                        print('Well we failed to generate a report my Friend, moving on swiftly nothing to see here')
                        pass

        if self.generateReport:
            try:
                output_list = [f for f in glob.glob(self.outputdir +'/Figures/'+"res_los_modlos*.mat")]
                GOC.main(self.outputfile,output_list,self.outputdir+'/Figures',int(self.number_trials*0.3),int(self.number_trials),num_faults=1)
            except:
                pass 
            





        # print(opt_model)
        # print([opt_model,'color','k','updipline','yes','projection','3D','origin',[self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],'final','yes','savedir',self.opt_model_vertex])
        self.eng.python_comp_drawmodel(self.outputfile,[float(self.DaN_object.event_object.time_pos_depth['Position'][1]),float(self.DaN_object.event_object.time_pos_depth['Position'][0])],self.opt_model_vertex,nargout=0)
        print('Here created vertex')
        Lon_GBIS, Lat_GBIS = self.eng.convert_location_output(str(self.GBIS_mat_file_location[0]),self.outputfile,nargout=2)
        # llh =  l2llh.local2llh([opt_model[5],opt_model[6]],[float(self.DaN_object.event_object.time_pos_depth['Position'][1]),float(self.DaN_object.event_object.time_pos_depth['Position'][0])])
        # Lon_GBIS, Lat_GBIS = llh[0], llh[1]
        # local2llh(xy, origin):
        print(Lon_GBIS,Lat_GBIS)
        # os.chdir(cwd)
        return opt_model, Lon_GBIS, Lat_GBIS
    
    def strip_outputs(self,NP):
        proc_images = glob.glob(self.DaN_object.event_object.LiCS_locations + "/*.png")
        proc_info =  glob.glob(self.DaN_object.event_object.LiCS_locations + "/*.txt")

        for image in proc_images:
            print(image)
            shutil.copy(image,self.DaN_object.event_object.GBIS_location)
        for info in proc_info:
            print(info)
            shutil.copy(info,self.DaN_object.event_object.GBIS_location)


        if isinstance(self.DaN_object.geoc_final_path,list):
            for ii in range(len(self.DaN_object.geoc_final_path)):
                final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_final_path[ii].split('/')[-1].split('floatml')[0] + "processing_report_NP" + str(NP))
                images = glob.glob(self.DaN_object.geoc_final_path[ii] + "/*.png") 
                if os.path.isdir(final_path):
                    pass 
                else:
                    os.mkdir(final_path)
                for image in images:
                    print(image)
                    shutil.copy(image,final_path)
        else:
            final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_final_path.split('/')[-1].split('floatml')[0] + "processing_report_NP" + str(NP))
            images = glob.glob(self.DaN_object.geoc_final_path + "/*.png") 
            if os.path.isdir(final_path):
                pass 
            else:
                os.mkdir(final_path)
            for image in images:
                shutil.copy(image,final_path)

        if isinstance(self.DaN_object.geoc_QA_path,list):
            for ii in range(len(self.DaN_object.geoc_QA_path)):
                semivarigram_dir = self.DaN_object.geoc_QA_path[ii] +'/semivariograms'
                network_png = self.DaN_object.geoc_QA_path[ii] +'/network/network11.png'
                info_txt =   self.DaN_object.geoc_QA_path[ii] +'/info/11ifg_stats.txt'
                final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_final_path[ii].split('/')[-1].split('floatml')[0] + "processing_report_NP" + str(NP))
                semivariograms = glob.glob(semivarigram_dir + "/*.png")
                 
                if os.path.isdir(final_path):
                    pass 
                else:
                    os.mkdir(final_path)

                for grams in semivariograms:
                    print(grams)
                    shutil.copy(grams,final_path)
                shutil.copy(network_png,final_path)
                shutil.copy(info_txt,final_path)
        else:
            semivarigram_dir = self.DaN_object.geoc_QA_path +'/semivariograms/'
            network_png = self.DaN_object.geoc_QA_path +'/network/network11.png'
            info_txt =   self.DaN_object.geoc_QA_path +'/info/11ifg_stats.txt'
            final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_final_path[ii].split('/')[-1].split('floatml')[0] + "processing_report_NP" + str(NP))

            semivariograms = glob.glob(semivarigram_dir + "*.png")
                
            if os.path.isdir(final_path):
                pass 
            else:
                os.mkdir(final_path)

            for grams in semivariograms:
                print(grams)
                shutil.copy(grams,final_path)

            shutil.copy(network_png,final_path)
            shutil.copy(info_txt,final_path)



    @timeout_decorator.timeout(3600)
    def gmt_output(self,NP):
        
        if isinstance(self.DaN_object.geoc_final_path,list):
            for ii in range(len(self.DaN_object.geoc_clipped_path)):
                try:
                    final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_clipped_path[ii].split('/')[-1].split('floatml')[0] + "INVERSION_Results_NP" + str(NP))
                    # final_path = self.DaN_object.geoc_final_path[ii] + "_INVERSION_Results"
                    om.output_model(self.DaN_object.geoc_clipped_path[ii],
                                    final_path,
                                    self.opt_model,
                                    [self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],
                                    self.opt_model_vertex,
                                    self.const_offs,
                                    self.date_order)

                    # profile.output_profile_plot(self.DaN_object.geoc_clipped_path[ii],
                    #                 final_path,
                    #                 self.opt_model,
                    #                 [self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],
                    #                 self.opt_model_vertex)
                    
                except:
                    pass
            else:
                try:
                    final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_clipped_path[0].split('/')[-1].split('floatml')[0]+ "INVERSION_Results_NP" + str(NP))
                    om.output_model(self.DaN_object.geoc_clipped_path[0],
                                    final_path,
                                    self.opt_model,
                                    [self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],
                                    self.opt_model_vertex,
                                    self.const_offs,
                                    self.date_order)
                    # profile.output_profile_plot(self.DaN_object.geoc_clipped_path[ii],
                    #                 final_path,
                    #                 self.opt_model,
                    #                 [self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],
                    #                 self.opt_model_vertex)
                except:
                    pass

            
        return 
    
    def location_output(self,NP):

        event_file = os.path.join(self.DaN_object.event_object.LiCS_locations,self.DaN_object.event_object.ID+'.txt')
        with open(event_file,'r') as f:
            for line in f.readlines():
                if 'latitude =' in line:
                    lat = float(line.split('=')[-1])
                if 'longitude ='  in line:  
                    lon = float(line.split('=')[-1])
        f.close()

        if isinstance(self.DaN_object.geoc_final_path,list):
            for ii in range(len(self.DaN_object.geoc_clipped_path)):
                try:
                    final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_clipped_path[ii].split('/')[-1].split('floatml')[0] + "INVERSION_Results_NP" + str(NP))
                    # final_path = self.DaN_object.geoc_final_path[ii] + "_INVERSION_Results"
                    olc.output_location_comp(self.DaN_object.geoc_clipped_path[ii],
                                    final_path,
                                    self.opt_model,
                                    [lon,lat],
                                    self.opt_model_vertex)
                except:
                    pass 

                           
        else:
            try:
                final_path = os.path.join(self.DaN_object.event_object.GBIS_location,self.DaN_object.geoc_clipped_path[0].split('/')[-1].split('floatml')[0]+ "INVERSION_Results_NP" + str(NP))
                olc.output_location_comp(self.DaN_object.geoc_clipped_path[0],
                                    final_path,
                                    self.opt_model,
                                    [lon,lat],
                                    self.opt_model_vertex)
            except:
                pass
            



    def create_catalog_entry(self,NP):
        if os.path.isdir(self.path_to_data+'/'+ self.DaN_object.event_object.ID+'_catalog_entry'):
            print('Made it here its a directory')
            pass 
        else:
            print('made it here')
            os.mkdir(self.path_to_data+'/'+ self.DaN_object.event_object.ID+'_catalog_entry/')
            print('created directory')

        catalog= self.path_to_data+'/'+ self.DaN_object.event_object.ID+'_catalog_entry' + '/' +self.DaN_object.event_object.ID+ '_entry_NP' + str(NP)+ '.csv'
        
        input_loc = self.GBIS_input_loc

        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        file.close()
        
        USGS_latitude = float(params[2].split('=')[-1])
        USGS_longitude = float(params[3].split('=')[-1])

        
        if NP == 1:
            strike = self.opt_model[4] - 180 
            if strike < 360:
                strike  = strike + 360
            SS = self.opt_model[7]
            DS = self.opt_model[8]
            rake = np.degrees(np.arctan2(-DS,-SS))
            total_slip = np.sqrt(SS**2 + DS**2)
            mu = 3.2e10
            length = self.opt_model[0]
            width = self.opt_model[1]
            M0_assuming_mu = mu*length*width*total_slip
            Mw = (2/3)*np.log10(M0_assuming_mu*10**7) - 10.7 
            dict_for_catalog ={'USGS_ID': self.DaN_object.event_object.ID,
                               'NP_selected': NP,
                                'USGS_Mag': self.DaN_object.event_object.MTdict['magnitude'],
                                'USGS_lat': USGS_latitude,
                                'USGS_lon': USGS_longitude,
                                'USGS_Depth': self.DaN_object.event_object.time_pos_depth['Depth'],
                                'USGS_Moment_Depth': self.DaN_object.event_object.MTdict['Depth_MT'],
                                'USGS_Strike': self.DaN_object.event_object.strike_dip_rake['strike'][0],
                                'USGS_Dip': self.DaN_object.event_object.strike_dip_rake['dip'][0],
                                'USGS_Rake': self.DaN_object.event_object.strike_dip_rake['rake'][0],
                                'Nifgms': len(self.InSAR_codes),
                                'NFrames': len(self.DaN_object.geoc_final_path),
                                'Location_run_Lat': self.DaN_object.event_object.time_pos_depth['Position'][0],
                                'Location_run_Lon': self.DaN_object.event_object.time_pos_depth['Position'][1],
                                'InSAR_Lat': self.GBIS_lat,
                                'InSAR_Lon': self.GBIS_lon,
                                'InSAR_Depth TR': self.opt_model[2],
                                'InSAR_Depth Centroid': self.opt_model[2] + ((width*1/2)*np.sin(np.abs(self.opt_model[3])*(np.pi/180))),
                                'InSAR_Strike': strike,
                                'InSAR_Dip': -self.opt_model[3],
                                'InSAR_Rake': rake,
                                'InSAR_Slip': total_slip,
                                'Length': self.opt_model[0],
                                'Width': self.opt_model[1],
                                'InSAR_Mw':Mw,
                                }
        elif NP == 2:
            strike = self.opt_model[4] - 180 
            if strike < 360:
                strike  = strike + 360
            SS = self.opt_model[7]
            DS = self.opt_model[8]
            rake = np.degrees(np.arctan2(-DS,-SS))
            total_slip = np.sqrt(SS**2 + DS**2)
            mu = 3.2e10
            length = self.opt_model[0]
            width = self.opt_model[1]
            M0_assuming_mu = mu*length*width*total_slip
            Mw = (2/3)*np.log10(M0_assuming_mu*10**7) - 10.7
            dict_for_catalog ={'USGS_ID': self.DaN_object.event_object.ID,
                                'NP_selected': NP,
                                'USGS_Mag': self.DaN_object.event_object.MTdict['magnitude'],
                                'USGS_lat': USGS_latitude,
                                'USGS_lon': USGS_longitude,
                                'USGS_Depth': self.DaN_object.event_object.time_pos_depth['Depth'],
                                'USGS_Moment Depth': self.DaN_object.event_object.MTdict['Depth_MT'],
                                'USGS_Strike': self.DaN_object.event_object.strike_dip_rake['strike'][1],
                                'USGS_Dip': self.DaN_object.event_object.strike_dip_rake['dip'][1],
                                'USGS_Rake': self.DaN_object.event_object.strike_dip_rake['rake'][1],
                                'Nifgms': len(self.InSAR_codes),
                                'NFrames': len(self.DaN_object.geoc_final_path),
                                'Location_run_Lat': self.DaN_object.event_object.time_pos_depth['Position'][0],
                                'Location_run_Lon': self.DaN_object.event_object.time_pos_depth['Position'][1],
                                'InSAR_Lat': self.GBIS_lat,
                                'InSAR_Lon': self.GBIS_lon,
                                'InSAR_Depth TR': self.opt_model[2],
                                'InSAR_Depth Centroid': self.opt_model[2] + ((width*1/2)*np.sin(np.abs(self.opt_model[3])*(np.pi/180))),
                                'InSAR_Strike': strike,
                                'InSAR_Dip': -self.opt_model[3],
                                'InSAR_Rake': rake,
                                'InSAR_Slip': total_slip,
                                'Length': self.opt_model[0],
                                'Width': self.opt_model[1],
                                'InSAR_Mw':Mw}
        df = pd.DataFrame([dict_for_catalog]) 
        df.to_csv(path_or_buf=catalog)
        return
    def plot_locations(self):
        if os.path.isdir(self.path_to_data+"/locations"):
            print('Made it here its a directory')
            pass
        else:
            print('made it here')
            os.mkdir(self.path_to_data+"/locations/")
            print('created directory')

        locations= self.path_to_data + "/locations/locations"
        f = open(locations,"a")
        f.write(str(self.GBIS_lon) + " " + str(self.GBIS_lat) + " GBIS" + "\n")
        f.write(str(self.DaN_object.event_object.time_pos_depth['Position'][1]) + " " + str(self.DaN_object.event_object.time_pos_depth['Position'][0]) + " USGS")
        f.close()
        print('file has been made')
        print(self.DaN_object.geoc_path)
        dates = LiCS_tools.get_ifgdates(self.DaN_object.geoc_path[0])
        ifgm_dir = [] 
        for ii in range(len(dates)):
           ifgm_dir.append(self.DaN_object.geoc_path[0] +'/'+dates[ii])

        print(ifgm_dir)
        print(ifgm_dir[0])
        # dirs_with_ifgms, meta_file_paths = self.DaN_object.data_block.get_path_names(self.DaN_object.geoc_path)
        onlyfiles = [f for f in os.listdir(ifgm_dir[0]) if os.path.isfile(os.path.join(ifgm_dir[0], f))]
        tifs = []
        for file in onlyfiles:
            if ".tif" in file and '.unw' in file:
                full_path = os.path.join(ifgm_dir[0],file)
                tifs.append(full_path)
            else:
                pass 
        sp.check_call(["./plot_locations.sh",tifs[0],locations,self.path_to_data])
    
    
    def create_beachball_InSAR(self,NP):
        from obspy.imaging.beachball import beachball 
        strike = (self.opt_model[4] + 180) % 360
        # if strike > 360:
        #         strike  = strike - 360
        Dip = -self.opt_model[3]
        SS = self.opt_model[7]
        DS = self.opt_model[8]
        rake = np.degrees(np.arctan2(-DS,-SS))
        figure = plt.figure()
        figure.suptitle('InSAR Fault Mechanism  strike: ' + str(round(strike, 2))+' Dip: ' + str(round(Dip, 2)) + ' Rake: ' + str(round(rake, 2)))
        mt = [strike,Dip,rake]
        beachball(mt,size=200,linewidth=2,facecolor='r',fig=figure)
        figure.savefig(os.path.join(self.path_to_data,self.DaN_object.event_object.ID+'_InSAR_beachball_NP' +str(NP)+'.png'))
        return 
if __name__ == "__main__":
    # # example event ID's us6000jk0t, us6000jqxc, us6000kynh,
    # preproc_object = DaN.deformation_and_noise("us6000jk0t",target_down_samp=1500,inv_soft='GBIS',single_ifgm=True,date_primary=20230108,date_secondary=20230213,stack=False,look_for_gacos=True,NP=2)

    crashed_events = []

#     full_test = ['us60007anp',
#                     'us7000abmk',
#                     'us6000kynh',
#                     'us6000dyuk',
#                     'us7000gebb',
#                     'us6000b26j',
#                     'us7000fu12',
#                     'us7000df40',
#                     # 'us6000d3zh',
#                     'us6000jk0t',
#                     'us6000ddge',
#                     'us7000m3ur',
#                     'us7000lsze',generateReport
#                     'us6000dkmk',
#                     'us7000cet5',
#                     'us6000bdq8',
#                     'us6000a8nh',
#                     'nn00725272',
#                     'us70006sj8',
#                     'us600068w0']

#     full_test = [  
#                     'us7000lsze',
#                     'us6000dkmk',
#                     'us7000cet5',
#                     'us6000bdq8',
#                     'us6000a8nh',
#                     'nn00725272',
#                     'us60006a6i',
#                     'us70006d0m',
#                     'us60006rp9',
#                     'us7000fu12',

# ]

    # full_test_new = [   
    #                 # ,
    #                 'us7000m3ur',
    #                 'us7000lsze',
    #                 'us6000dkmk',
    #                 'us7000cet5',
    #                 'us6000bdq8',
    #                 'us6000a8nh',
    #                 'nn00725272',
    #                 'us70006sj8',
    #                 'us600068w0']
    # # ]

#     full_test = [    
                    
                  
#                     'us6000bdq8',
#                     'us6000a8nh',
                   
#                     # ,
#                     'us60006rp9',
#                     'us70006d0m',
#                     'us60006a6i',
#                     'us7000lsze',
#                     'us7000fu12',
#                     'us6000dkmk',
#    ]
    #            
               
    #             'us60007anp',]
    array = ['us6000a8nh']
    df = pd.read_csv('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/gCent_Catalog_no_header_info.csv')
    df_china = df[df['Location'] == 'China']
    df_Iran = df[df['Location'] == 'Iran']
    df_turkey = df[df['Location'] == 'Turkey']
    
    full_test = array + list(df_china.ComCatID)+ list(df_Iran.ComCatID) + list(df_turkey.ComCatID) 
    
    # full_test = ['us7000g9zq']
    # full_test = ['us6000a8nh']
    # full_test = ['us70008cld']
    # full_test = ['us600068w0']
    
    # full_test = ['us70006sj8']

    full_test = ['us70006sj8',
                'us7000g9zq',
                'us6000a8nh',
                'us70008cld',
                'us600068w0',
                'us70006d0m'
    ]

    full_test = full_test + list(df_china.ComCatID)+ list(df_Iran.ComCatID) + list(df_turkey.ComCatID) 

    # full_test = ['us70006d0m']
    # print(full_test)
    # full_test = ['us6000jk0t']
    # full_test = ['us6000mjpj']

    # full_test = ['us7000ljvg',
    #             'us7000dkv0',
    #             ]
    # ]
    # full_test = ['us6000jk0t']
    # test_events = ['us7000kynh',
    #                 'us7000abmk']

    # test_events = ['us7000abmk'] # edge tester 

    # afghgan_event = ['us6000len8']
    # test_events = ['us6000jk0t']
    # test_events = afghgan_event
    # Frame = ['092D_05554_141414']
    GBIS_run = True
    locations_usgs = []
    locations_reloc_NP1 = []
    locations_reloc_NP2 = []  
    opt_models_NP1 = [] 
    opt_models_NP2 = [] 
    for ii in range(len(full_test)):
        try:
            print('RUNNING AGAIN ')
            # with open(full_test[ii]+'_preproc.pkl', 'wb') as outp:
            preproc_object = DaN.deformation_and_noise(full_test[ii],
                                                    target_down_samp=1000,
                                                    inv_soft='GBIS',
                                                    look_for_gacos=True,
                                                    # NP=2,
                                                    all_coseis=True,
                                                    single_ifgm=False,
                                                    coherence_mask=0.1,
                                                    min_unw_coverage=0.3, 
                                                    pygmt=True,
                                                    # date_primary=20200531,
                                                    # date_secondary=20200718,
                                                    # frame=Frame,
                                                    scale_factor_mag=0.075,
                                                    scale_factor_depth=0.075,
                                                    scale_factor_clip_mag=0.25,
                                                    scale_factor_clip_depth=0.0055,
                                                    loop_processing_flow=True
                                                    )
            preproc_object_copy = preproc_object
        
            GBIS_object = auto_GBIS(preproc_object,'./GBIS.location',NP=1,number_trials=1e5,
                                        pygmt=False,generateReport=True,location_search=False,limit_trials=False)

            shutil.move(GBIS_object.outputdir,GBIS_object.outputdir + '_location_run')
            locations_usgs.append([float(preproc_object.event_object.time_pos_depth['Position'][0]),float(preproc_object.event_object.time_pos_depth['Position'][1])])
            preproc_object.event_object.time_pos_depth['Position'][0] = GBIS_object.GBIS_lat
            preproc_object.event_object.time_pos_depth['Position'][1] = GBIS_object.GBIS_lon
            # preproc_object.scale_factor_depth = 0.065
            # preproc_object.scale_factor_mag = 0.065
            shutil.copytree(preproc_object.event_object.LiCS_locations,preproc_object.event_object.LiCS_locations + '_USGS_location_used')
            preproc_object.flush_processing()
            preproc_object.geoc_path, preproc_object.gacos_path = preproc_object.data_block.pull_frame_coseis()
            attempt = 0
            while attempt < 4: # incase data is pulled incorrectly or copied wrong
                try:
                   
                    preproc_object.run_processing_flow(preproc_object.geoc_path, preproc_object.gacos_path,True)
                    preproc_object.geoc_final_path = preproc_object.geoc_ds_path
                    preproc_object.move_final_output()
                    attempt = 5
                except Exception as e:
                    preproc_object.flush_processing()
                    preproc_object.geoc_path, preproc_object.gacos_path = preproc_object.data_block.pull_frame_coseis()
                    preproc_object.check_data_pull()
                    preproc_object.geoc_path =  preproc_object.check_geoc_has_data(preproc_object.geoc_path)
                    attempt+=1
                    print(e)
                    print('processing flow failed trying again with attempt number ' + str(attempt))
                    if attempt > 4:
                        print('Failed 5 times on processing flow check errors')

            # preproc_object.run_processing_flow(preproc_object.geoc_path,preproc_object.gacos_path,True)
           
                # pickle.dump(preproc_object,outp)
             
                    # print('GBIS Initial run failed using usgs location for subsequent runs')
                    # preproc_object.flush_processing()
                    # attempt = 0
                    # while attempt < 4: # incase data is pulled incorrectly or copied wrong
                    #     try:
                    #         preproc_object.run_processing_flow( preproc_object.geoc_path, preproc_object.gacos_path,look_for_gacos)
                    #         attempt = 5
                    #     except:
                    #         preproc_object.flush_processing()
                    #         preproc_object.check_data_pull()
                    #         preproc_object.geoc_path =  preproc_object.check_geoc_has_data( preproc_object.geoc_path)
                    #         attempt+=1
                    #         print('processing flow failed trying again with attempt number ' + str(attempt))
                    #         if attempt > 4:
                    #             print('Failed 5 times on processing flow check errors')
                  
                    # preproc_object.geoc_final_path = preproc_object.geoc_ds_path
                    # preproc_object.move_final_output()
                    # pickle.dump(preproc_object,outp)



# if GBIS_run == True: 
# for ii in range(len(full_test[ii])):
#     with open(full_test[ii]+'_preproc.pkl', 'rb') as inp:
        # preproc_object = pickle.load(inp)
        
            try:
                GBIS_object = auto_GBIS(preproc_object,'./GBIS.location',NP=1,number_trials=1e5,pygmt=True,limit_trials=False)
                locations_reloc_NP1.append([GBIS_object.GBIS_lat,GBIS_object.GBIS_lon]) 
                opt_models_NP1.append(GBIS_object.opt_model)
            except Exception as e:
                    print(e)
                    pass 
            try:
                print('I am in this try')
                GBIS_object = auto_GBIS(preproc_object,'./GBIS.location',NP=2,number_trials=1e5,pygmt=True,limit_trials=False)
                locations_reloc_NP2.append([GBIS_object.GBIS_lat,GBIS_object.GBIS_lon]) 
                opt_models_NP2.append(GBIS_object.opt_model)
            except Exception as e:
                print(e)
                pass 
            # shutil.rmtree(preproc_object.event_object.LiCS_locations)
            # shutil.move(preproc_object.event_object.LiCS_locations + '_USGS_location_used',preproc_object.event_object.LiCS_locations)

        except Exception as e:
            print(e)
            crashed_events.append(full_test[ii])
            print(crashed_events)
        ### Location plots ####

    import pygmt
    fig = pygmt.Figure()
    print(np.array(locations_usgs))
    print(np.array(locations_reloc_NP1))
    print(np.array(locations_reloc_NP2))
    # Make a global Mollweide map with automatic ticks
    fig.basemap(region="d", projection="W20c", frame=True)
    fig.coast(
    region="d",
    # projection="Cyl_stere/12c",
    projection = "W20c",
    land="darkgray",
    water="white",
    borders="1/0.5p",
    shorelines="1/0.5p",
    # frame="a",
    )
    font = "2p,Helvetica-Bold"
    fig.plot(x=np.array(locations_usgs)[:,1],
                        y=np.array(locations_usgs)[:,0],
                        pen='1p,black',
                        fill='darkorange',
                        style='a',
                        # transparency=80,
                        region='d',
                        projection='W20c',
            )
    
    fig.plot(x=np.array(locations_reloc_NP1)[:,1],
                        y=np.array(locations_reloc_NP1)[:,0],
                        pen='1p,black',
                        fill='lightblue',
                        style='a',
                        # transparency=80,
                        region='d',
                        projection='W20c',
            )
    
    fig.plot(x=np.array(locations_reloc_NP2)[:,1],
                        y=np.array(locations_reloc_NP2)[:,0],
                        pen='1p,black',
                        fill='blue',
                        style='a',
                        # transparency=80,
                        region='d',
                        projection='W20c',
            )
    fig.text(x=np.array(locations_reloc_NP2)[:,1], y=np.array(locations_reloc_NP2)[:,0] + 1, text="InSAR Location", font=font)
    fig.text(x=np.array(locations_usgs)[:,1], y=np.array(locations_usgs)[:,0] + 1, text="USGS Location", font=font)
    fig.savefig('global_location_plot_comp.png')





    # preproc_object = DaN.deformation_and_noise("us6000ldpg",target_down_samp=2500,inv_soft='GBIS',look_for_gacos=True,NP=1,all_coseis=False,date_primary=20230925,date_secondary=20231008,single_ifgm=False,frame=['020D_05533_131313','013A_05597_131313'],coherence_mask=0.001,pygmt=False)
    # preproc_object = DaN.deformation_and_noise("us6000kynh",target_down_samp=1500,inv_soft='GBIS',look_for_gacos=True,NP=2,all_coseis=True) # Tanzania 
    # preproc_object = DaN.deformation_and_noise("us6000d3zh",target_down_samp=1500,inv_soft='GBIS',look_for_gacos=True,NP=2,all_coseis=True) # Croatia
    # preproc_object = DaN.deformation_and_noise("us7000g9zq",target_down_samp=2000,inv_soft='GBIS',single_ifgm=True,date_primary=20211224,date_secondary=20220117,stack=False,look_for_gacos=True,frame='128A_05172_131313',NP=2)
    # preproc_object = DaN.deformation_and_noise("us6000bdq8",target_down_samp=1500,inv_soft='GBIS',single_ifgm=True,date_primary=20200713,date_secondary=20200830,stack=False,all_coseis=True)
    # preproc_object = DaN.deformation_and_noise("us6000bdq8",target_down_samp=1500,inv_soft='GBIS',single_ifgm=True,date_primary=None,date_secondary=None,stack=False,all_coseis=True)
    # preproc_object = DaN.deformation_and_noise("us6000ldpg",target_down_samp=1500,inv_soft='GBIS',date_primary=20230926,date_secondary=20231008,frame='020D_05533_131313')
    # preproc_object = DaN.deformation_and_noise("us7000ljvg",target_down_samp=1500,inv_soft='GBIS',date_primary=20231202,date_secondary=20231226,frame='135D_05421_131313')
    # preproc_object = DaN.deformation_and_noise("us6000lfn5",target_down_samp=1500,inv_soft='GBIS',date_primary=20231007,date_secondary=20231019,frame='013A_05597_131313',single_ifgm=True)
    # GBIS_object = auto_GBIS(preproc_object,'/Users/jcondon/phd/code/auto_inv/GBIS.location',NP=1,number_trials=1e5)
    # GBIS_object = auto_GBIS(preproc_object,'/home/ee18jwc/code/auto_inv/GBIS.location',NP=1,number_trials=1e5)
   


