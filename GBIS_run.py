import matlab.engine
import sys 
from scipy.io import loadmat
import pandas as pd
import numpy as np
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


class auto_GBIS:
    def __init__(self,deformation_and_noise_object,GBIS_loc,NP=1,number_trials=1e6):
        self.DaN_object = deformation_and_noise_object
        self.path_to_data = self.DaN_object.event_object.GBIS_location
        self.number_trials = number_trials
        
       
        self.npzfiles = self.get_data_location()
        self.GBIS_mat_file_location, self.sill_nug_range = self.convert_GBIS_Mat_format()
        self.show_noise_params()
        self.path_to_GBIS = self.read_input(GBIS_loc)
        self.eng = self.start_matlab_set_path()

        self.estimate_length = self.calc_square_start()
        self.max_dist = self.maximum_dist()
        self.boundingbox = self.calc_boundingbox()
        self.create_insar_input()
        self.edit_input_priors(NP=NP)
        self.opt_model, self.GBIS_lon, self.GBIS_lat = self.gbisrun()
        # self.plot_locations()
        self.create_catalog_entry(NP)
        self.gmt_output(NP)
       
      

    def convert_GBIS_Mat_format(self):
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
        alltext = [] 
        f = open(GBIS_loc)
        for line in f:
            if line.startswith('#'):
                continue 
            else:
                alltext.append(line)
        path = alltext[0].strip('\n')
        return path 
    
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
        
        boundingbox = [np.min(min_lons),np.max(max_lats),np.max(max_lons),np.min(min_lats)]
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
        max_dist = np.mean(max_dists)
            
        return max_dist
    
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

    def create_insar_input(self):
        input_loc = self.DaN_object.event_object.GBIS_insar_template

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

        with open(input_loc,'r') as file:
            lines = file.readlines() 
        #     file.readlines() 
        # write = [] 
        for ii in range(len(lines)):
            if any('insarID' in x for x in lines[ii]):
                lines[ii] = ' '
            else:
                pass 
        
          
        strings_to_write = [] 
        for ii in range(len(self.data)):
            strings_to_write.append('insarID = ' + str(ii + 1) + ';' + '\n' +
                                'insar{insarID}.dataPath = ' +' \'' + self.GBIS_mat_file_location[ii] + '\'' +';' + '\n' + 
                                'insar{insarID}.wavelength = 0.056;' + '\n' +
                                'insar{insarID}.constOffset = \'y\';' + '\n' +
                                'insar{insarID}.rampFlag = \'y\';' +   '\n' +
                                'insar{insarID}.sillExp =' + str(self.sill_nug_range[ii][0]) +';' + '\n' +
                                'insar{insarID}.range =' + str(self.sill_nug_range[ii][2]) +';' + '\n' +
                                'insar{insarID}.nugget=' + str(self.sill_nug_range[ii][1]) +';' + '\n'+ 
                                'insar{insarID}.quadtreeThresh = 0;' +'\n')
            
        with open(input_loc, 'w') as file:
            file.writelines(lines)
            file.writelines(strings_to_write)
    
        return 

    def edit_input_priors(self,NP=1):
        input_loc = self.DaN_object.event_object.GBIS_insar_template
        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        name = params[0].split('=')[-1]
        time = params[1].split('=')[-1]
        latitude = float(params[2].split('=')[-1])
        longitude = float(params[3].split('=')[-1])
        magnitude = float(params[4].split('=')[-1])
        moment = float(params[5].split('=')[-1])
        depth = float(params[6].split('=')[-1])
        catalog = params[7].split('=')[-1]
        strike1 = float(params[8].split('=')[-1])
        dip1 = float(params[9].split('=')[-1])
        rake1 = float(params[10].split('=')[-1])
        strike2 = float(params[11].split('=')[-1])
        dip2 = float(params[12].split('=')[-1])
        rake2 = float(params[13].split('=')[-1])
        # if -dip1 - 30 < -90:
        #     dip1 = 0 
        dip1_lower = -dip1 - 20
        dip2_lower = -dip2 - 20
        if dip1_lower < -89.9:
            dip1_lower = -89.9 
        if dip2_lower < -89.9:
            dip2_lower = -89.9

        dip1_upper = -dip1 + 20
        dip2_upper = -dip2 + 20
        if dip1_upper > -0.1:
            dip1_upper = -0.1 
        if dip2_upper > -0.1:
            dip2_upper = -0.1




        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  " + str(strike2))
        #Convert from USGS to GBIS convention 
        strike1 = strike1 - 180 
        strike2 = strike2 - 180 

        if strike1 < 0: 
            strike1 = strike1 + 360 
        if strike2 < 0:
            strike2 = strike2 + 360 
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  " + str(strike2))

        strike_upper1 = strike1 + 60   
        strike_upper2 = strike2 + 60
        strike_lower1 = strike1 - 60 
        strike_lower2 = strike2 - 60 

        if strike_upper1 > 360:
            strike_upper1 = strike_upper1 - 360 
        if strike_upper2 > 360:
            strike_upper2 = strike_upper2 - 360

        if strike_lower1 < 0: 
            strike_lower1 = 360 + strike_lower1 
        if strike_lower2 < 0:
            strike_lower2 = 360 + strike_lower2

        if strike1 < strike_upper1 and strike1 < strike_lower1:
            if strike_upper1 > strike_lower1:
                strike_upper1 = strike_upper1 - 360 
            else: 
                strike_lower1 = strike_lower1 - 360 
        
        if strike2 < strike_upper2 and strike2 < strike_lower2:
            if strike_upper2 > strike_lower2:
                strike_upper2 = strike_upper2 - 360 
            else: 
                strike_lower2 = strike_lower2 - 360 



        # sq_dim = self.calc_square_start()
        print("################################################")
        # print(str(int(np.max(self.data['lonlat_m'][:,0]))))
        # print(str(int(np.max(self.data['lonlat_m'][:,1]))))
        # print(self.sill_range_nug)

        # print(len(self.data['lonlat_m'][:,1]))
        # print(np.shape(self.data['lonlat_m']))
        print(self.GBIS_mat_file_location)
        print(input_loc)
        print('     L       W      Z     Dip     Str      X       Y      SS       DS ')
        with open(input_loc,'r') as file:
            lines = file.readlines() 
        for ii in range(len(lines)):
            # print(lines[ii])
            if 'geo.referencePoint' in lines[ii]:
                lines[ii] = ('geo.referencePoint =['
                    + str(longitude) + ";"
                    + str(latitude) + "];" + '\n')
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
                if strike_lower1 < strike_upper1:
                    strike_bound = strike_lower1
                else:
                    strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.8)) + ';  ' 
                            + str(int(self.estimate_length*0.8)) + ';  '
                            + str(int(depth*0.01)) + ';  '
                            + str(dip1_lower) + ';  '
                            + str(int(strike_bound)) + ';  '
                            + str(int(-self.max_dist/4)) + ';  '
                            + str(int(-self.max_dist/4)) + ';  '
                            + str(-10.0) + ';  '
                            + str(-10.0) + '];'
                            +'\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.lower' in lines[ii] and NP==2:
                if strike_lower2 < strike_upper2:
                    strike_bound = strike_lower2
                else:
                    strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.lower=['
                            + str(int(self.estimate_length*0.8)) + ';  ' 
                            + str(int(self.estimate_length*0.8)) + ';  '
                            + str(int(depth*0.01)) + ';  '
                            + str(int(dip2_lower)) + ';  '
                            + str(int(strike_bound)) + ';  '
                            + str(int(-self.max_dist/4)) + ';  '
                            + str(int(-self.max_dist/4)) + ';  '
                            + str(-10.0) + ';  '
                            + str(-10.0) + '];'
                            '\n'
                            )
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==1:
                if strike_lower1 > strike_upper1:
                    strike_bound = strike_lower1
                else:
                    strike_bound = strike_upper1
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*2.5)) + ';  ' 
                            + str(int(self.estimate_length*2.5)) + ';  '
                            + str(int(depth*3)) + ';  '
                            + str(int(dip1_upper)) + ';  '
                            + str(int(strike_bound)) + ';  '
                            + str(int(self.max_dist/4)) + ';  '
                            + str(int(self.max_dist/4)) + ';  '
                            + str(10.0) + ';  '
                            + str(10.0) + '];'
                            '\n'
                            ) 
                print(lines[ii])
            elif 'modelInput.fault.upper' in lines[ii] and NP==2:
                if strike_lower2 > strike_upper2:
                    strike_bound = strike_lower2
                else:
                    strike_bound = strike_upper2
                lines[ii] = ('modelInput.fault.upper=['
                            + str(int(self.estimate_length*2.5)) + ';  ' 
                            + str(int(self.estimate_length*2.5)) + ';  '
                            + str(int(depth*3)) + ';  '
                            + str(int(dip2_upper)) + ';  '
                            + str(int(strike_bound)) + ';  '
                            + str(int(self.max_dist/4)) + ';  '
                            + str(int(self.max_dist/4)) + ';  '
                            + str(10.0) + ';  '
                            + str(10.0) + '];'
                            '\n'
                            ) 
                print(lines[ii])
        with open(input_loc, 'w') as file:
            file.writelines(lines)
        return
    

    def gbisrun(self):
        cwd = os.getcwd()
        # os.chdir(self.DaN_object.event_object.GBIS_location)
        if len(self.data) == 1:
            self.InSAR_codes = 1
        else:
            self.InSAR_codes = np.arange(len(self.data) + 1)[1:-1]
            InSAR_codes_string = "invert_"
            for ii in range(len(self.InSAR_codes)):
                InSAR_codes_string = InSAR_codes_string + str(self.InSAR_codes[ii]) + "_"
            InSAR_codes_string = InSAR_codes_string + "F"
        print(self.InSAR_codes)
        self.outputdir= "./" + self.DaN_object.event_object.GBIS_insar_template.split('/')[-1][:-4] + "/" + InSAR_codes_string
        self.outputfile = self.outputdir + "/" + InSAR_codes_string +".mat"
        self.opt_model_vertex = self.outputdir +'/' + 'optmodel_vertex.mat'
        if os.path.isdir(self.outputdir):
            print("Inversion Results already present skipping")
        else:
           self.eng.GBISrun(self.DaN_object.event_object.GBIS_insar_template,self.InSAR_codes,'n','F',self.number_trials,'n','n','n',nargout=0)
    
        output_matrix = loadmat(self.outputfile,squeeze_me=True)
        opt_model = np.array(output_matrix['invResults'][()].item()[-1])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~optimal model results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(opt_model)
        print(np.size(opt_model))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~optimal model results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if os.path.isdir(self.outputdir): # temporary skip
            pass 
        else:
            self.eng.generateFinalReport(self.outputfile,self.number_trials*0.2,nargout=0)
        # print(opt_model)
        print([opt_model,'color','k','updipline','yes','projection','3D','origin',[self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],'final','yes','savedir',self.opt_model_vertex])
        self.eng.python_comp_drawmodel(self.outputfile,[float(self.DaN_object.event_object.time_pos_depth['Position'][1]),float(self.DaN_object.event_object.time_pos_depth['Position'][0])],self.opt_model_vertex,nargout=0)
        Lon_GBIS, Lat_GBIS = self.eng.convert_location_output(self.GBIS_mat_file_location[0],self.outputfile,nargout=2)
        print(Lon_GBIS,Lat_GBIS)
        # os.chdir(cwd)
        return opt_model, Lon_GBIS, Lat_GBIS
    
    def gmt_output(self,NP):
        input_loc = self.DaN_object.event_object.GBIS_insar_template
        with open(self.DaN_object.event_object.event_file_path,'r') as file:
            params = file.readlines()
        name = params[0].split('=')[-1]
        time = params[1].split('=')[-1]
        latitude = float(params[2].split('=')[-1])
        longitude = float(params[3].split('=')[-1])
        magnitude = float(params[4].split('=')[-1])
        moment = float(params[5].split('=')[-1])
        depth = float(params[6].split('=')[-1])
        catalog = params[7].split('=')[-1]
        strike1 = float(params[8].split('=')[-1])
        dip1 = float(params[9].split('=')[-1])
        rake1 = float(params[10].split('=')[-1])
        strike2 = float(params[11].split('=')[-1])
        dip2 = float(params[12].split('=')[-1])
        rake2 = float(params[13].split('=')[-1])
        slip_rate=5.5e-5
        slip = self.estimate_length * slip_rate
        if NP ==1:
            usgs_model = [ 0,0,strike1,dip1,rake1,slip,self.estimate_length,depth,self.estimate_length]
        elif NP == 2:
            usgs_model = [ 0,0,strike2,dip2,rake2,slip,self.estimate_length,depth,self.estimate_length]

        if isinstance(self.DaN_object.geoc_final_path,list):
            for ii in range(len(self.DaN_object.geoc_final_path)):
                final_path = self.DaN_object.geoc_final_path[ii] + "_INVERSION_Results"
                op.produce_final_GBISoutput(self.DaN_object.geoc_clipped_path[ii],
                                            final_path,
                                            self.opt_model,
                                            self.opt_model_vertex,
                                            [self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],
                                            usgs_model)
            else:
                final_path = self.DaN_object.geoc_final_path[0] + "_INVERSION_Results"
                op.produce_final_GBISoutput(self.DaN_object.geoc_clipped_path[0],
                                            final_path,
                                            self.opt_model,
                                            self.opt_model_vertex,
                                            [self.DaN_object.event_object.time_pos_depth['Position'][1],self.DaN_object.event_object.time_pos_depth['Position'][0]],
                                            usgs_model)

        return 
    def create_catalog_entry(self,NP):
        if os.path.isdir(self.path_to_data+'/'+ self.DaN_object.event_object.ID+'_catalog_entry'):
            print('Made it here its a directory')
            pass 
        else:
            print('made it here')
            os.mkdir(self.path_to_data+'/'+ self.DaN_object.event_object.ID+'_catalog_entry/')
            print('created directory')

        catalog= self.path_to_data+'/'+ self.DaN_object.event_object.ID+'_catalog_entry' + '/' +self.DaN_object.event_object.ID+ '_entry.csv'
        if NP == 1:
            strike = self.opt_model[4] - 180 
            if strike < 360:
                strike  = strike + 360
            SS = self.opt_model[7]
            DS = self.opt_model[8]
            rake = np.degrees(np.arctan(DS/SS))
            total_slip = np.sqrt(SS**2 + DS**2)
            mu = 3.2e10
            length = self.opt_model[0]
            width = self.opt_model[1]
            M0_assuming_mu = mu*length*width*total_slip
            Mw = (2/3)*np.log10(M0_assuming_mu) - 9.1 
            dict_for_catalog ={'USGS ID': self.DaN_object.event_object.ID,
                               'NP selected': NP,
                                'USGS Mag': self.DaN_object.event_object.MTdict['magnitude'],
                                'USGS Lat': self.DaN_object.event_object.time_pos_depth['Position'][0],
                                'USGS Lon': self.DaN_object.event_object.time_pos_depth['Position'][1],
                                'USGS Depth': self.DaN_object.event_object.time_pos_depth['Depth'],
                                'USGS Moment Depth': self.DaN_object.event_object.MTdict['Depth_MT'],
                                'USGS Strike': self.DaN_object.event_object.strike_dip_rake['strike'][0],
                                'USGS Dip': self.DaN_object.event_object.strike_dip_rake['dip'][0],
                                'USGS Rake': self.DaN_object.event_object.strike_dip_rake['rake'][0],
                                'Nifgms': len(self.InSAR_codes),
                                'NFrames': len(self.DaN_object.geoc_final_path),
                                'InSAR Lat': self.GBIS_lat,
                                'InSAR Lon': self.GBIS_lon,
                                'InSAR Depth TR': self.opt_model[2],
                                'InSAR Depth Centroid': self.opt_model[2] + ((width*1/2)*np.sin(np.abs(self.opt_model[3])*(np.pi/180))),
                                'InSAR Strike': strike,
                                'InSAR Dip': -self.opt_model[3],
                                'InSAR Rake': rake,
                                'InSAR Slip': total_slip,
                                'Length': self.opt_model[0],
                                'Width': self.opt_model[1],
                                'InSAR Mw':Mw,
                                }
        elif NP == 2:
            strike = self.opt_model[4] - 180 
            if strike < 360:
                strike  = strike + 360
            SS = self.opt_model[7]
            DS = self.opt_model[8]
            rake = np.degrees(np.arctan(DS/SS))
            total_slip = np.sqrt(SS**2 + DS**2)
            mu = 3.2e10
            length = self.opt_model[0]
            width = self.opt_model[1]
            M0_assuming_mu = mu*length*width*total_slip
            Mw = (2/3)*np.log10(M0_assuming_mu) - 9.1 
            dict_for_catalog ={'USGS ID': self.DaN_object.event_object.ID,
                                'USGS Mag': self.DaN_object.event_object.MTdict['magnitude'],
                                'USGS Lat': self.DaN_object.event_object.time_pos_depth['Position'][0],
                                'USGS Lon': self.DaN_object.event_object.time_pos_depth['Position'][1],
                                'USGS Depth': self.DaN_object.event_object.time_pos_depth['Depth'],
                                'USGS Moment Depth': self.DaN_object.event_object.MTdict['Depth_MT'],
                                'USGS Strike': self.DaN_object.event_object.strike_dip_rake['strike'][1],
                                'USGS Dip': self.DaN_object.event_object.strike_dip_rake['dip'][1],
                                'USGS Rake': self.DaN_object.event_object.strike_dip_rake['rake'][1],
                                'Nifgms': len(self.InSAR_codes),
                                'NFrames': len(self.DaN_object.geoc_final_path),
                                'InSAR Lat': self.GBIS_lat,
                                'InSAR Lon': self.GBIS_lon,
                                'InSAR Depth TR': self.opt_model[2],
                                'InSAR Depth Centroid': self.opt_model[2] + ((width*1/2)*np.sin(np.abs(self.opt_model[3])*(np.pi/180))),
                                'InSAR Strike': strike,
                                'InSAR Dip': -self.opt_model[3],
                                'InSAR Rake': rake,
                                'InSAR Slip': total_slip,
                                'Length': self.opt_model[0],
                                'Width': self.opt_model[1],
                                'InSAR Mw':Mw}
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
    
if __name__ == "__main__":
    # # example event ID's us6000jk0t, us6000jqxc, us6000kynh,
    # preproc_object = DaN.deformation_and_noise("us6000jk0t",target_down_samp=1500,inv_soft='GBIS',single_ifgm=True,date_primary=20230108,date_secondary=20230213,stack=False,look_for_gacos=True,NP=2)
    preproc_object = DaN.deformation_and_noise("us6000lfn5",target_down_samp=1500,inv_soft='GBIS',look_for_gacos=True,NP=1,all_coseis=False,date_primary=20231007,date_secondary=20231101,single_ifgm=False,frame=['013A_05597_131313','020D_05533_131313']) # Iran_turkey 
    # preproc_object = DaN.deformation_and_noise("us6000kynh",target_down_samp=1500,inv_soft='GBIS',look_for_gacos=True,NP=2,all_coseis=True) # Tanzania 
    # preproc_object = DaN.deformation_and_noise("us6000d3zh",target_down_samp=1500,inv_soft='GBIS',look_for_gacos=True,NP=2,all_coseis=True) # Croatia
    # preproc_object = DaN.deformation_and_noise("us7000g9zq",target_down_samp=2000,inv_soft='GBIS',single_ifgm=True,date_primary=20211224,date_secondary=20220117,stack=False,look_for_gacos=True,frame='128A_05172_131313',NP=2)
    # preproc_object = DaN.deformation_and_noise("us6000bdq8",target_down_samp=1500,inv_soft='GBIS',single_ifgm=True,date_primary=20200713,date_secondary=20200830,stack=False,all_coseis=True)
    # preproc_object = DaN.deformation_and_noise("us6000bdq8",target_down_samp=1500,inv_soft='GBIS',single_ifgm=True,date_primary=None,date_secondary=None,stack=False,all_coseis=True)
    # preproc_object = DaN.deformation_and_noise("us6000ldpg",target_down_samp=1500,inv_soft='GBIS',date_primary=20230926,date_secondary=20231008,frame='020D_05533_131313')
    # preproc_object = DaN.deformation_and_noise("us7000ljvg",target_down_samp=1500,inv_soft='GBIS',date_primary=20231202,date_secondary=20231226,frame='135D_05421_131313')
    # preproc_object = DaN.deformation_and_noise("us6000lfn5",target_down_samp=1500,inv_soft='GBIS',date_primary=20231007,date_secondary=20231019,frame='013A_05597_131313',single_ifgm=True)
    # GBIS_object = auto_GBIS(preproc_object,'/Users/jcondon/phd/code/auto_inv/GBIS.location',NP=1,number_trials=1e5)
    GBIS_object = auto_GBIS(preproc_object,'/Users/jcondon/phd/code/auto_inv/GBIS.location',NP=1,number_trials=1e5)



