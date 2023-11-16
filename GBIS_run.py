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


class auto_GBIS:
    def __init__(self,deformation_and_noise_object,GBIS_loc):
        self.DaN_object = deformation_and_noise_object
        self.path_to_data = self.DaN_object.event_object.GBIS_location
       
        self.npzfiles = self.get_data_location()
        self.GBIS_mat_file_location, self.sill_range_nug = self.convert_GBIS_Mat_format()

        self.path_to_GBIS = self.read_input(GBIS_loc)
        self.eng = self.start_matlab_set_path()

        self.estimate_length = self.calc_square_start()
        self.boundingbox = self.calc_boundingbox()
        self.edit_input_priors()
        self.gbisrun()
      

    def convert_GBIS_Mat_format(self):
        self.data = np.load(self.npzfiles[0])
        print(self.data.files)
        
        Phase = self.data['ph_disp'].T
        Lat = self.data['lonlat'][:,1].T
        Lon = self.data['lonlat'][:,0].T
        Inc = self.data['la'].T
        Heading = self.data['heading'].T
        print(np.shape(Phase))
        sill_range_nug = self.data['sill_range_nug']
        tmpnpz_gbis_format =self.npzfiles[0][:-3] + 'GBIS.npz'
        np.savez(tmpnpz_gbis_format, Phase=Phase,Lat=Lat,Lon=Lon,Inc=Inc,Heading=Heading)
        LiCS_lib.npz2mat(tmpnpz_gbis_format)
        GBIS_mat_file_location = tmpnpz_gbis_format[:-3] +'mat'
        return GBIS_mat_file_location ,sill_range_nug
     

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
        max_lat = np.max(self.data['lonlat'][:,1])
        min_lat = np.min(self.data['lonlat'][:,1])
        max_lon = np.max(self.data['lonlat'][:,0])
        min_lon = np.min(self.data['lonlat'][:,0])
        boundingbox = [min_lon,max_lat,max_lon,min_lat]
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
        if len(npzfiles) > 1:

            print("Two mat files presenent please remove one")
        return npzfiles
    
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
        # sq_dim = self.calc_square_start()
        print("################################################")
        print(str(int(np.max(self.data['lonlat_m'][:,0]))))
        print(str(int(np.max(self.data['lonlat_m'][:,1]))))
        print(self.GBIS_mat_file_location)
        print(input_loc)

        with open(input_loc,'r') as file:
    
            lines = file.readlines() 
        for ii in range(len(lines)):
            print(lines[ii])
            if 'insar{insarID}.dataPath' in lines[ii]:
                print('Made it to this line')
                lines[ii] = 'insar{insarID}.dataPath = ' +' \'' +self.GBIS_mat_file_location + '\'' +';'
            elif 'insar{insarID}.sillExp' in lines[ii]:
                lines[ii] = 'insar{insarID}.sillExp =' + str(self.sill_range_nug[0]) +';'
            elif 'insar{insarID}.range' in lines[ii]:
                lines[ii] =  'insar{insarID}.range=' + str(self.sill_range_nug[1]) +';'
            elif 'insar{insarID}.nugget' in lines[ii]:
                lines[ii] =  'insar{insarID}.nugget=' + str(self.sill_range_nug[2]) +';'
            elif 'geo.referencePoint' in lines[ii]:
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
                            + str(self.estimate_length) + ';  ' 
                            + str(self.estimate_length) + ';  '
                            + str(depth) + ';  '
                            + str(dip1) + ';  '
                            + str(strike1) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            +'\n'
                            )
            elif 'modelInput.fault.start' in lines[ii] and NP==2:
                lines[ii] = ('modelInput.fault.start=['
                            + str(self.estimate_length) + ';  ' 
                            + str(self.estimate_length) + ';  '
                            + str(depth) + ';  '
                            + str(dip2) + ';  '
                            + str(strike2) + ';  '
                            + str(0) + ';  '
                            + str(0) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            +'\n'
                            )
            elif 'modelInput.fault.step' in lines[ii] and NP==1:
                pass 
            elif 'modelInput.fault.step' in lines[ii] and NP==2:
                pass 
            elif 'modelInput.fault.lower' in lines[ii] and NP==1:
                lines[ii] = ('modelInput.fault.lower=['
                            + str(self.estimate_length*0.25) + ';  ' 
                            + str(self.estimate_length*0.25) + ';  '
                            + str(depth*0.25) + ';  '
                            + str(dip1*0.25) + ';  '
                            + str(strike1*0.25) + ';  '
                            + str(int(np.min(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(int(np.min(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            +'\n'
                            )
            elif 'modelInput.fault.lower' in lines[ii] and NP==2:
                lines[ii] = ('modelInput.fault.lower=['
                            + str(self.estimate_length*0.25) + ';  ' 
                            + str(self.estimate_length*0.25) + ';  '
                            + str(depth*0.25) + ';  '
                            + str(dip2*0.25) + ';  '
                            + str(strike2*0.25) + ';  '
                            + str(int(np.min(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(int(np.min(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            '\n'
                            )
            elif 'modelInput.fault.upper' in lines[ii] and NP==1:
                lines[ii] = ('modelInput.fault.upper=['
                            + str(self.estimate_length*2) + ';  ' 
                            + str(self.estimate_length*2) + ';  '
                            + str(depth) + ';  '
                            + str(dip1) + ';  '
                            + str(strike1) + ';  '
                            + str(int(np.max(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(int(np.max(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            '\n'
                            ) 
            elif 'modelInput.fault.upper' in lines[ii] and NP==2:
                lines[ii] = ('modelInput.fault.upper=['
                            + str(self.estimate_length*2) + ';  ' 
                            + str(self.estimate_length*2) + ';  '
                            + str(depth) + ';  '
                            + str(dip1) + ';  '
                            + str(strike1) + ';  '
                            + str(int(np.max(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(int(np.max(self.data['lonlat_m'][:,0]))) + ';  '
                            + str(1.0) + ';  '
                            + str(1.0) + '];'
                            '\n'
                            ) 
        with open(input_loc, 'w') as file:
            file.writelines(lines)
        return
    

    def gbisrun(self):
        cwd = os.getcwd()
        os.chdir(self.DaN_object.event_object.GBIS_location)
        self.eng.GBISrun(self.DaN_object.event_object.GBIS_insar_template,1,'n','F','1e5','n','n','n',nargout=0)
        self.outputdir= "./" + self.DaN_object.event_object.GBIS_insar_template.split('/')[-1][:-4] + "/invert_1_F"
        self.outputfile = self.outputdir + "/" + "invert_1_F.mat"
        self.eng.generateFinalReport(self.outputfile,1e4,nargout=0)
        os.chdir(cwd)
  
if __name__ == "__main__":
    # # example event ID's us6000jk0t, us6000jqxc, us6000kynh,
    preproc_object = DaN.deformation_and_noise("us6000jk0t",target_down_samp=500,inv_soft='GBIS')
    grond_object = auto_GBIS(preproc_object,'/Users/jcondon/phd/code/auto_inv/GBIS.location')



