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
import mat2kite as m2k
import subprocess as sp 
import os
import h5py
from scipy import io


class auto_grond:
    def __init__(self,deformation_and_noise_object):
        self.DaN_object = deformation_and_noise_object
        self.path_to_data = self.DaN_object.event_object.Grond_insar
        self.matfiles = self.create_kite_scene()
        self.data = self.read_data_params()
        self.edit_input_yml_event_title()
        self.edit_input_yml_invt_params(NP=1)
        self.edit_dec_number(5)
        self.run_grond()
        
    def create_kite_scene(self):
        onlyfiles = [f for f in os.listdir(self.path_to_data) if os.path.isfile(os.path.join(self.path_to_data, f))]
        matfiles = []
        for file in onlyfiles:
            if ".mat" in file:
                full_path = os.path.join(self.path_to_data,file)
                matfiles.append(full_path)
            else:
                pass 
        if len(matfiles) > 1:
            print("Two mat files presenent please remove one")
        else:
            pass 
        m2k.main(matfiles[0],os.path.join(self.path_to_data,matfiles[0])[:-3]+"kiteModified",(100,100))
        return matfiles 
    
    def run_grond(self):
        cwd = os.getcwd() 
        os.chdir(self.DaN_object.event_object.Grond_location)
        sp.call("../grond_script.sh")
        os.chdir(cwd)
        return 
    
    def edit_dec_number(self,dec_fac):
        yml_loc = os.path.join(self.DaN_object.event_object.Grond_config,"insar_rectangular_template.gronf")
        with open(yml_loc,'r') as file:
            lines = file.readlines()
        for ii in range(len(lines)):
            if  '  decimation_factor: 1' in lines[ii]:
                lines[ii] = '  decimation_factor: ' + str(dec_fac) + '\n'
            else:
                pass 
        with open(yml_loc, 'w') as file:
            file.writelines( lines )
        return 
    
    def edit_input_yml_event_title(self):
        yml_loc = os.path.join(self.DaN_object.event_object.Grond_config,"insar_rectangular_template.gronf")
        with open(yml_loc,'r') as file:
            lines = file.readlines()
        for ii in range(len(lines)):
            #event_names:
             #- 'gfz2018pmjk'
            if '#- \'gfz2018pmjk\'' in lines[ii]:
                print("FOOOOOUND")
                lines[ii] = '- ' + '\'' + self.DaN_object.event_object.ID + '\'' + '\n'
            elif  '#event_names:' in lines[ii]:
                print("FFFFFFFFFOUND EVENT NAMES")
                lines[ii] = 'event_names:' + "\n"
            elif 'name_template: rect_2009LaAquila' in lines[ii]:
                lines[ii] = '  name_template: rect_' + self.DaN_object.event_object.ID + '\n'
        with open(yml_loc, 'w') as file:
            file.writelines( lines )
       
    def read_data_params(self):
        try:
            data = h5py.File(self.matfiles[0], 'r')
        except OSError:
            data = io.loadmat(self.matfiles[0])
        return data 


    def calc_square_start(self):
        mu = 3.2e10
        slip_rate=5.5e-5
        L = np.cbrt(float(self.DaN_object.event_object.MTdict['moment'])/(slip_rate*mu))
        return L

    
    def edit_input_yml_invt_params(self,NP=1):
        yml_loc = os.path.join(self.DaN_object.event_object.Grond_config,"insar_rectangular_template.gronf")
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
        sq_dim = self.calc_square_start()
        print("################################################")
        print(str(int(np.max(self.data['lonlat_m'][0]))))
        print(str(int(np.max(self.data['lonlat_m'][1]))))
        print('#####################less mong ####################')
        print(str(int(np.max(self.data['lonlat_m'][:,0])/2)))
        print(str(int(np.max(self.data['lonlat_m'][:,1])/2)))
        print(str(int(np.min(self.data['lonlat_m'][:,0])/2)))
        print(str(int(np.min(self.data['lonlat_m'][:,1])/2)))

        with open(yml_loc,'r') as file:
            yml_lines = file.readlines() 
        for ii in range(len(yml_lines)):
            if 'depth: 2500 .. 7000' in yml_lines[ii]:
                if (depth - 5000) < 0:
                       yml_lines[ii] = '    depth: 0 .. ' + str(int(20000)) + '\n'
                else: 
                    # yml_lines[ii] = '    depth: ' + str(int(depth-5000)) +  ' .. ' + str(int(depth+5000)) +'\n'
                    yml_lines[ii] = '    depth: ' + str(int(1000)) +  ' .. ' + str(int(20000)) +'\n'
            elif 'east_shift: 0 .. 20000' in yml_lines[ii]:
                yml_lines[ii] = '    east_shift: ' + str(int(np.min(self.data['lonlat_m'][:,0])/2)) + ' .. ' + str(int(np.max(self.data['lonlat_m'][:,0])/2)) + '\n'
                #  yml_lines[ii] = '    east_shift: 0 .. ' + str(2000.0) + '\n'
            elif 'north_shift: 0 .. 20000' in yml_lines[ii]:
                yml_lines[ii] = '    north_shift: ' + str(int(np.min(self.data['lonlat_m'][:,0])/2)) + ' .. ' + str(int(np.max(self.data['lonlat_m'][:,1])/2)) + '\n'
                # yml_lines[ii] = '    north_shift: 0 .. ' + str(2000.0) + '\n'
            elif 'length: 5000 .. 10000' in yml_lines[ii]:
                if (sq_dim - 2000) < 0:
                    yml_lines[ii] = '    length: 0 .. ' + str(int(sq_dim+5000)) + '\n'
                else:
                    yml_lines[ii] = '    length: ' + str(int(sq_dim - 2000)) + ' .. ' + str(int(sq_dim + 2000)) + '\n'
            elif 'width: 2000 .. 10000' in yml_lines[ii]:
                if (sq_dim - 2000) < 0:
                    yml_lines[ii] = '    width: 0 .. ' + str(int(sq_dim+5000)) + '\n'
                else:
                    yml_lines[ii] = '    width: ' + str(int(sq_dim - 2000)) + ' .. ' + str(int(sq_dim + 2000)) + '\n'
            elif 'slip: .2 .. 20' in yml_lines[ii]:
                pass 
            elif 'dip: 10 .. 50' in yml_lines[ii]:
                if NP==1:
                    if int(dip1+30) >90:
                        yml_lines[ii] = '    dip: ' + str(int(dip1 - 30)) + ' .. ' + str(int(90)) + '\n'
                    elif int(dip1-30) <-90:
                        yml_lines[ii] = '    dip: ' + str(-90) + ' .. ' + str(int(dip1 + 30)) + '\n'     
                    else:
                        yml_lines[ii] = '    dip: ' + str(int(dip1 - 30)) + ' .. ' + str(int(dip1 + 30)) + '\n'
                elif NP == 2:
                    if int(dip2+30) >90:
                        yml_lines[ii] = '    dip: ' + str(int(dip2 - 30)) + ' .. ' + str(int(90)) + '\n'
                    elif int(dip1-30) <-90:
                        yml_lines[ii] = '    dip: ' + str(-90) + ' .. ' + str(int(dip2 + 30)) + '\n'     
                    else:
                        yml_lines[ii] = '    dip: ' + str(int(dip2 - 30)) + ' .. ' + str(int(dip2 + 30)) + '\n'
                                
            elif 'rake: 120 .. 360' in yml_lines[ii]:
                if NP==1:
                    yml_lines[ii] = '    rake: ' + str(int(rake1 - 30)) + ' .. ' + str(int(rake1 + 30)) + '\n'
                elif NP == 2:
                    yml_lines[ii] = '    rake: ' + str(int(rake2 - 30)) + ' .. ' + str(int(rake2 + 30)) + '\n'
                     
            elif 'strike: 220 .. 360' in yml_lines[ii]:
                if NP==1:
                    yml_lines[ii] = '    strike: ' + str(int(strike1 - 50)) + ' .. ' + str(int(strike1 + 50)) + '\n'
                elif NP == 2:
                    yml_lines[ii] = '    strike: ' + str(int(strike2 - 50)) + ' .. ' + str(int(strike2 + 50)) + '\n'
                 
        with open(yml_loc, 'w') as file:
            file.writelines(yml_lines)
        return
if __name__ == "__main__":
    # # example event ID's us6000jk0t, us6000jqxc, us6000kynh,
    preproc_object = DaN.deformation_and_noise("us6000jk0t",target_down_samp=4000)
    grond_object = auto_grond(preproc_object)

