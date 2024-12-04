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
import GBIS_run as GR 
import shutil 
import gc
import generate_final_report as gfr
import test_set_selection as tss

# @timeout_decorator.timeout(10800) # Times out after 6 hours 
def main(USGS_ID):
    locations_usgs = []
    locations_reloc_NP1 = []
    locations_reloc_NP2 = []  
    opt_models_NP1 = [] 
    opt_models_NP2 = [] 
    # try:
    print('RUNNING AGAIN ')
    # with open(full_test[ii]+'_preproc.pkl', 'wb') as outp:
    preproc_one_attempt = 0
    # while preproc_one_attempt < 5:
    #     try:
    preproc_object = DaN.deformation_and_noise(USGS_ID,
                                                    target_down_samp=1750,
                                                    inv_soft='GBIS',
                                                    look_for_gacos=True,
                                                    # NP=2,
                                                    all_coseis=True,
                                                    single_ifgm=False,
                                                    coherence_mask=0.015,
                                                    min_unw_coverage=0.015, 
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
        #     preproc_one_attempt = 10
        # except:
        #     preproc_one_attempt += 1

    preproc_object_copy = preproc_object

    try:
        GBIS_object = GR.auto_GBIS(preproc_object,'./GBIS',NP=1,number_trials=1e5,
                                    pygmt=False,generateReport=True,location_search=False,limit_trials=False)
    except:
        preproc_object = DaN.deformation_and_noise(USGS_ID,
                                                    target_down_samp=1750,
                                                    inv_soft='GBIS',
                                                    look_for_gacos=True,
                                                    # NP=2,
                                                    all_coseis=True,
                                                    single_ifgm=False,
                                                    coherence_mask=0.015,
                                                    min_unw_coverage=0.015, 
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
        GBIS_object = GR.auto_GBIS(preproc_object,'./GBIS',NP=1,number_trials=1e5,
                                    pygmt=False,generateReport=True,location_search=False,limit_trials=False)


    shutil.move(GBIS_object.outputdir,GBIS_object.outputdir + '_location_run')
    locations_usgs.append([float(preproc_object.event_object.time_pos_depth['Position'][0]),float(preproc_object.event_object.time_pos_depth['Position'][1])])

    print( preproc_object.event_object.time_pos_depth['Position'][0])
    print( preproc_object.event_object.time_pos_depth['Position'][1])
    original_lat = preproc_object.event_object.time_pos_depth['Position'][0]
    original_lon = preproc_object.event_object.time_pos_depth['Position'][1]

    preproc_object.event_object.time_pos_depth['Position'][0] = GBIS_object.GBIS_lat
    preproc_object.event_object.time_pos_depth['Position'][1] = GBIS_object.GBIS_lon
    print( preproc_object.event_object.time_pos_depth['Position'][0])
    print( preproc_object.event_object.time_pos_depth['Position'][1])
    # preproc_object.scale_factor_depth = 0.065
    # preproc_object.scale_factor_mag = 0.065
    if os.path.isdir(preproc_object.event_object.LiCS_locations + '_USGS_location_used'):
        pass
    else:
        try:
            shutil.copytree(preproc_object.event_object.LiCS_locations,preproc_object.event_object.LiCS_locations + '_USGS_location_used')
        except Exception as e:
            print(e)
            pass 
    preproc_object.flush_for_second_run()
    # preproc_object.geoc_path, preproc_object.gacos_path = preproc_object.data_block.pull_frame_coseis()
    # preproc_object.check_data_pull()
    # preproc_object.geoc_path =  preproc_object.check_geoc_has_data(preproc_object.geoc_path)
    attempt = 0
    while attempt < 8: # incase data is pulled incorrectly or copied wrong
        try:
                # preproc_object.geoc_path, preproc_object.gacos_path = preproc_object.data_block.pull_frame_coseis()
                # preproc_object.check_data_pull()
                # preproc_object.geoc_path =  preproc_object.check_geoc_has_data(preproc_object.geoc_path)
                # flush_all_processing
                original_diameter_mask = preproc_object.event_object.diameter_mask_in_m
                preproc_object.run_processing_flow(True,good_location=True)
                preproc_object.geoc_final_path = preproc_object.geoc_ds_path
                preproc_object.move_final_output()
                attempt = 8
        except Exception as e:
                preproc_object.flush_all_processing()
                preproc_object.geoc_path, preproc_object.gacos_path = preproc_object.data_block.pull_frame_coseis()
                preproc_object.check_data_pull()
                preproc_object.geoc_path =  preproc_object.check_geoc_has_data(preproc_object.geoc_path)
                preproc_object.event_object.diameter_mask_in_m = original_diameter_mask 
                attempt+=1
                print(e)
                print('processing flow failed trying again with attempt number ' + str(attempt))
                if attempt > 4:
                    print('Failed 5 times on processing flow check errors')
                    # preproc_object.event_object.time_pos_depth['Position'][0] = original_lat
                    # preproc_object.event_object_time_pos_depth['Position'][1] = original_lon


        
                # pickle.dump(preproc_object,outp)



# if GBIS_run == True: 
# for ii in range(len(full_test[ii])):
#     with open(full_test[ii]+'_preproc.pkl', 'rb') as inp:
    # preproc_object = pickle.load(inp)
    
    try:
        GBIS_object = GR.auto_GBIS(preproc_object,'./GBIS',NP=1,number_trials=1e6,pygmt=True,limit_trials=False)
        locations_reloc_NP1.append([GBIS_object.GBIS_lat,GBIS_object.GBIS_lon]) 
        opt_models_NP1.append(GBIS_object.opt_model)
    except:
        pass 
             
    try:
        print('I am in this try')
        GBIS_object = GR.auto_GBIS(preproc_object,'./GBIS',NP=2,number_trials=1e6,pygmt=True,limit_trials=False)
        locations_reloc_NP2.append([GBIS_object.GBIS_lat,GBIS_object.GBIS_lon]) 
        opt_models_NP2.append(GBIS_object.opt_model)
    except:
        pass
        

    try:
        GBIS_Area_path = GBIS_object.path_to_data 
        GBIS_res_1 = GBIS_Area_path + '/../'+ USGS_ID + '_NP1'
        GBIS_res_2 = GBIS_Area_path + '/../'+ USGS_ID + '_NP2'
        gfr.generateReport(GBIS_Area_path,GBIS_res_1,GBIS_res_2,USGS_ID)
    except:
        pass

    # shutil.rmtree(preproc_object.event_object.LiCS_locations)
    # shutil.rmtree(preproc_object.event_object.LiCS_locations + '_USGS_location_used')

    # except Exception as e:
    #     print(e)
    #     print(crashed_events)
 

if __name__ == '__main__':
    # array = ['us6000a8nh']
    df = pd.read_csv('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/gCent_Catalog_no_header_info.csv')
    df_land = pd.read_csv('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test_set_on_land.csv')

    df_china = df[df['Location'] == 'China']
    df_Iran = df[df['Location'] == 'Iran']
    df_turkey = df[df['Location'] == 'Turkey']
    df_pakist = df[df['Location'] == 'Pakistan']

    # rest_of_data = df[df['Location'] !='Pakistan' and df['Location'] != 'Iran' and df['Location']!='Turkey' and df['Location']!= 'China']
    
    # full_test = array + list(df_china.ComCatID)+ list(df_Iran.ComCatID) + list(df_turkey.ComCatID) 
    
    # full_test = ['us7000g9zq']
    # full_test = ['us6000a8nh']
    # full_test = ['us70008cld']
    # full_test = ['us600068w0']
    
    # full_test = ['us70006sj8']

    # full_test = [
    #             # 'us7000gebb',
    #             'us6000jk0t',
    #             # 'us6000kynh',
    #             # 'us7000htbb',
    #             'us7000abmk',
    #             # 'us7000cet5',
    #             'us7000df40',
    #             # 'us7000fu12',
    #             'us60007anp',
    #             'us70006sj8', 
    #             'us7000g9zq',  
    #             'us70008cld',
    #             'us600068w0',
    #             'us7000m3ur',
    #             'us6000a8nh',
    #             'us7000mpe2',
    #             'us7000lt1i',
    #             'us7000mbuv',
    #             'us6000dyuk',
    #             'us6000mjpj',
    #             'us6000ldpm',
    #             'us6000lfn5',
    #             'us6000len8',  
    #             'us6000abnv',  
    #             'us6000iaqi',
    #             'us7000ljvg',
    #             'us7000gx8m',  
    #             'us6000529r',
    #             'us60007d2r',
    #             'us70008db1',
    #             'us6000e2k3',
    #             'us60007d2r', 
    #             'us6000b26j',
    #             'us6000bdq8',
    #             'us6000ddge',
    #             'us6000dxge',     
    # ]

    # full_test = [ 
               
    #               'us60007anp',
    #               'us6000iaqi',
    #              'us60007d2r', 
    #              'us6000529r',
    #               'us6000ldpm',
    #               'us7000g9zq', 
    #              'us6000lfn5',
    #               'us6000len8',
    #               ] 
    # failed_tests = [] 
    # full_test = full_test + list(df_china.ComCatID)+ list(df_Iran.ComCatID) + list(df_turkey.ComCatID) +list(df_pakist.ComCatID) + list(df_land.id)
    # full_test = []
    # full_test = ['us70008cld']
    
    # full_test =   ['us7000gebb']
 
    # full_test = ['us7000mpe2','us7000lt1i', 'us6000b26j',]
    # full_test = ['us6000jk0t']
    # full_test = ['us7000gebb']
    # full_test = ['us7000htbb']
    # full_test = ['us6000a8nh']
    # full_test = ['us6000dxge']
    # full_test = ['us6000iaqi']
    # full_test = ['us6000dxge']
    # full_test = ['us6000kynh']
    filtered_df =  tss.LiCS_test_set_selection('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/LiCS_EQ_Responder.csv','2019-01-01','2024-08-30',5.0,6.5)
    full_test = list(filtered_df.ID) #+ full_test
    missing_list = ['us6000b26j',
                    'us6000dxge',
                    'us7000j0ne',
                    'us6000jmhm',
                    'us7000ki5u',
                    'us6000kx3e',
                    'us7000kufc',
                    'us7000mbuv',
                    'us7000mpe2',
                    'us6000mjpj']
    full_test = missing_list + full_test
    # full_test = ['us6000jk0t']
    # full_test = ['us6000jk0t']
    # print(full_test)
 
    done_list  = [
    'us7000df40',
    'us7000dimf',
    'us6000e2k3',
    'us6000h4z7',
    'us6000hz8x',
    'us6000j5sn',
    'us6000jbsk',
    'us7000biqb',
    # 'us6000jk0t',
    'us6000kynh',
    'us6000ldpg',
    'us7000fu12',
    'us7000gebb',
    'us7000gx8m',
    'us7000hj3u',
    'us7000k5zj',
    'us7000ljvg',
    'us60007anp',
    'us60007d2r',
    'us6000frf2',
    'us6000df9r',
    'us6000g7wd',
    'us6000hfqj',
    'us6000i1yw',    
    'us6000j7uj',
    'us6000ky5l',
    'us6000ldpv',
    'us6000len8',
    'us7000hszl',
    'us600068w0',
    'us60006rpn',
    'us70006d0m',
    'us700069f4'
]   

    # directories = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]
    
    # for dir in directories:
    #     if 'insar_processing':
    #         id_done  = dir.split('_')[0]
    #         done_list.append(id_done)
   
    # full_test = ['us6000jk0t']
    
    directories = os.listdir('.')
    for dir in directories:
        if 'insar_processing' in dir:
            id_done  = dir.split('_')[0]
            done_list.append(id_done)

    for i in range(len(done_list)):
        print(done_list[i])
        if done_list[i] in full_test:
            full_test.remove(done_list[i])

    full_test  = ['us70007v9g'] + full_test
    # print('DDDDDDDOOOONE LIST')
    # print(len(done_list))
    # print('FULL_TEST')
    # print(len(full_test))
    # full_test = ['us70006a4q']
    # full_test = ['us700069f4']

    for ii in range(len(full_test)):
        try:
            main(full_test[ii])
            gc.collect()
        except Exception as e: 
            print(e)
            pass
         
            # print(full_test[ii])
            # failed_tests.append(full_test[ii])
