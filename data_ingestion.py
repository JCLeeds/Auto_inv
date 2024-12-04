import pandas as pd 
import LiCSBAS01_get_geotiff_new as LiC_get
import LiCSBAS02_ml_prep as LiCS_ml 
import numpy as np 
import scrape_USGS as sUSGS
import os 
import shutil
import timeout_decorator

class DataBlock:
    """
    Class for handling of all InSAR data pulling associated with the event formed in the event stage. 
    """
    def __init__(self,USGS_Event):
        self.USGS_event = USGS_Event
        self.eq_df = None
        self.create_df()
        self.add_flags_to_df()
        self.geoc_path = []
        self.gacos_path = [] 
        # self.geocml_path = []
        # self.gacosml_path = []
       
    def create_df(self):
        """
        Creates data frame from event csv generated in Event class. 
        """
        self.eq_df = pd.read_csv(self.USGS_event.ifgms_csv)
        print(self.eq_df)
        return 
    def add_flags_to_df(self):
        """
        Assigns co-seismic, post-seismic and pre-seismic flags to event data frame. 
        """
        print(self.USGS_event.time_pos_depth['DateTime'])
        dates = self.eq_df['dates']
        self.eq_df[['start_aqu','end_aqu']] = self.eq_df['dates'].apply(lambda x: pd.Series(str(x).split('_')))
        self.eq_df['start_aqu_dt'] = pd.to_datetime(self.eq_df['start_aqu'],format='%Y%m%d')
        self.eq_df['end_aqu_dt'] = pd.to_datetime(self.eq_df['end_aqu'],format='%Y%m%d')
        self.eq_df['start_aqu'] = pd.to_numeric(self.eq_df['start_aqu'])
        self.eq_df['end_aqu'] = pd.to_numeric(self.eq_df['end_aqu']) 
        self.eq_df['Flag'] = np.where((self.eq_df['start_aqu_dt'] <= self.USGS_event.time_pos_depth['DateTime'].split(' ')[0]) 
                                      & (self.eq_df['end_aqu_dt'] >=self.USGS_event.time_pos_depth['DateTime'].split(' ')[0]),
                                      'coseismic','other')
        self.eq_df.loc[self.eq_df['start_aqu_dt']>self.USGS_event.time_pos_depth['DateTime'],'Flag'] = 'post_seismic'
        self.eq_df.loc[self.eq_df['Flag'] == 'other','Flag'] = 'pre_seismic'
        print(self.eq_df)
        return 
    @timeout_decorator.timeout(3600)
    def LiCS_data_pull_single(self,index,single_date=True):
        """
        Wrapped LiCS step one, takes a single df entry and gives the interfergram assocated with those two dates and nothing else. 
        """
        #LiCSBAS01_get_geotiff.py [-f frameID] [-s yyyymmdd] [-e yyyymmdd] [--get_gacos] [--n_para int]
        cwd = os.getcwd()
        LiC_get.main(auto=[self.eq_df['frame'][index],self.eq_df['start_aqu'][index],self.eq_df['end_aqu'][index]],single_date=single_date)
        os.chdir(cwd)
        return 
    @timeout_decorator.timeout(3600)
    def LiCS_data_pull_date_range(self,frame,start,end,single_date=False):
        """
        Wrapped LiCS step one which takes frame, startdate, enddata and single_date flag 
        single date determines if all combinations are pulled or just a single ifgm between the dates  
        
        """
        #LiCSBAS01_get_geotiff.py [-f frameID] [-s yyyymmdd] [-e yyyymmdd] [--get_gacos] [--n_para int]
        cwd = os.getcwd()
        LiC_get.main(auto=[str(frame),start,end],single_date=single_date)
        os.chdir(cwd)
        return 
    
    # def pull_frames_dates_frames_shortest_ifgm(self):
    #     frame_list = self.get_frame_list()
    #     date_range_for_frames
    @timeout_decorator.timeout(3600)
    def pull_frame_coseis(self):
        """
        Filters dataframe to be only coseismic dates and then calls LiCS_data_pull_single to pull each date listen for each frame 
        saves in directory based on frame ID 
        """
        # print(self.eq_df)
        self.reset_df()
        frame_list = self.get_frame_list()
        geoc_path = []
        gacos_path = []
        print(frame_list)
        for ii in range(len(frame_list)):
            self.group_by_flag('coseismic')
            self.group_by_frame(frame_list[ii])
            if self.eq_df.empty:
                print(frame_list[ii])
                print('EARTHQUAKE FRAME NO COSEISMIC')
                self.reset_df() 
                continue 
            else:
                pass 
            print(self.eq_df)
            data_exist, path_GEOC, path_GACOS = self.data_checker(frame_list[ii])

            if data_exist is False:
                for jj in range(len(self.eq_df)):
                    print(self.eq_df['frame'][jj])
                    self.LiCS_data_pull_single(jj)
                path_GEOC, path_GACOS = self.rename_LiCS_dir(frame_list[ii])
            else:
                print('Data for ' + frame_list[ii] + ' Already downloaded')

            geoc_path.append(path_GEOC)
            gacos_path.append(path_GACOS)
            self.reset_df()
        return geoc_path, gacos_path
    @timeout_decorator.timeout(3600)
    def pull_data_frame_dates(self,startdate,enddate,frame=None,single_ifgm=False):
        """
        Given a startdate and end data pull all data from frames associated with the Event,
        if a frame is provided only pull data for that frame.
        """
        if frame is None:
            geoc_path = []
            gacos_path = []
            frame_list = self.get_frame_list()
            for ii in range(len(frame_list)):
                    data_exist, path_GEOC, path_GACOS = self.data_checker(frame_list[ii])
                    if data_exist is False:
                        self.LiCS_data_pull_date_range(frame_list[ii],startdate,enddate,single_date=single_ifgm)
                        # path_GEOC, path_GACOS = self.rename_LiCS_dir(frame_list[ii]+'_'+str(startdate)+"_"+str(enddate))

                        path_GEOC, path_GACOS = self.rename_LiCS_dir(frame_list[ii])
                        geoc_path.append(path_GEOC)
                        gacos_path.append(path_GACOS)
                    else:
                        geoc_path.append(path_GEOC)
                        gacos_path.append(path_GACOS)
            self.reset_df()
            return geoc_path, gacos_path
        
        else:
            if isinstance(frame,list):
                geoc_path = [] 
                gacos_path = []
                for ii in range(len(frame)):
                    data_exist, single_geoc, single_gacos = self.data_checker(frame[ii])
                    if data_exist is False:
                        self.LiCS_data_pull_date_range(frame[ii],startdate,enddate,single_date=single_ifgm)
                        single_geoc, single_gacos = self.rename_LiCS_dir(frame[ii])
                        geoc_path.append(single_geoc)
                        gacos_path.append(single_gacos)
                    else:
                        geoc_path.append(single_geoc)
                        gacos_path.append(single_gacos)
                    
                # return geoc_path, gacos_path

            else:
                data_exist, geoc_path, gacos_path = self.data_checker(frame)
                if data_exist is False:
                    self.LiCS_data_pull_date_range(frame,startdate,enddate,single_date=single_ifgm)
                    geoc_path, gacos_path = self.rename_LiCS_dir(frame)
                else:
                    geoc_path.append(geoc_path)
                    gacos_path.append(gacos_path)
                    
            self.reset_df()
            return geoc_path, gacos_path
        
    def rename_LiCS_dir(self,identifier):
        """
        Renames output from LiCS step one to add an identifier 
        """
        cwd = os.getcwd()
        # if os.path.isdir(os.path.join(cwd,self.USGS_event.ID +'_insar_processing')):
        print("######################## I HAVE ARRIVED ##################################")
        os.rename(os.path.join(cwd,"GEOC"),os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GEOC_"+identifier))
        if os.path.isdir(os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GACOS_"+identifier)):
            shutil.rmtree(os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GACOS_"+identifier))
        os.rename(os.path.join(cwd,"GACOS"),os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GACOS_"+identifier))
        return os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GEOC_"+identifier), os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GACOS_"+identifier)
        # else:
        #     os.rename(os.path.join(cwd,"GEOC"),os.path.join(cwd,"GEOC_"+identifier))
        #     os.rename(os.path.join(cwd,"GACOS"),os.path.join(cwd,"GACOS_"+identifier))
        #     return os.path.join(cwd,"GEOC_"+identifier), os.path.join(cwd,"GACOS_"+identifier)
    
    def data_checker(self,identifier):
        cwd = os.getcwd()
        if os.path.isdir(os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GEOC_"+identifier)):
            data_exist = True 
            geoc_path, gacos_path = os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GEOC_"+identifier), os.path.join(os.path.join(cwd,self.USGS_event.ID +'_insar_processing'),"GACOS_"+identifier)
        else:
            data_exist = False 
            geoc_path = None 
            gacos_path = None
        return data_exist, geoc_path, gacos_path


                    
    def group_by_frame(self,Frame):
        """
        group event dataframe by specific frame 
        """
        self.eq_df = self.eq_df.loc[self.eq_df['frame'] == Frame].reset_index()
        return 
    
    def group_by_flag(self,Flag):
        """
        group event dataframe by specific Flag 
        """
        self.eq_df = self.eq_df.loc[self.eq_df['Flag'] == Flag].reset_index()
        return 
    
    def get_frame_list(self):
        """
        return list of all frame in event dataframe 
        """
        frame_list = self.eq_df.frame.unique().tolist()
        return frame_list

    def reset_df(self):
        """
        refresh event dataframe back to original form from csv. 
        """
        self.create_df()
        self.add_flags_to_df()
        return 
    
    def get_path_names(self,path):
        """
        Given a path or a list of paths store path of all directorys and files in the subdirectory. 
        """
        if isinstance(path,list):
            total_dir_list = []
            total_meta_file_paths = []
            for ii in range(len(path)):
                dir_list = os.listdir(path[ii])
                meta_file_paths = []
                dir_with_ifgm_data = [] 
                for jj in range(len(dir_list)):
                    if os.path.isfile(os.path.join(path[ii],dir_list[jj])):
                        meta_file_paths.append(os.path.join(path[ii],dir_list[jj]))
                    elif os.path.isdir(os.path.join(path[ii],dir_list[jj])):
                        dir_with_ifgm_data.append(os.path.join(path[ii],dir_list[jj]))
                    else:
                        pass
                total_dir_list.append(dir_with_ifgm_data)
                total_meta_file_paths.append(meta_file_paths)
            return total_dir_list, total_meta_file_paths       
        else:
            dir_list = os.listdir(path)
            meta_file_paths = []
            dir_with_ifgm_data = [] 
            for ii in range(len(dir_list)):
                if os.path.isfile(os.path.join(path,dir_list[ii])):
                    meta_file_paths.append(os.path.join(path,dir_list[ii]))
                elif os.path.isdir(os.path.join(path,dir_list[ii])):
                    dir_with_ifgm_data.append(os.path.join(path,dir_list[ii]))
                else:
                    pass 
            return dir_with_ifgm_data, meta_file_paths

    
    # def create_geoc_ml(self,input_geoc_gacos):
    #     """
    #     Converting TIFS to Float32 using LiCS step two
    #     """
    #     if isinstance(input_geoc_gacos,list):
    #         geoc_ml_path = []
    #         for ii in range(len(input_geoc_gacos)):
    #             output_geoc_gacos = str(input_geoc_gacos[ii]+"_floatml")
    #             geoc_ml_path.append(output_geoc_gacos)
    #             if os.path.isdir(output_geoc_gacos): 
    #                 print(output_geoc_gacos + '  path in use using data already populated')
    #             else:
    #                 LiCS_ml.main(auto=[input_geoc_gacos[ii],output_geoc_gacos])

    #     else:
    #         output_geoc_gacos = str(input_geoc_gacos+"_floatml")
    #         if os.path.isdir(output_geoc_gacos): 
    #             print(output_geoc_gacos + '  path in use using data already populated')
    #         else:
    #             LiCS_ml.main(auto=[input_geoc_gacos,output_geoc_gacos])
    #         geoc_ml_path = output_geoc_gacos

    #     return geoc_ml_path


if __name__ == '__main__':
    test_event = sUSGS.USGS_event("us6000jk0t")
    test_block = DataBlock(test_event)
    print("####coseismic######")
    print(test_block.eq_df.loc[test_block.eq_df['Flag'] =='coseismic'])
    print("####post_seismic#####")
    print(test_block.eq_df.loc[test_block.eq_df['Flag'] =='post_seismic'])
    print("Pulling Data")
    geoc_path,gacos_path = test_block.pull_frame_coseis()
    print("####Path to GEOC####")
    print(geoc_path)
    geoc_ml_path = test_block.create_geoc_ml(geoc_path)
    print("####files in GEOC #####")
    dirs_with_ifgms, meta_file_paths = test_block.get_path_names(geoc_ml_path)  
    print(dirs_with_ifgms)
    print(meta_file_paths)
    # co_index = test_block.eq_df.index[test_block.eq_df['Flag'] == 'coseismic'].tolist()

    