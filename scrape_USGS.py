# from bs4 import BeautifulSoup 
# import requests 
# import chromedriver_binary
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import obspy
from obspy.clients.fdsn import Client 
from obspy import UTCDateTime 

# import bs4 as bs
# import urllib.request
import time 
import subprocess as sp
import os 
import shutil

class USGS_event:
    """
    Class for handling of all prior event information from USGS 
    """
    def __init__(self,ID):
        self.ID = ID 
        self.event_page = self.define_event_page(ID)
        self.time_pos_depth, self.strike_dip_rake, self.MTdict = self.pull_USGS_info(self.ID)
        # self.time_pos_depth, self.strike_dip_rake, self.MTdict = self.scrape(self.event_page)
        self.ifgms_csv = './event_ifgms_'+ self.ID +'.csv'
        self.run_event_ifgm_RScrape()
        self.create_folder_stuct()
        self.create_event_file()

    def define_event_page(self,ID):
        """
        Create link to USGS page from event ID 
        """
        USGS_link="https://earthquake.usgs.gov/earthquakes/eventpage/" + ID +"/moment-tensor"
        print(USGS_link)
        return USGS_link 
    def scrape(self,USGS_link):
        """
        Scrape source parameter information from USGS: returns dictionaries of:
        Time, position and depth (time_pos_depth)
        Strike, Dip and Rake (strike_dip_rake)
        moment tensor information (MTDict)
        """
        url = USGS_link
        # configure webdriver
        options = Options()
        options.add_argument("--window-size=1920,1080")  # set window size to native GUI size
        options.add_argument("start-maximized")  # ensure window is full-screen
        options.add_argument("--headless") # hide GUI

        # configure chrome browser to not load images and javascript
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option(
            # this will disable image loading
            "prefs", {"profile.managed_default_content_settings.images": 2}
        )
        service = Service()
        driver = webdriver.Chrome(service=service,options=options)
        #                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        driver.get(USGS_link)
        time.sleep(2) # This sleep is key to allow for the javascript on the USGS to run before pulling 
    # Finds position of time position and depth from static html section
        time_pos_depth = {"DateTime":None,
                        "Position":None,
                        "Depth":None}
        
        for ii in range(len(driver.page_source.split('event-page-header')[1].split('>'))):
            # print((driver.page_source.split('event-page-header')[1].split('>'))[ii])
            if '(UTC)' in (driver.page_source.split('event-page-header')[1].split('>'))[ii]:
                time_pos_depth['DateTime'] = driver.page_source.split('event-page-header')[1].split('>')[ii].replace('(UTC)</li','')
            if 'S' in (driver.page_source.split('event-page-header')[1].split('>'))[ii] or 'N' in (driver.page_source.split('event-page-header')[1].split('>'))[ii] or 'E' in (driver.page_source.split('event-page-header')[1].split('>'))[ii] or 'W' in (driver.page_source.split('event-page-header')[1].split('>'))[ii]:
                array = driver.page_source.split('event-page-header')[1].split('>')[ii].split(' ')
                if 'S' in array[0]:
                    array[0] = str(float(array[0].replace('S','').replace('°','')) * -1) 
                else:
                    array[0] = array[0].replace('N','').replace('°','') 

                if 'W'in array[1]:
                    array[1] = str(float(array[1].replace('W</li','').replace('°','')) * -1)
                else:
                    array[1] = array[1].replace('E</li','').replace('°','')

                time_pos_depth['Position'] = array

                # time_pos_depth['Position'] = [driver.page_source.split('event-page-header')[1].split('>')[ii].replace('N','').replace('E</li','').replace('°','').replace('S','').replace('W</li','')
            if 'depth' in (driver.page_source.split('event-page-header')[1].split('>'))[ii].lower():
                time_pos_depth["Depth"] = float(driver.page_source.split('event-page-header')[1].split('>')[ii].replace('depth</li','').replace('km',''))

        # print("CHEEEEEEECKEREERS BABBBBY")
        print(time_pos_depth)
        # Find position of strike dip and rake from javascript tabel in USGS page 
        strike_dip_rake = {"strike":0,
                        "dip":0, 
                        "rake":0}
        strike = []
        dip = []
        rake =[]
        for ii in range(len(driver.page_source.split('<shared-nodal-planes')[1].split("="))):
            # print(driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii])
            if "strike" in driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii].lower():
                strike.append(driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii].split('>')[1].replace('</mat-cell','').replace('°','').replace('</td',''))
            if "rake" in driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii].lower():
                rake.append(driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii].split('>')[1].replace('</mat-cell','').replace('°','').replace('</td',''))
            if "dip" in driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii].lower():
                dip.append(driver.page_source.split('<shared-nodal-planes')[1].split("=")[ii].split('>')[1].replace('</mat-cell','').replace('°','').replace('</td',''))
        
       
        strike_dip_rake["strike"] =strike[1:3]
        strike_dip_rake["rake"]=rake[1:3]
        strike_dip_rake["dip"]=dip[1:3]

       
        # Pull and strip text from Moment tensor tabel 
        MTdict = {'moment':None,
                'magnitude':None,
                'Depth_MT':None,
                'PercentDC':None,
                'Half Duration':None
                } 
        for ii in range(len(driver.page_source.split('<moment-tensor-info')[1].split("="))):

            # print(driver.page_source.split('<moment-tensor-info')[1].split("=")[ii])
            if 'N-m</dd><dt _ngcontent' in driver.page_source.split('<moment-tensor-info')[1].split("=")[ii]:
                # print(driver.page_source.split('<moment-tensor-info')[1].split("=")[ii])
                MTdict['moment'] = driver.page_source.split('<moment-tensor-info')[1].split("=")[ii].split('<')[0].replace('"">','').replace('N-m','')
            if 'Mww <!----><!----></dd><dt _ngcontent' in driver.page_source.split('<moment-tensor-info')[1].split("=")[ii]:
                print(driver.page_source.split('<moment-tensor-info')[1].split("=")[ii])
                MTdict['magnitude'] = driver.page_source.split('<moment-tensor-info')[1].split("=")[ii].split('<')[0].replace('"">','').replace('Mww','')
            if 'km' in driver.page_source.split('<moment-tensor-info')[1].split("=")[ii]:
                print(driver.page_source.split('<moment-tensor-info')[1].split("=")[ii])
                MTdict['Depth_MT'] = driver.page_source.split('<moment-tensor-info')[1].split("=")[ii].split('<')[0].replace('"">','').replace('km','')
            if '%' in driver.page_source.split('<moment-tensor-info')[1].split("=")[ii]:
                print(driver.page_source.split('<moment-tensor-info')[1].split("=")[ii])
                MTdict['PercentDC'] = driver.page_source.split('<moment-tensor-info')[1].split("=")[ii].split('<')[0].replace('"">','').replace('%','')
            if 'Half Duration' in driver.page_source.split('<moment-tensor-info')[1].split("=")[ii]:
                print(driver.page_source.split('<moment-tensor-info')[1].split("=")[ii])
                MTdict['Half Duration'] = driver.page_source.split('<moment-tensor-info')[1].split("=")[ii+1].split('<')[0].replace('"">','').replace('s','')
        
        print(time_pos_depth)
        print(time_pos_depth['Depth'])
        print(strike_dip_rake)
        print(MTdict)
        return time_pos_depth, strike_dip_rake, MTdict
    
    def pull_USGS_info(self,ID,client='USGS'):
        MTdict = {'moment':None,
                    'magnitude':None ,
                    'Depth_MT':None,
                    'PercentDC':None,
                    'Half Duration':None,
                    'centroid_lat':None,
                    'centroid_lon':None
                    }
        strike_dip_rake = {"strike":0,
                            "dip":0, 
                            "rake":0}
        time_pos_depth = {"DateTime":None,
                            "Position":None,
                            "Depth":None}
        client = Client(client)
        event = client.get_events(eventid=ID)[0]
        for x in event.magnitudes:
            if x['magnitude_type'] == 'Mww':
                MTdict['magnitude'] =  x.mag
        # for x in event.focal_mechanisms[0]:
        x = event.focal_mechanisms[0]
        for x in event.focal_mechanisms:
            print(x.resource_id)
            if 'mww' in str(x.resource_id):
                strike_dip_rake['strike'] = [x.nodal_planes.nodal_plane_1.strike,x.nodal_planes.nodal_plane_2.strike]
                strike_dip_rake['dip'] = [x.nodal_planes.nodal_plane_1.dip,x.nodal_planes.nodal_plane_2.dip] 
                strike_dip_rake['rake'] = [x.nodal_planes.nodal_plane_1.rake,x.nodal_planes.nodal_plane_2.rake] 
                MTdict['PercentDC'] = x.moment_tensor.double_couple
                MTdict['moment'] = x.moment_tensor.scalar_moment
                MTdict['type'] = 'mww'
                break 
            elif 'mwb' in str(x.resource_id):
                strike_dip_rake['strike'] = [x.nodal_planes.nodal_plane_1.strike,x.nodal_planes.nodal_plane_2.strike]
                strike_dip_rake['dip'] = [x.nodal_planes.nodal_plane_1.dip,x.nodal_planes.nodal_plane_2.dip] 
                strike_dip_rake['rake'] = [x.nodal_planes.nodal_plane_1.rake,x.nodal_planes.nodal_plane_2.rake] 
                MTdict['PercentDC'] = x.moment_tensor.double_couple
                MTdict['moment'] = x.moment_tensor.scalar_moment
                MTdict['type'] = 'mwb'
                break
        for x in event.origins:
            if 'mww' in str(x.resource_id) and x.depth_type == 'from moment tensor inversion' and MTdict['type'] == 'mww':
                MTdict['centroid_lat'] = x.latitude
                MTdict['centroid_lon'] = x.longitude
                MTdict['Depth_MT'] = x.depth/1000
                break
            elif 'mwb' in str(x.resource_id) and x.depth_type =='from moment tensor inversion' and MTdict['type'] == 'mwb':
                MTdict['centroid_lat'] = x.latitude
                MTdict['centroid_lon'] = x.longitude
                MTdict['Depth_MT'] = x.depth/1000
                break
        time_pos_depth['Depth'] = event.origins[0].depth/1000
        time_pos_depth['DateTime'] = str(UTCDateTime(event.origins[0].time)).replace('T',' ').replace('Z','')
        time_pos_depth['Position'] = [event.origins[0].latitude,event.origins[0].longitude]
        return  time_pos_depth, strike_dip_rake, MTdict
    def run_event_ifgm_RScrape(self):
        """
        Creates event csv for USGS event from LiCSEarthquake catalog
        """
        args=["Rscript", "./scott_scripts/event_ifgms.R", self.ID ,self.ifgms_csv]
        sp.run(args,shell=False)
        return 
    
    def convert_sdr_to_gbis(self):
        """
        Converts strike dip and rake to GBIS format (Needs Triple Checking)
        """
        for ii in range(len(self.strike_dip_rake['strike'])):
            if float(self.strike_dip_rake['strike'][ii])-180<0:
                self.strike_dip_rake['strike'][ii] = float(self.strike_dip_rake['strike'])+180 
            else:
                self.strike_dip_rake['strike'][ii] = float(self.strike_dip_rake['strike'][ii]) - 180
            self.strike_dip_rake['dip'][ii] = -float(self.strike_dip_rake['dip'][ii])  
        return 

    def scrape_update(self):
        """
        Rerun the USGS scrape allows for updates. 
        """
        self.time_pos_depth, self.strike_dip_rake, self.MTdict = self.scrape(self.event_page)
        return 
    

    def create_event_file(self):
        self.event_file_path = os.path.join(self.specific_event,'event.txt')
        if os.path.isfile(self.event_file_path):
            os.remove(self.event_file_path)
        else:
            pass 
        with open(self.event_file_path,'w') as f:
            f.write("name = " + self.ID +'\n')
            f.write("time = " + self.time_pos_depth['DateTime'] +"\n")
            f.write("latitude = " + str(self.time_pos_depth['Position'][0]) + '\n')
            f.write("longitude = " + str(self.time_pos_depth['Position'][1]) + '\n')
            f.write("magnitude = " + str(self.MTdict['magnitude']) + '\n')
            f.write("moment = " + str(self.MTdict['moment']) + '\n')
            f.write("depth = " + str(float(self.time_pos_depth['Depth'])*1000) + "\n") 
            # f.write("region = " + "\n")
            f.write("catalog = USGS \n")
            # f.write("mnn =  \n")
            # f.write("mee = \n")
            # f.write("mdd = \n")
            # f.write("mne = \n")
            # f.write("mnd = \n")
            # f.write("med = \n")
            f.write("strike1 = " + str(self.strike_dip_rake['strike'][0]) + '\n')
            f.write("dip1 = " + str(self.strike_dip_rake['dip'][0]) + '\n')
            f.write("rake1 = " + str(self.strike_dip_rake['rake'][0]) + '\n')
            f.write("strike2 = " + str(self.strike_dip_rake['strike'][1]) + '\n')
            f.write("dip2 = " + str(self.strike_dip_rake['dip'][1]) + '\n')
            f.write("rake2 = " + str(self.strike_dip_rake['rake'][1]) + '\n')
            # f.write("duration = ")
        shutil.copy(self.event_file_path, os.path.join(self.LiCS_locations,self.ID+'.txt'))
        return 
   
    def create_folder_stuct(self):
        cwd = os.getcwd()
        self.Grond_location = os.path.join(cwd,self.ID +"_grond_area")
        self.Grond_config = os.path.join(self.Grond_location,"config")
        self.Grond_data = os.path.join(self.Grond_location,"data")
    
        self.LiCS_locations = os.path.join(cwd,self.ID +'_insar_processing')
        self.grond_insar_template = os.path.join(self.Grond_config,'insar_rectangular_template.gronf')
        self.grond_event = os.path.join(self.Grond_data,'events')
        self.specific_event = os.path.join(self.grond_event,self.ID)
        self.Grond_insar = os.path.join(self.specific_event,'insar')
        self.gf_stores = os.path.join(self.Grond_location,'gf_stores')
        self.crust_location = os.path.join(self.gf_stores,'crust2_ib_static')

        #GBIS 
        self.GBIS_location = os.path.join(cwd,self.ID +"_GBIS_area")
        self.GBIS_insar_template = os.path.join(self.GBIS_location,self.ID + '.inp')


       
        if os.path.isdir(self.Grond_location):
            pass
        else:
            os.mkdir(self.Grond_location)
        if os.path.isdir(self.Grond_data):
            pass 
        else:
            os.mkdir(self.Grond_data)
        if os.path.isdir(self.Grond_config):
            pass 
        else: 
            os.mkdir(self.Grond_config)
        if os.path.isdir(self.LiCS_locations):
            pass 
        else:
            os.mkdir(self.LiCS_locations)
        if os.path.isdir(self.grond_event):
            pass 
        else:
            os.mkdir(self.grond_event)
        if os.path.isdir(self.specific_event):
            pass 
        else: 
            os.mkdir(self.specific_event)
        if os.path.isdir(self.Grond_insar):
            pass 
        else:
            os.mkdir(self.Grond_insar)
        if os.path.isfile(self.grond_insar_template):
            pass 
        else:
            shutil.copy("insar_rectangular_template.gronf",self.grond_insar_template)

        if os.path.isdir(self.gf_stores):
            pass
        else:
            cwd = os.getcwd() 
            os.chdir(self.Grond_location)
            sp.call("../download_gf_stores.sh")
            os.chdir(cwd)
        if os.path.isdir(self.crust_location):
            pass 
        else:
            cwd = os.getcwd() 
            os.chdir(self.gf_stores)
            sp.call("../../download_gf_stores.sh")
            os.chdir(cwd)

        if os.path.isdir(self.GBIS_location):
            pass 
        else:
            os.mkdir(self.GBIS_location)

        if os.path.isfile(self.GBIS_insar_template):
            pass 
        else: 
            shutil.copy("example_GBIS_input.inp",self.GBIS_insar_template)
        





        return 


if __name__=="__main__":
    test_event = USGS_event("us6000jk0t")
    print(test_event.time_pos_depth)
    print(test_event.strike_dip_rake)
    print(test_event.MTdict)
    print("############################## rescrapping ##################")
    test_event.scrape_update()
    print(test_event.time_pos_depth)
    print(test_event.strike_dip_rake)
    print(test_event.MTdict)
    print("############################## Complete ##################")
    print("############################## Converting to GBIS format ##################")
    test_event.convert_sdr_to_gbis()
    print(test_event.strike_dip_rake)
    print("############################## Test Complete ##################")

    


