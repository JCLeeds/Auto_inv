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
import pylab as plt
# import bs4 as bs
# import urllib.request
import time 
import subprocess as sp
import os 
import shutil
import coseis_lib as cslib
import numpy as np
import timeout_decorator

class USGS_event:
    """
    Class for handling of all prior event information from USGS 
    """
    def __init__(self,ID):
        self.ID = ID 
      
        self.event_page = self.define_event_page(ID)
        # if self.ID.startswith('nn'):
        #         self.time_pos_depth, self.strike_dip_rake, self.MTdict = self.scrape(self.event_page)
        # else:
        self.time_pos_depth, self.strike_dip_rake, self.MTdict = self.pull_USGS_info(self.ID)
        # self.time_pos_depth, self.strike_dip_rake, self.MTdict = self.scrape(self.event_page)
        self.ifgms_csv = './event_ifgms_'+ self.ID +'.csv'
        try:
            self.run_event_ifgm_RScrape()
        except Exception as e:
            print(e)

        self.create_folder_stuct()
        self.create_event_file()
        self.create_beachball()
        self.diameter_mask_in_m = self.diameter_mask()

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
                    'centroid_lon':None,
                    'magnitude_type': None
                    }
        strike_dip_rake = {"strike":0,
                            "dip":0, 
                            "rake":0}
        time_pos_depth = {"DateTime":None,
                            "Position":None,
                            "Depth":None}
        client = Client(client)
        event = client.get_events(eventid=ID)[0]
        # print(event)
        for x in event.magnitudes:
            if x['magnitude_type'] == 'Mww':
                MTdict['magnitude'] =  x.mag 
                MTdict['magnitude_type'] = x['magnitude_type']
                break 
            elif x['magnitude_type'] == 'Mwb':
                MTdict['magnitude'] =  x.mag 
                MTdict['magnitude_type'] = x['magnitude_type']
                break
            elif  x['magnitude_type'] == 'Mwr':
                MTdict['magnitude'] =  x.mag 
                MTdict['magnitude_type'] = x['magnitude_type']
                break
         
            # print(event.magnitudes)
        if MTdict['magnitude'] is None: # Ask tim about mwb
            MTdict['magnitude'] =  event.magnitudes[0].mag 
            MTdict['magnitude_type'] = x['magnitude_type']
        # for x in event.focal_mechanisms[0]:
        # x = event.focal_mechanisms[0]
        # print(event.origins)
        for x in event.focal_mechanisms:
            # print(x)
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
            elif 'mwr' in str(x.resource_id):
                strike_dip_rake['strike'] = [x.nodal_planes.nodal_plane_1.strike,x.nodal_planes.nodal_plane_2.strike]
                strike_dip_rake['dip'] = [x.nodal_planes.nodal_plane_1.dip,x.nodal_planes.nodal_plane_2.dip] 
                strike_dip_rake['rake'] = [x.nodal_planes.nodal_plane_1.rake,x.nodal_planes.nodal_plane_2.rake] 
                MTdict['PercentDC'] = x.moment_tensor.double_couple
                MTdict['moment'] = x.moment_tensor.scalar_moment
                MTdict['type'] = 'mwr'
                break
        if MTdict['moment'] is None:
            x = event.focal_mechanisms[0]
            strike_dip_rake['strike'] = [x.nodal_planes.nodal_plane_1.strike,x.nodal_planes.nodal_plane_2.strike]
            strike_dip_rake['dip'] = [x.nodal_planes.nodal_plane_1.dip,x.nodal_planes.nodal_plane_2.dip] 
            strike_dip_rake['rake'] = [x.nodal_planes.nodal_plane_1.rake,x.nodal_planes.nodal_plane_2.rake] 
            MTdict['PercentDC'] = x.moment_tensor.double_couple
            MTdict['moment'] = x.moment_tensor.scalar_moment
            MTdict['type'] = 'mwr'


        for x in event.origins:
            if 'mww' in str(x.resource_id) and x.depth_type =='from moment tensor inversion' and MTdict['type'] == 'mww':
                MTdict['centroid_lat'] = x.latitude
                MTdict['centroid_lon'] = x.longitude
                MTdict['Depth_MT'] = x.depth/1000
                break
            elif 'mwb' in str(x.resource_id) and x.depth_type =='from moment tensor inversion' and MTdict['type'] == 'mwb':
                MTdict['centroid_lat'] = x.latitude
                MTdict['centroid_lon'] = x.longitude
                MTdict['Depth_MT'] = x.depth/1000
                break
            elif 'mwr' in str(x.resource_id) and x.depth_type =='from moment tensor inversion' and MTdict['type'] == 'mwr':
                MTdict['centroid_lat'] = x.latitude
                MTdict['centroid_lon'] = x.longitude
                MTdict['Depth_MT'] = x.depth/1000
                break
        if MTdict['centroid_lat'] is None:
            x = event.origin[0]
            MTdict['centroid_lat'] = x.latitude
            MTdict['centroid_lon'] = x.longitude
            MTdict['Depth_MT'] = x.depth/1000
         

        time_pos_depth['Depth'] = event.origins[0].depth/1000
        time_pos_depth['DateTime'] = str(UTCDateTime(event.origins[0].time)).replace('T',' ').replace('Z','')
        time_pos_depth['Position'] = [event.origins[0].latitude,event.origins[0].longitude]
        time_pos_depth['Position_USGS'] = [event.origins[0].latitude,event.origins[0].longitude]
        print(time_pos_depth)
        print(strike_dip_rake)
        print(MTdict)
        return  time_pos_depth, strike_dip_rake, MTdict
    @timeout_decorator.timeout(1000)
    def run_event_ifgm_RScrape(self):
        """
        Creates event csv for USGS event from LiCSEarthquake catalog
        """
        args=["Rscript", "/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/scott_scripts/event_ifgms.R", self.ID ,self.ifgms_csv]
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
            f.write("magnitude type = " + str(self.MTdict['magnitude_type']) + '\n')
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

    def create_beachball(self):
        from obspy.imaging.beachball import beachball 
        figure = plt.figure()
        figure.suptitle('USGS Fault Mechanism: NP1 Strike: ' + 
                        str(round(float(self.strike_dip_rake['strike'][0]),2)) +
                        ' Dip: ' + str(round(float(self.strike_dip_rake['dip'][0]),2))+ 
                        ' Rake: '  +str(round(float(self.strike_dip_rake['rake'][0]),2)) + '\n'
                        'NP2 Strike: ' +  str(round(float(self.strike_dip_rake['strike'][1]),2)) +
                        ' Dip: ' + str(round(float(self.strike_dip_rake['dip'][1]),2))+ 
                        ' Rake: ' + str(round(float(self.strike_dip_rake['rake'][1]),2)))
        mt = [self.strike_dip_rake['strike'][0],self.strike_dip_rake['dip'][0],self.strike_dip_rake['rake'][0]]
        beachball(mt,size=200,linewidth=2,facecolor='r',fig=figure)
        figure.savefig(os.path.join(self.LiCS_locations,self.ID+'_Seismic_beachball.png'))
        return 

    def diameter_mask(self):
        xmin,xmax,xint = -150000,150000,1000
        ymin,ymax,yint = -150000,150000,1000
        x = np.arange(xmin,xmax,xint)
        y= np.arange(ymin,ymax,yint)
        xx,yy = np.meshgrid(x,y)
        xx_vec = np.reshape(xx, -1)
        yy_vec = np.reshape(yy,-1)
        xcen,ycen = 0,0 
        strike = self.strike_dip_rake['strike'][0]
        dip = self.strike_dip_rake['dip'][0]
        rake = self.strike_dip_rake['rake'][0]
        mu = 3.2e10
        slip_rate=5.5e-5
        L = np.cbrt(float(self.MTdict['moment'])/(slip_rate*mu))
        slip = L * slip_rate
        centroid_depth =  self.time_pos_depth['Depth']*1000 
        print('DEPTH FOR GENERATION')
        print(centroid_depth) # setting depth as depth always too deep for MTDict
       
        # print(centroid_depth)
        width = L 
        length = L 
        centroid_depth = centroid_depth + (width/2) * np.sin(np.deg2rad(dip)) 
        widths = []
        checker = False
        while checker is False:
            print('centroid_depth used to generate 0.005m signal mask')
            print(centroid_depth)
            # centroid_depth = 10000
    
            # model = [xcen,ycen,strike,dip,rake,slip,length,centroid_depth,width]
            disp = cslib.disloc3d3(xx_vec,yy_vec,xoff=xcen,yoff=ycen,depth=centroid_depth,length=length,
                                width=width,slip=slip,opening=0,strike=strike,dip=dip,rake=rake,nu=0.25)
            
            disp_E = disp[0,:]
            disp_N = disp[1,:]
            disp_V = disp[2,:]
            # print(len(disp_V))
            # print(len(xx))
            # print(len(xx_vec))
            # cslib.plot_enu(disp,model,x,y)
            # plt.show()
            # print(np.max(disp_E))
            # print(np.min(disp_E)
      
            index_E_above = np.argwhere(np.abs(disp_E)>0.005)
            if len(index_E_above) == 0:
                pass 
            else: 
                widths.append(np.abs(np.max(xx_vec[index_E_above]) - np.min(xx_vec[index_E_above])))
                # print(width_E_X)
                widths.append(np.abs(np.max(yy_vec[index_E_above]) - np.min(yy_vec[index_E_above])))

            index_N_above = np.argwhere(np.abs(disp_N)>0.005)
            if len(index_N_above) == 0:
                pass 
            else: 
                widths.append(np.abs(np.max(xx_vec[index_N_above]) - np.min(xx_vec[index_N_above])))
                # print(width_N_X)
                widths.append(np.abs(np.max(yy_vec[index_N_above]) - np.min(yy_vec[index_N_above])))

            index_V_above = np.argwhere(np.abs(disp_V)>0.005)
            if len(index_V_above) == 0:
                pass 
            else: 
                widths.append(np.abs(np.max(xx_vec[index_V_above]) - np.min(xx_vec[index_V_above])))
                # print(width_V)
                widths.append(np.abs(np.max(yy_vec[index_V_above]) - np.min(yy_vec[index_V_above])))

            centroid_depth = centroid_depth - 1000
            print(centroid_depth)
            if centroid_depth < 0:
                centroid_depth = centroid_depth + 1000
                widths.append((111.13*0.75*1e3))

            if len(widths) == 0:
                checker = False
            elif len(widths) > 0:
                checker = True 

            
            if checker == True and np.max(np.array(widths)) < (111.13*0.5*1e3/3):
                widths =  [111.13*0.5*1e3/3]
            
            if checker == True and np.max(np.array(widths)) > (111.13*1*1e3)/3:
                print('MASK IS HERE')
                widths = [111.13*1*1e3/4]
            # checker = [x if len(widths)>0 else len(widths) == 0]
            # print(len)
            # print(checker)

        print(widths)
        print('mask width')
        print(str(np.max(np.array(widths))))
        print('x3 for fudge factor')
        print(str(np.max(np.array(widths))*3))
        print('Degrees')
        print(str(np.max(np.array(widths))*3/(111.13*1e3)))
        print('potentail Clip')
        print(str(np.max(np.array(widths))*3*4/(111.13*1e3)))
        print('slip')
        print(str(slip))
        print('length-width')
        print(str(L))
        

        return np.max(np.array(widths))*3


    def manual_input(self,strike,dip,rake,moment):
        return        

    def create_folder_stuct(self):
        cwd = os.getcwd()
        print(cwd)
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
        self.GBIS_insar_template_NP1 = os.path.join(self.GBIS_location,self.ID + '_NP1.inp')
        self.GBIS_insar_template_NP2 = os.path.join(self.GBIS_location,self.ID + '_NP2.inp')


       
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
        # if os.path.isfile(self.grond_insar_template):
        #     pass 
        # else:
        #     shutil.copy("insar_rectangular_template.gronf",self.grond_insar_template)

        # if os.path.isdir(self.gf_stores):
        #     pass
        # else:
        #     cwd = os.getcwd() 
        #     os.chdir(self.Grond_location)
        #     sp.call("../download_gf_stores.sh")
        #     os.chdir(cwd)
        # if os.path.isdir(self.crust_location):
        #     pass 
        # else:
        #     cwd = os.getcwd() 
        #     os.chdir(self.gf_stores)
        #     sp.call("../../download_gf_stores.sh")
        #     os.chdir(cwd)

        if os.path.isdir(self.GBIS_location):
            pass 
        else:
            os.mkdir(self.GBIS_location)

        if os.path.isfile(self.GBIS_insar_template_NP1):
            print('Im here ')
            # os.remove(self.GBIS_insar_template_NP1)
            # os.chdir(cwd)
            shutil.copy('example_GBIS_input.inp',self.GBIS_insar_template_NP1)
        else: 
            shutil.copy("example_GBIS_input.inp",self.GBIS_insar_template_NP1)
        
        if os.path.isfile(self.GBIS_insar_template_NP2):
            # os.chdir(cwd)
            # os.remove(self.GBIS_insar_template_NP2)
            shutil.copy("example_GBIS_input.inp",self.GBIS_insar_template_NP2)
        else: 
            shutil.copy('example_GBIS_input.inp',self.GBIS_insar_template_NP2)
        





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

    


