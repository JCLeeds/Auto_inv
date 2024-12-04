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
import pylab as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
def diameter_mask(strike, dip, rake, depth, Mw,xx_vec,yy_vec):

    xcen,ycen = 0,0 
  
    # strike = self.strike_dip_rake['strike'][0]
    # dip = self.strike_dip_rake['dip'][0]
    # rake = self.strike_dip_rake['rake'][0]
    strike = strike
    dip = dip
    rake = rake
    mu = 3.2e10
    slip_rate=5.5e-5
    # L = np.cbrt(float(self.MTdict['moment'])/(slip_rate*mu))
    L = 8000
    # slip = L * slip_rate
    slip = 10**((Mw+10.7)*3/2-7)/(mu*L*L)
    # slip = 0.264161793185
    # L = 4802.94169428
    # print(slip)
    # centroid_depth =  self.MTdict['Depth_MT']
    centroid_depth = depth

    width = L 
    length = L 
    # centroid_depth = depth + (width/2) * np.sin(np.deg2rad(dip)) # Top depth given convert to centorid depth
    model = [xcen,ycen,strike,dip,rake,slip,length,centroid_depth,width]
    disp = cslib.disloc3d3(xx_vec,yy_vec,xoff=0,yoff=0,depth=centroid_depth,length=length,
                        width=width,slip=slip,opening=0,strike=strike,dip=dip,rake=rake,nu=0.25)
    
    disp_E = disp[0,:]
    disp_N = disp[1,:]
    disp_V = disp[2,:]
    # print(len(disp_V))
    # print(len(xx))
    # print(len(xx_vec))
    # # cslib.plot_enu(disp,model,x,y)
    # # plt.show()
    # print(np.max(disp_E))
    # print(np.min(disp_E))

    
    

    # index_E_above = np.argwhere(np.abs(disp_E)>0.01)
    # # print(len(index_E_above))
    # # print(np.shape(index_E_above))
    # # print(len(disp_E))
    # # print(len(disp_E) -len(index_E_above))
    # widths = []
    # if len(index_E_above) == 0:
    #     pass 
    # else: 
    #     widths.append(np.abs(np.max(xx_vec[index_E_above]) - np.min(xx_vec[index_E_above])))
    #     # print(width_E_X)
    #     widths.append(np.abs(np.max(yy_vec[index_E_above]) - np.min(yy_vec[index_E_above])))

    # index_N_above = np.argwhere(np.abs(disp_N)>0.01)
    # if len(index_N_above) == 0:
    #     pass 
    # else: 
    #     widths.append(np.abs(np.max(xx_vec[index_N_above]) - np.min(xx_vec[index_N_above])))
    #     # print(width_N_X)
    #     widths.append(np.abs(np.max(yy_vec[index_N_above]) - np.min(yy_vec[index_N_above])))

    # index_V_above = np.argwhere(np.abs(disp_V)>0.01)
    # if len(index_V_above) == 0:
    #     pass 
    # else: 
    #     widths.append(np.abs(np.max(xx_vec[index_V_above]) - np.min(xx_vec[index_V_above])))
    #     # print(width_V)
    #     widths.append(np.abs(np.max(yy_vec[index_V_above]) - np.min(yy_vec[index_V_above])))



    index_E_above = np.argwhere(np.abs(disp_E)>0.001)
    widths = []
    checker = False
    while checker is False:
        if len(index_E_above) == 0:
            pass 
        else: 
            widths.append(np.abs(np.max(xx_vec[index_E_above]) - np.min(xx_vec[index_E_above])))
            # print(width_E_X)
            widths.append(np.abs(np.max(yy_vec[index_E_above]) - np.min(yy_vec[index_E_above])))

        index_N_above = np.argwhere(np.abs(disp_N)>0.001)
        if len(index_N_above) == 0:
            pass 
        else: 
            widths.append(np.abs(np.max(xx_vec[index_N_above]) - np.min(xx_vec[index_N_above])))
            # print(width_N_X)
            widths.append(np.abs(np.max(yy_vec[index_N_above]) - np.min(yy_vec[index_N_above])))

        index_V_above = np.argwhere(np.abs(disp_V)>0.001)
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
            widths.append((111.13*0.2*1e3))

        if len(widths) == 0:
            checker = False
        elif len(widths) > 0:
            checker = True 
    
        if checker == True and np.max(np.array(widths)) < (111.13*0.5*1e3/3):
            widths =  [111.13*0.5*1e3/3]
        
        if checker == True and np.max(np.array(widths)) > (111.13*1*1e3)/3:
            print('MASK IS HERE')
            widths = [111.13*1*1e3/4]

    plt.scatter(xx_vec,yy_vec,c=disp_E)
    plt.colorbar()
    plt.show()
    plt.scatter(xx_vec[~index_E_above],yy_vec[~index_E_above],c=disp_E[~index_E_above])
    plt.colorbar()
    plt.show()
    plt.scatter(xx_vec[index_E_above],yy_vec[index_E_above],c=disp_E[index_E_above])
    plt.colorbar()
    plt.show()
    
    return np.max(np.array(widths))*3
    # if widths:
    #     return np.max(np.array(widths))
    # else:
    #     return 0
    


  


    # plt.scatter(xx_vec[~index_N_above],yy_vec[~index_N_above],c=disp_E[~index_N_above])
    # plt.scatter(np.max(xx_vec[index_N_above]),np.max(yy_vec[index_N_above]),c='red')
    # plt.scatter(np.min(xx_vec[index_N_above]),np.min(yy_vec[index_N_above]),c='red')
    # plt.colorbar()
    # plt.show()
    # plt.scatter(np.max(xx_vec[index_V_above]),np.max(yy_vec[index_V_above]),c='red')
    # plt.scatter(np.min(xx_vec[index_V_above]),np.min(yy_vec[index_V_above]),c='red')
    # plt.scatter(xx_vec[~index_V_above],yy_vec[~index_V_above],c=disp_E[~index_V_above])
    # plt.colorbar()
    # plt.show()
    
    # return np.max(np.array(widths))

if __name__ == "__main__":

 
    latitude = 33.1157
    longitude = 92.7977
    magnitude = 5.7
    moment = 1.95e+17
    depth = 100
    strike1 = 336.11
    dip1 = 86.04
    rake1 = -167.58
    strike2 = 245.24
    dip2 = 77.61
    rake2 = -4.06
    xmin,xmax,xint = -150000,150000,1000
    ymin,ymax,yint = -150000,150000,1000
    x = np.arange(xmin,xmax,xint)
    y= np.arange(ymin,ymax,yint)
    xx,yy = np.meshgrid(x,y)
    xx_vec = np.reshape(xx, -1)
    yy_vec = np.reshape(yy,-1)


    # # name = us7000ljvg
    # # time = 2023-12-18 15:59:30.352000
    # latitude = 35.7386
    # longitude = 102.8149
    # magnitude = 5.9
    # # magnitude type = Mww
    # moment = 1.01e+18
    # depth = 100
    # # catalog = USGS 
    # strike1 = 156.36
    # dip1 = 28.19
    # rake1 = 93.1
    # strike2 = 332.84
    # dip2 = 61.85
    # rake2 = 88.34

    # name = us7000abnv
    strike = 174
    dip = 83 
    rake = 0 
    depth = 10000
    magnitude = 5.9 
  




    widths = diameter_mask(strike1, dip1,rake1,depth,magnitude,xx_vec,yy_vec)
    print(widths)
    print((np.max(widths)*4 /(111.13*1e3)))
    # width = diameter_mask(strike,dip,rake,depth,Mw)
    # print(width)
    # eq_data = {'magnitude': [5.5,5.6,5.7,5.8,5.9,6, \
    #                           6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7, \
    #                           7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8, \
    #                           8.1,8.2,8.3,8.4,8.5],
    #             'distance': [20,20,21,25,30,36, \
    #                   42,50,60,71,84,100,119,141,167,199, \
    #                   236,280,333,396,471,559,664,790,938,1115, \
    #                   1325, 1575, 1872, 2225, 2500],
    #             'depth': [10,11,12,14,16,18, \
    #                21,24,28,32,36,42,48,55,63,73, \
    #                83,96,110,127,146,168,193,222,250,250, \
    #                250, 250, 250, 250, 250]}
    # # X, Y = np.meshgrid(eq_data['magnitude'], eq_data['depth'])
    # # plt.scatter(eq_data['magnitude'],eq_data['depth'],c=eq_data['distance'])
    # # sns.contourf(X, Y, eq_data['distance'])
    # # plt.show()
    # eq_limits = pd.DataFrame(eq_data)
    
    
    # strike_values = np.random.randint(low=0,high=360,size=100) 
    # rake_values = np.random.randint(low=-180,high=180,size=100) 
    # dip_values = np.random.randint(low=0,high=90,size=100) 

    # strike = 180 
    # dip = 90
    # rake = 180


    # depth = np.arange(1,251,5) * 1000 # top depth converted in function 
    # Mw = np.arange(5,8.5,((8.5-5)/50))
    # all_distances = []
    # all_Mw = [] 
    # all_depth = []

    # xmin,xmax,xint = -250000,250000,1500
    # ymin,ymax,yint = -250000,250000,1500
    # x = np.arange(xmin,xmax,xint)
    # y= np.arange(ymin,ymax,yint)
    # xx,yy = np.meshgrid(x,y)
    # xx_vec = np.reshape(xx, -1)
    # yy_vec = np.reshape(yy,-1)

    # for ii in range(len(Mw)):
    #     for jj in range(len(depth)):
    #         all_distances.append(diameter_mask(strike,dip,rake,depth[jj],Mw[ii],xx_vec,yy_vec)/ 1000)
    #         all_Mw.append(Mw[ii])
    #         all_depth.append(depth[jj]/1000)
    

    # print(len(all_depth))
    # print(len(all_Mw))
    # print(len(all_distances))

    # np.save('all_dist.npz',all_distances)
    # np.save('all_Mw.npz',all_Mw)
    # np.save('all_depth.npz',all_depth)


    # # print(all_depth[0:10])
    # # print(all_Mw[0:10])
    # # print(all_distances[0:10])
    

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    
    # ax.scatter(all_Mw,all_depth,all_distances,marker='^')
    # ax.scatter(eq_data['magnitude'],eq_data['depth'],eq_data['distance'],c='red', marker='o')
    # ax.set_xlabel('Magnitude')
    # ax.set_ylabel('Depth')
    # ax.set_zlabel('Distance')
    # plt.show()

    # new_fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.scatter(eq_data['magnitude'],eq_data['depth'],eq_data['distance'], marker='o')
    # plt.scatter(all_Mw,all_distances)
    # plt.scatter(eq_data['magnitude'],eq_data['distance'])
    # plt.xlabel('Magnitude')
    # plt.ylabel('Distance')
    # plt.show()

    # new_new_fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.scatter(eq_data['magnitude'],eq_data['depth'],eq_data['distance'], marker='o')
    # plt.scatter(all_depth,all_distances)
    # plt.scatter(eq_data['depth'],eq_data['distance'])
    # plt.xlabel('Depth')
    # plt.ylabel('Distance')
    # plt.show()


    # new_new_fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.scatter(eq_data['magnitude'],eq_data['depth'],eq_data['distance'], marker='o')
    # plt.scatter(all_Mw,all_depth,c=all_distances)
    # plt.scatter(eq_data['magnitude'],eq_data['depth'],c='red')
    # plt.xlabel('magnitude')
    # plt.ylabel('depth')
    # plt.title('Deformation at 1cm for a Pure Strike-Slip Earthquake over a Range of Depth and Magnitudes')
    # plt.colorbar(label='Distance over which 1 cm deformation occurs (km)')
    # plt.show()

