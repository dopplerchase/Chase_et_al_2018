import h5py
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates
import pyresample as pr
from scipy.spatial import cKDTree
from pyproj import Proj
from scipy.interpolate import interp1d
import scipy
import pandas as pd
import netCDF4

def apr3tocit(apr3filename,fl,sphere_size,psd_filename_2ds,psd_filename_HVPS,query_k = 1,plotson=False,QC=False,slimfast=True,cit_aver=False,cit_aver2=False,
              attenuation_correct=False,O2H2O={},per_for_atten = 50,
              return_indices=False,BB=True,bbguess=500,
              cal_adj_bool = False,cal_adj=0,
              cloudtop=True,rollfix=True):
    
    """
    =================
    
    This function finds either the closest gate or averages over a number of gates (query_k) nearest to 
    the citation aircraft in the radar volume of the APR3. It can return a dict of the original full length
    arrays and the matched arrays. 
    
    =====
    Vars:
    =====
    apr3filename = str, filename of the apr hdf file
    fl = awot object, the citation awot object
    sphere_size = int, maximum distance allowed in the kdTree search
    psd_filename_2ds = str, filename of the processed 2DS file
    psd_filename_HVPS = str, filename of the processed HVPS3 file
    query_k = int, number of gates considered in the average (if 1, use closest)
    plotson = boolean, will create some premade plots that describe the matched data
    QC = boolean, will apply a simple QC method: eliminates any gate within 0.5 km to the surface and the outliers 
    (plus/minus 1.5IQR)
    slimfast = boolean, will not save original data. Cuts down on output file size by only outputting the matched data and the citation data.
    cit_aver = boolean, averages the ciation data varibles using a 5 second moving average (there is overlap)
    cit_aver2 = boolean, averages the ciation data varibles using a 5 second discrete average (there is NO overlap)
    O2H20 = dict, data from sounding to correct for attenuation from O2 and H2O vapor 
    attenuation_correct = boolean, corrects for attenuation using LWC prof and Sounding. Uses 50th percentile of LWC Prof
    per_for_atten = int, the percentile for the supercooled liquid water profile used in the attenuation correction. 
    return_indeices of matches = boolean, returns the matched gates in 1d coords
    BB = boolean, mask gates from the BB and lower. Masks data using the BB_alt algorithm
    bbguess = int, give your first guess of where the Bright Band is to assist the BB_alt algorithm
    cal_adj_bool = bool, turn on calibration adjustment or not. 
    cal_adj = array, array of the adjustment needed for correct calibration between frequencies. [ka_adj, w_adj]
    cloudtop = bool, eliminates sensativity issues with the Ku-band data (~ < 10 dBZ) by masking out the cloudtop noise using a gausian filter
    rollfix = bool, turn on or off the masking of data where the plane is rolling more than 10 degrees (can change the degree of degrees). 
    
    =================
    """
    
    #get citation times (datetimes)
    cit_time = fl['time']['data']

    #Eliminate BB?
    if BB:
    #Get rid of anything below the melting level + 250 m 
        apr = apr3read(apr3filename)
        
        #there are two methods to this. One is more conservative (using mean Ku) the other more intense with LDR Ku
        #apr = BB_alt(apr,bbguess) #old
        
        if cloudtop:
            print('Removing cloudtop noise..')
            apr = cloudtopmask(apr)
            
            ###new BB tech 2/27/18 RJC
            print('Removing BB and below')
            apr = mask_surf(apr)
            apr['ldr'] = np.ma.masked_where(apr['Ku'].mask,apr['ldr'])
            #find bb profs
            bb = precip_echo_filt3D(apr['ldr'],thresh=7)
            ind1 = np.where(bb[12,:] == 1) #BB profiles based on LDR
            top_a = find_bb(apr,ind1)
            bb_long = extend_bb(ind1,apr['timedates'][12,:],top_a)
            apr['Ku'][:,:,:] = np.ma.masked_where(apr['alt_gate'][:,:,:] <= bb_long,apr['Ku'][:,:,:])
            apr['Ka'] = np.ma.masked_where(apr['Ku'].mask,apr['Ka'])
            apr['W'] = np.ma.masked_where(apr['Ku'].mask,apr['W'])
            ###
            
        #correct for attenuation using SLW and Ku
        if attenuation_correct:
            print('correcting for attenuation...')
            apr = atten_cor3(apr,fl,per_for_atten,O2H2O,lwc_alt=False)
            print('corrected.')
            maxchange = apr['maxchange']
    elif attenuation_correct:
        print('correcting for attenuation...')
        apr = atten_cor2(apr3filename,fl,per_for_atten,O2H2O,lwc_alt=False)
        print('corrected.')
        maxchange = apr['maxchange']
        
    else:    
        apr = apr3read(apr3filename)
        if cloudtop:
            print('Removing cloudtop noise..')
            apr = cloudtopmask(apr)
    
    if cal_adj_bool:
        print('adding calibration means...')
        # These values come from the analysis preformed by 3 reasearch groups: NASA JPL, University of Leister, and the University of Illinois. Techniques use sigma_0 of the ocean surface, comparision of frequencies at low Z and numerical simulations of particles.(error/uncertainty:+- 0.5 dB)
        apr['Ku'] = apr['Ku'] + 0.8
        apr['Ka'] = apr['Ka'] + 1
        #Whh is the only one with a time varient calibration adjustment 
        apr['W'] = apr['W'] + cal_adj
    
    #While calibrating the data, radar artifacts showed up when the roll of the aircraft was > 10degrees. 
    if rollfix:
        roll = apr['roll']
        roll3d = np.zeros(apr['Ku'].shape)
        for i in np.arange(0,apr['Ku'].shape[1]):
            for j in np.arange(0,apr['Ku'].shape[2]):
                roll3d[:,i,j] = roll[i,j]
        
        apr['Ku'] = np.ma.masked_where(np.abs(roll3d) > 10, apr['Ku'])
        apr['Ka'] = np.ma.masked_where(np.abs(roll3d) > 10, apr['Ka'])
        apr['W'] = np.ma.masked_where(np.abs(roll3d) > 10, apr['W'])
    
    #Get APR3 times (datetimes)
    time_dates = apr['timedates'][:,:]
    
    #fix a few radar files where w-band disapears
    if time_dates[12,0] >= datetime.datetime(2015,12,18,6,58):
        for i in np.arange(0,time_dates.shape[0]):
            for j in np.arange(0,550):
                temp = np.ma.masked_where(time_dates[12,:] >= datetime.datetime(2015,12,18,7,6),apr['W'][j,i,:])
                apr['W'][j,i,:] = temp

    if time_dates[12,0] >= datetime.datetime(2015,12,1,23,43,48) and time_dates[12,0] <=datetime.datetime(2015,12,1,23,43,49):
        for i in np.arange(0,time_dates.shape[0]):
            for j in np.arange(0,550):
                temp = np.ma.masked_where(time_dates[12,:] >= datetime.datetime(2015,12,2,0,1,40),apr['W'][j,i,:])
                apr['W'][j,i,:] = temp

    #Check if radar file is large enought to use (50 gates is arbitrary)
    if time_dates[12,:].shape[0] < 50:
        print('Limited radar gates in time')
        #return
    #
    
    #Load PSD
    dtime_psd,ND,dD,midpoints = PSD_load(psd_filename_2ds,psd_filename_HVPS,day = time_dates[0,0].day,month=time_dates[0,0].month)
    #
    
    #Make ND a masked array (i.e. get rid of nans from loading it in)
    ind = np.isnan(ND)
    ND = np.ma.masked_where(ind,ND)

    #for plotting routine
    fontsize=14
    #

    #Varibles needed for the kdtree
    leafsize = 16
    query_eps = 0
    query_p=2
    query_distance_upper_bound = sphere_size
    query_n_jobs =1
    Barnes = True
    K_d = sphere_size
    #


    #Pre-Determine arrays
    Ku_gate = np.ma.array([])
    Ka_gate = np.ma.array([])
    W_gate =  np.ma.array([])
    DFR_gate =  np.ma.array([])
    DFR2_gate =  np.ma.array([])
    DFR3_gate =  np.ma.array([])
    lon_c = np.ma.array([])
    lat_c = np.ma.array([])
    alt_c = np.ma.array([])
    t_c = np.ma.array([])
    lon_r = np.ma.array([])
    lat_r = np.ma.array([])
    alt_r = np.ma.array([])
    t_r = np.ma.array([])
    dis_r = np.ma.array([])
    ind_r = np.ma.array([])
    conc_hvps3 =  np.ma.array([])
    T_c =  np.ma.array([])
    lwc_c =  np.ma.array([])
    ice_c =  np.ma.array([])
    cdp_c =  np.ma.array([])
    twc_c =  np.ma.array([])
    iwc_c =  np.ma.array([])
    #


    #Set reference point (Currently Mount Olympus, Washington)
    lat_0 = 47.7998
    lon_0 = -123.7066
    #

    #Set up map projection to calculate cartesian distances
    p = Proj(proj='laea', zone=10, ellps='WGS84',
             lat_0=lat_0,
             lon_0=lon_0)
    #

    #make a 1d array of times and find radar start and end times
    td = np.ravel(time_dates)
    datestart = td[0]
    dateend = td[td.shape[0]-1] 
    #
    
    #Expand apr3 time to plus/minus 4 mins (added 11/8/17) 4 minutes is arbitrary, but what I used for 'good' matches.
    datestart = datestart - datetime.timedelta(minutes=4)
    dateend = dateend + datetime.timedelta(minutes=4)
    #

    #Constrain Citation data to radar time
    ind = np.where(cit_time > datestart)
    ind2 = np.where(cit_time < dateend)
    ind3 = np.intersect1d(ind,ind2)
    cit_time2 = fl['time']['data'][ind3]
    cit_lon = fl['longitude']['data'][ind3]
    cit_lat = fl['latitude']['data'][ind3]
    cit_alt = fl['altitude']['data'][ind3]
    
    bigins = 0
    #

    #Average Citation data 
    if cit_aver:
        
        #Moving average tech.

        temp1 = fl['temperature']['data']
        temp2 = fl['lwc1']['data']
        temp3 = fl['mso_frequency']['data']
        temp4 = fl['Conc_CDP']['data']
        temp5 = fl['twc']['data']
        temp6 = fl['Nev_IWC']['data']
        temp7 = fl['dewpoint_temperature1']['data']
        temp8 = fl['Wwind']['data']
        temp9 = fl['static_pressure']['data']
        temp10 = fl['mixing_ratio']['data']
        temp11 = fl['Uwind']['data']
        temp12 = fl['Vwind']['data']


        nsecs = 2
        indarray1 = ind3 - nsecs
        indarray2 = ind3 + nsecs + 1

        temperature_1 = np.ma.zeros(len(ind3))
        lwc = np.ma.zeros(len(ind3))
        ice = np.ma.zeros(len(ind3)) 
        cdp = np.ma.zeros(len(ind3)) 
        twc = np.ma.zeros(len(ind3)) 
        iwc = np.ma.zeros(len(ind3))
        td = np.ma.zeros(len(ind3))
        w = np.ma.zeros(len(ind3))
        P = np.ma.zeros(len(ind3))
        mix = np.ma.zeros(len(ind3))
        U = np.ma.zeros(len(ind3))
        V = np.ma.zeros(len(ind3))
        for i in np.arange(0,len(ind3)):
            temperature_1[i] = np.ma.mean(temp1[indarray1[i]:indarray2[i]])
            lwc[i] = np.ma.mean(temp2[indarray1[i]:indarray2[i]])
            ice[i] = np.ma.mean(temp3[indarray1[i]:indarray2[i]])
            cdp[i] = np.ma.mean(temp4[indarray1[i]:indarray2[i]])
            twc[i] = np.ma.mean(temp5[indarray1[i]:indarray2[i]])
            iwc[i] = np.ma.mean(temp6[indarray1[i]:indarray2[i]])
            td[i] = np.ma.mean(temp7[indarray1[i]:indarray2[i]])
            w[i] = np.ma.mean(temp8[indarray1[i]:indarray2[i]])
            P[i] = np.ma.mean(temp9[indarray1[i]:indarray2[i]])
            mix[i] = np.ma.mean(temp10[indarray1[i]:indarray2[i]])
            U[i] = np.ma.mean(temp11[indarray1[i]:indarray2[i]])
            V[i] = np.ma.mean(temp12[indarray1[i]:indarray2[i]])
            
        #Find average N(D)
        ND_sub_a = np.ma.zeros(ND[0,:].shape)
        ND_aver = np.ma.zeros([ind3.shape[0],ND[0,:].shape[0]])
        
        for i in np.arange(0,ind3.shape[0]):
            
            if indarray2[i] > ND.shape[0]:
                print('indarray4 is too big')
                break
                
            ND_sub = ND[indarray1[i]:indarray2[i],:]
            ind = np.where(ND_sub < 0)
            ND_sub[ind] = np.ma.masked
            
            for j in np.arange(ND.shape[1]):
                ND_sub_a[j] = np.ma.mean(ND_sub[:,j])
            
            ND_aver[i,:] = ND_sub_a      
    
    elif cit_aver2:
        
        #Discrete average tech.
        
        temp1 = fl['temperature']['data'][ind3]
        temp2 = fl['lwc1']['data'][ind3]
        temp3 = fl['mso_frequency']['data'][ind3]
        temp4 = fl['Conc_CDP']['data'][ind3]
        temp5 = fl['twc']['data'][ind3]
        temp6 = fl['Nev_IWC']['data'][ind3]
        temp7 = fl['dewpoint_temperature1']['data'][ind3]
        temp8 = fl['Wwind']['data'][ind3]
        temp9 = fl['static_pressure']['data'][ind3]
        temp10 = fl['mixing_ratio']['data'][ind3]
        temp11 = fl['Uwind']['data'][ind3]
        temp12 = fl['Vwind']['data'][ind3]
        
        ND = ND[ind3,:]
        
        max_dtime = cit_time2.max()
        min_dtime = cit_time2.min()
        total_seconds = max_dtime-min_dtime
        total_seconds = total_seconds.total_seconds()
        dtime_1s = np.zeros(int(total_seconds)-1,dtype=object)
        its = dtime_1s.shape[0]/5.
        dtime_5s = np.zeros(int(its),dtype=object)
        array = np.ma.zeros(int(its))
        array2 = np.ma.zeros(int(its))
        array3 = np.ma.zeros(int(its))
        array4 = np.ma.zeros(int(its))
        array5 = np.ma.zeros(int(its))
        array6 = np.ma.zeros(int(its))
        array7 = np.ma.zeros(int(its))
        array8 = np.ma.zeros(int(its))
        array9 = np.ma.zeros(int(its))
        array10 = np.ma.zeros(int(its))
        array11 = np.ma.zeros(int(its))
        array12 = np.ma.zeros(int(its))
        array13 = np.ma.zeros(int(its))
        array14 = np.ma.zeros(int(its))
        array15 = np.ma.zeros(int(its))
        #create dtime_array monotonic increase but 5 seconds 
        for i in np.arange(0,int(its)):
            dtime_5s[i] = min_dtime + datetime.timedelta(seconds = i*5)


        print('time averaging into 5 second averages...')
        for i in np.arange(1,dtime_5s.shape[0]):
            time_left = dtime_5s[i-1]
            time_right = dtime_5s[i]

            ind = np.where(cit_time2 >= time_left)
            ind2 = np.where(cit_time2 < time_right)
            ind3 = np.intersect1d(ind,ind2)

            if len(ind3) >= 1:
                temp = temp1[ind3]
                array[i-1] = np.ma.mean(temp)
                temp = temp2[ind3]
                array2[i-1] = np.ma.mean(temp)
                temp = temp3[ind3]
                array3[i-1] = np.ma.mean(temp) 
                temp = temp4[ind3]
                array4[i-1] = np.ma.mean(temp)
                temp = temp5[ind3]
                array5[i-1] = np.ma.mean(temp)
                temp = temp6[ind3]
                array6[i-1] = np.ma.mean(temp)
                temp = temp7[ind3]
                array7[i-1] = np.ma.mean(temp)
                temp = temp8[ind3]
                array8[i-1] = np.ma.mean(temp)
                temp = temp9[ind3]
                array9[i-1] = np.ma.mean(temp) 
                temp = temp10[ind3]
                array10[i-1] = np.ma.mean(temp)
                temp = temp11[ind3]
                array11[i-1] = np.ma.mean(temp)
                temp = temp12[ind3]
                array12[i-1] = np.ma.mean(temp)
                temp = cit_lat[ind3]
                array13[i-1] = np.ma.mean(temp)
                temp = cit_lon[ind3]
                array14[i-1] = np.ma.mean(temp)
                temp = cit_alt[ind]
                array15[i-1] = np.ma.mean(temp)


            else:

                array[i-1] = np.ma.masked
                array2[i-1] = np.ma.masked
                array3[i-1] = np.ma.masked
                array4[i-1] = np.ma.masked
                array5[i-1] = np.ma.masked
                array6[i-1] =np.ma.masked
                array7[i-1] = np.ma.masked
                array8[i-1] = np.ma.masked
                array9[i-1] = np.ma.masked
                array10[i-1] = np.ma.masked
                array11[i-1] = np.ma.masked
                array12[i-1] = np.ma.masked
                array13[i-1] = np.ma.masked
                array14[i-1] = np.ma.masked
                array15[i-1] = np.ma.masked

                continue
                
        

        #pre-allocate arrays 
        ND_sub_a = np.ma.zeros(ND[0,:].shape)
        ND_aver = np.ma.zeros([dtime_5s.shape[0],ND[0,:].shape[0]])
        #
        ind = np.where(ND < 0)
        ND[ind] = np.ma.masked
        for i in np.arange(1,dtime_5s.shape[0]):
            time_left = dtime_5s[i-1]
            time_right = dtime_5s[i]

            ind = np.where(cit_time2 >= time_left)
            ind2 = np.where(cit_time2 < time_right)
            ind3 = np.intersect1d(ind,ind2)

            if len(ind3) >= 1:
                ND_sub = ND[ind3,:]
                for j in np.arange(ND.shape[1]):
                    ND_sub_a[j] = np.ma.mean(ND_sub[:,j])

                ND_aver[i-1,:] = ND_sub_a
            else:
                ND_aver[i-1,:] = np.ma.masked
                    
        #get rid of last point (less than 5 obs needed for average)
        temperature_1 = array[:-1]
        lwc = array2[:-1]
        ice = array3[:-1]
        cdp = array4[:-1]
        twc = array5[:-1]
        iwc = array6[:-1]
        td = array7[:-1]
        w = array8[:-1]
        P = array9[:-1]
        mix = array10[:-1]
        U = array11[:-1]
        V = array12[:-1]
        cit_lat = array13[:-1]
        cit_lon = array14[:-1]
        cit_alt = array15[:-1]
        ND_aver = ND_aver[:-1,:]
        
        #In reality our time should be the midpoint of each time interval. I will add 2.5 seconds to the 5s array
        cit_time2 = dtime_5s[:-1] + datetime.timedelta(seconds=2.5)
        
        
        #get rid of masked spatial cit data. Kd tree doesnt liked masked values (i.e. fill_values sneak in)
        ind = cit_lon.mask
        cit_lon = cit_lon[~ind]
        cit_lat = cit_lat[~ind]
        cit_alt = cit_alt[~ind]
        cit_time2 = cit_time2[~ind]
        temperature_1 = temperature_1[~ind]
        lwc = lwc[~ind]
        ice = ice[~ind]
        cdp = cdp[~ind]
        twc = twc[~ind]
        iwc = iwc[~ind]
        td = td[~ind]
        w = w[~ind]
        P = P[~ind]
        mix = mix[~ind]
        U = U[~ind]
        V = V[~ind]
        ND_aver = ND_aver[~ind,:]

        ind = cit_lat.mask
        cit_lon = cit_lon[~ind]
        cit_lat = cit_lat[~ind]
        cit_alt = cit_alt[~ind]
        cit_time2 = cit_time2[~ind]
        temperature_1 = temperature_1[~ind]
        lwc = lwc[~ind]
        ice = ice[~ind]
        cdp = cdp[~ind]
        twc = twc[~ind]
        iwc = iwc[~ind]
        td = td[~ind]
        w = w[~ind]
        P = P[~ind]
        mix = mix[~ind]
        U = U[~ind]
        V = V[~ind]
        ND_aver = ND_aver[~ind,:]

        ind = cit_alt.mask
        cit_lon = cit_lon[~ind]
        cit_lat = cit_lat[~ind]
        cit_alt = cit_alt[~ind]
        cit_time2 = cit_time2[~ind]
        temperature_1 = temperature_1[~ind]
        lwc = lwc[~ind]
        ice = ice[~ind]
        cdp = cdp[~ind]
        twc = twc[~ind]
        iwc = iwc[~ind]
        td = td[~ind]
        w = w[~ind]
        P = P[~ind]
        mix = mix[~ind]
        U = U[~ind]
        V = V[~ind]
        ND_aver = ND_aver[~ind,:]
      
    else:
        
        #no averaging tech.
        temperature_1 = fl['temperature']['data'][ind3]
        lwc = fl['lwc1']['data'][ind3]
        ice = fl['mso_frequency']['data'][ind3]
        cdp = fl['Conc_CDP']['data'][ind3]
        twc = fl['twc']['data'][ind3]
        iwc = fl['Nev_IWC']['data'][ind3]
        td = fl['dewpoint_temperature1']['data'][ind3]
        w = fl['Wwind']['data'][ind3]
        P = fl['static_pressure']['data'][ind3]
        mix = fl['mixing_ratio']['data'][ind3]
        U = fl['Uwind']['data'][ind3]
        V = fl['Vwind']['data'][ind3]
        ND = ND[ind3,:]
        #
    # ND is in cm**-4 and dD+midpoints is in mm

    
    #Find the echotop of Ku at near nadir 
    print('finding Ku echotop and constraining Cit...')
    precip_yn = precip_echo_filt(apr['Ku'][:,12,:])
    ind = np.where(precip_yn ==1)
    ku_filt = np.squeeze(apr['Ku'][:,12,ind])
    alt_filt = np.squeeze(apr['alt_gate'][:,12,ind])
    echo = find_echo(ku_filt,alt_filt) 
    scan = 12
    lat_0 = apr['lat'][scan,0]
    lon_0 = apr['lon'][scan,0]
    p2 = Proj(proj='laea', zone=10, ellps='WGS84',
             lat_0=lat_0,
             lon_0=lon_0)

    x = apr['lon_gate'][:,scan,:]
    y = apr['lat_gate'][:,scan,:]

    x2,y2 = p2(x,y)
    x3,y3 = p2(lon_0,lat_0)
    x_c,y_c = p2(cit_lon,cit_lat)
    alt_c = cit_alt

    x4 = np.array([])
    y4 = np.array([])
    x2_c = np.array([])
    y2_c = np.array([])
    for j in np.arange(0,x2.shape[1]):
        x4 = np.append(x4,x2[0,j]-x3)
        y4 = np.append(y4,y2[0,j]-y3)
    for j in np.arange(0,x_c.shape[0]):
        x2_c = np.append(x2_c,x_c[j]-x3)
        y2_c = np.append(y2_c,y_c[j]-y3)

    R = np.sqrt(x4**2+y4**2)/1000.
    R_c = np.sqrt(x2_c**2+y2_c**2)/1000.
    R_echo = R[ind]
    echo_func = interp1d(R_echo,echo,kind='cubic',bounds_error=False)
    echo_c = echo_func(R_c)
    
    ind = np.where(alt_c <= echo_c + 50) #can change this threshold, just arbitrary
    cit_lon = cit_lon[ind]
    cit_lat = cit_lat[ind]
    cit_alt = cit_alt[ind]
    cit_time2 = cit_time2[ind]
    temperature_1 = temperature_1[ind]
    lwc = lwc[ind]
    ice = ice[ind]
    cdp = cdp[ind]
    twc = twc[ind]
    iwc = iwc[ind]
    td = td[ind]
    w = w[ind]
    P = P[ind]
    mix = mix[ind]
    U = U[ind]
    V = V[ind]
    ND_aver = np.squeeze(ND_aver[ind,:])
    R_c = R_c[ind]
    echo_c = echo_c[ind]
    #
    
    if BB: 
        print('Constraining Cit above BB..')
        bb_func = interp1d(R,bb_long,kind='cubic',bounds_error=False)
        bb_c = bb_func(R_c)
        ind = np.where(cit_alt >= bb_c - 100) #can change this threshold, just arbitrary
        cit_lon = cit_lon[ind]
        cit_lat = cit_lat[ind]
        cit_alt = cit_alt[ind]
        cit_time2 = cit_time2[ind]
        temperature_1 = temperature_1[ind]
        lwc = lwc[ind]
        ice = ice[ind]
        cdp = cdp[ind]
        twc = twc[ind]
        iwc = iwc[ind]
        td = td[ind]
        w = w[ind]
        P = P[ind]
        mix = mix[ind]
        U = U[ind]
        V = V[ind]
        ND_aver = np.squeeze(ND_aver[ind,:])
        R_c = R_c[ind]
        echo_c = echo_c[ind]
    #


    
    #Mask out warmer than 0 (i.e. when particles melt)
    ind = np.where(temperature_1 > 0)
    ND_aver[ind,:] = np.ma.masked
    #
    
    #Calculate some PSD parameters (could add other things here, i.e. running IGF for Mu,lambda and N0)
    rho_tot2,iwc_HY = rho_e(midpoints/10.,dD/10.,ND_aver,np.zeros(ND_aver.shape),2,2,twc,return_ice=True) #HYs
    rho_tot3,iwc_BF = rho_e(midpoints/10.,dD/10.,ND_aver,np.zeros(ND_aver.shape),2,3,twc,return_ice=True) #BF
    rho_tot4 = rho_e(midpoints/10.,dD/10.,ND_aver,np.zeros(ND_aver.shape),2,4,twc) #BF
    dmm_BF = Dmm(ND_aver*1e8,midpoints/1000.,dD/1000.,0)
    dmm_HY = Dmm(ND_aver*1e8,midpoints/1000.,dD/1000.,1)
    
#     rho_tot2 = 0
#     rho_tot3 =0
#     dmm_BF = Dmm(ND_aver/1e8,midpoints/1000.,dD/1000.,0)
#     dmm_HY = Dmm(ND_aver/1e8,midpoints/1000.,dD/1000.,1)
    #
    

    #Print out number of potential match points
    print(cit_lon.shape)
    #


    #Make 1-D arrays of radar spatial data
    apr_x = np.ravel(apr['lon_gate'][:,:,:])
    apr_y = np.ravel(apr['lat_gate'][:,:,:])
    apr_alt = np.ravel(apr['alt_gate'][:,:,:])
    apr_t = np.ravel(apr['time_gate'][:,:,:])
    #
    
    
    #Make 1-D arrays of radar data
    apr_ku = np.ma.ravel(apr['Ku'][:,:,:])
    apr_ka = np.ma.ravel(apr['Ka'][:,:,:])
    apr_w = np.ma.ravel(apr['W'][:,:,:])
    #
            
    #If you want to neglect masked gates throw them out here (Speeds things up and gives better results)
    #ku
    ind = apr_ku.mask
    apr_x = apr_x[~ind]
    apr_y = apr_y[~ind]
    apr_alt = apr_alt[~ind]
    apr_t = apr_t[~ind]
    apr_ku = apr_ku[~ind]
    apr_ka = apr_ka[~ind]
    apr_w = apr_w[~ind]
    #ka
    ind = apr_ka.mask
    apr_x = apr_x[~ind]
    apr_y = apr_y[~ind]
    apr_alt = apr_alt[~ind]
    apr_t = apr_t[~ind]
    apr_ku = apr_ku[~ind]
    apr_ka = apr_ka[~ind]
    apr_w = apr_w[~ind]
    #w
    ind = apr_w.mask
    apr_x = apr_x[~ind]
    apr_y = apr_y[~ind]
    apr_alt = apr_alt[~ind]
    apr_t = apr_t[~ind]
    apr_ku = apr_ku[~ind]
    apr_ka = apr_ka[~ind]
    apr_w = apr_w[~ind]
    #
    
    #Use projection to get cartiesian distances
    apr_x2,apr_y2 = p(apr_x,apr_y)
    cit_x2,cit_y2 = p(cit_lon,cit_lat)
    #

    #Kdtree things (this is where the matchups are found)
    kdt = cKDTree(zip(apr_x2, apr_y2, apr_alt), leafsize=leafsize)

    prdistance, prind1d = kdt.query(zip(cit_x2,cit_y2,cit_alt),k=query_k, eps=query_eps, p=query_p,
                            distance_upper_bound=query_distance_upper_bound,n_jobs=query_n_jobs)

    #


    #if query_k >1 means you are considering more than one gate and an average is needed

    if query_k > 1:

        #Issue with prind1d being the size of apr_ku... that means that it is outside you allowed upperbound (sphere_size)
        ind = np.where(prind1d == apr_ku.shape[0])
        if len(ind[0]) > 0 or len(ind[1]) > 0:
            print('gate was outside distance upper bound, eliminating those instances')
            
            #mask values outside search area. Actually setting values to 0?
#             prind1d = np.ma.masked_where(prind1d == apr_ku.shape[0],prind1d)
#             prdistance = np.ma.masked_where(prind1d == apr_ku.shape[0],prdistance)
                        
            prind1d[ind] = np.ma.masked
            prdistance[ind] = np.ma.masked
            
        if QC:
        
            #Eliminate observations that are outliers before averaging the data (i.e. get rid of skin paints)
            Ku_sub = apr_ku[prind1d]
            Ku_sub = np.ma.masked_where(prind1d == 0,Ku_sub)
            Q_med = np.array([])
            Q_max = np.array([])
            Q_min = np.array([])
            Q_1 = np.array([])
            Q_2 = np.array([])
            n_1 = np.array([])
            for i in np.arange(Ku_sub.shape[0]):
                kk = Ku_sub[i,:]
                numberofmasks = kk.mask
                kk = kk[~numberofmasks]
                if len(kk) < 1:
                    Q_med = np.append(Q_med,np.nan)
                    Q_max = np.append(Q_max,np.nan)
                    Q_min = np.append(Q_min,np.nan)
                    Q_1 = np.append(Q_1,np.nan)
                    Q_2 = np.append(Q_2,np.nan)
                    n_1 = np.append(n_1,0)
                    continue
                Q = np.nanpercentile(kk,[0,10,25,50,75,90,100])
                Q_med = np.append(Q_med,Q[3])
                Q_max = np.append(Q_max,Q[6])
                Q_min = np.append(Q_min,Q[0])
                Q_1 = np.append(Q_1,Q[2])
                Q_2 = np.append(Q_2,Q[4])
                numberofmasks = np.isnan(kk)
                kk = kk[~numberofmasks]
                #print(notmask)
                notmask = kk.shape[0]
                n_1 = np.append(n_1,notmask)
                
                
            IQR = Q_2 - Q_1
            outlierup = Q_2 + 1.5*IQR
            outlierdown = Q_1- 1.5*IQR

            IQR_ku = IQR
            
            Ku_sub = apr_ku[prind1d]
            Ku_sub = np.ma.masked_where(prind1d == 0,Ku_sub)
            for i in np.arange(Ku_sub.shape[0]):
                Ku_subsub = Ku_sub[i,:]
                Ku_subsub = np.ma.masked_where(Ku_subsub >= outlierup[i],Ku_subsub)
                Ku_sub[i,:] = Ku_subsub

            Ka_sub = apr_ka[prind1d]
            Ka_sub = np.ma.masked_where(prind1d == 0,Ka_sub)
            Q_med = np.array([])
            Q_max = np.array([])
            Q_min = np.array([])
            Q_1 = np.array([])
            Q_2 = np.array([])
            n_2 = np.array([])
            for i in np.arange(Ka_sub.shape[0]):
                kk = Ka_sub[i,:]
                numberofmasks = kk.mask
                kk = kk[~numberofmasks]
                if len(kk) < 1:
                    Q_med = np.append(Q_med,np.nan)
                    Q_max = np.append(Q_max,np.nan)
                    Q_min = np.append(Q_min,np.nan)
                    Q_1 = np.append(Q_1,np.nan)
                    Q_2 = np.append(Q_2,np.nan)
                    n_2 = np.append(n_2,0)
                    continue
                Q = np.nanpercentile(kk,[0,10,25,50,75,90,100])
                Q_med = np.append(Q_med,Q[3])
                Q_max = np.append(Q_max,Q[6])
                Q_min = np.append(Q_min,Q[0])
                Q_1 = np.append(Q_1,Q[2])
                Q_2 = np.append(Q_2,Q[4])
                numberofmasks = np.isnan(kk)
                kk = kk[~numberofmasks]
                notmask = kk.shape[0]
                n_2 = np.append(n_2,notmask)
                
                
            IQR = Q_2 - Q_1
            outlierup = Q_2 + 1.5*IQR
            outlierdown = Q_1- 1.5*IQR

            IQR_ka = IQR
            
            Ka_sub = apr_ka[prind1d]
            Ka_sub = np.ma.masked_where(prind1d == 0,Ka_sub)
            for i in np.arange(Ka_sub.shape[0]):
                Ka_subsub = Ka_sub[i,:]
                Ka_subsub = np.ma.masked_where(Ka_subsub >= outlierup[i],Ka_subsub)
                Ka_sub[i,:] = Ka_subsub

            W_sub = apr_w[prind1d]
            W_sub = np.ma.masked_where(prind1d == 0,W_sub)
            Q_med = np.array([])
            Q_max = np.array([])
            Q_min = np.array([])
            Q_1 = np.array([])
            Q_2 = np.array([])
            n_3 = np.array([])
            for i in np.arange(W_sub.shape[0]):
                kk = W_sub[i,:]
                numberofmasks = kk.mask
                kk = kk[~numberofmasks]
                if len(kk) < 1:
                    Q_med = np.append(Q_med,np.nan)
                    Q_max = np.append(Q_max,np.nan)
                    Q_min = np.append(Q_min,np.nan)
                    Q_1 = np.append(Q_1,np.nan)
                    Q_2 = np.append(Q_2,np.nan)
                    n_3 = np.append(n_3,0)
                    continue
                Q = np.nanpercentile(kk,[0,10,25,50,75,90,100])
                Q_med = np.append(Q_med,Q[3])
                Q_max = np.append(Q_max,Q[6])
                Q_min = np.append(Q_min,Q[0])
                Q_1 = np.append(Q_1,Q[2])
                Q_2 = np.append(Q_2,Q[4])
                numberofmasks = np.isnan(kk)
                kk = kk[~numberofmasks]
                #print(notmask)
                notmask = kk.shape[0]
                n_3 = np.append(n_3,notmask)
                
                
            IQR = Q_2 - Q_1
            outlierup = Q_2 + 1.5*IQR
            outlierdown = Q_1- 1.5*IQR

            IQR_w = IQR
            
            W_sub = apr_w[prind1d]
            W_sub = np.ma.masked_where(prind1d == 0,W_sub)
            for i in np.arange(W_sub.shape[0]):
                W_subsub = W_sub[i,:]
                W_subsub = np.ma.masked_where(W_subsub >= outlierup[i],W_subsub)
                W_sub[i,:] = W_subsub
                
            apr_DFR = apr_ku - apr_ka
            apr_DFR2 = apr_ku - apr_w
            apr_DFR3 = apr_ka - apr_w
            
            #Barnes weighting
            ku_getridof0s = Ku_sub
            ku_getridof0s = np.ma.masked_where(prind1d == 0,ku_getridof0s)
            ku_getridof0s = np.ma.masked_where(np.isnan(ku_getridof0s),ku_getridof0s)
            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(ku_getridof0s), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * 10. **(ku_getridof0s / 10.),axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            ku_temp = 10. * np.ma.log10(w1/w2)


            #Find weighted STD
            IQR_ku2 = np.ma.zeros([ku_getridof0s.shape[0]])
            for i in np.arange(ku_getridof0s.shape[0]):
                ts = np.ma.zeros(len(ku_getridof0s[i,:]))
                for j in np.arange(0,len(ku_getridof0s[i,:])):
                    diffs = np.ma.subtract(ku_getridof0s[i,j],ku_temp[i])
                    diffs = np.ma.power(diffs,2.)
                    ts[j] = diffs
                temporary =  np.ma.sqrt((np.ma.sum(ts)/n_1[i]))
                IQR_ku2[i] = temporary

            ka_getridof0s = Ka_sub
            ka_getridof0s = np.ma.masked_where(prind1d == 0,ka_getridof0s)

            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(ka_getridof0s), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * 10. **(ka_getridof0s / 10.),axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            ka_temp = 10. * np.ma.log10(w1/w2)

            #Find weighted STD
            IQR_ka2 = np.ma.zeros([ka_getridof0s.shape[0]])
            for i in np.arange(ka_getridof0s.shape[0]):
                ts = np.ma.zeros(len(ka_getridof0s[i,:]))
                for j in np.arange(0,len(ka_getridof0s[i,:])):
                    diffs = np.ma.subtract(ka_getridof0s[i,j],ka_temp[i])
                    diffs = np.ma.power(diffs,2.)
                    ts[j] = diffs
                temporary =  np.ma.sqrt((np.ma.sum(ts)/n_2[i]))
                IQR_ka2[i] = temporary

            w_getridof0s = W_sub
            w_getridof0s = np.ma.masked_where(prind1d == 0,w_getridof0s) 

            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(w_getridof0s), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * 10. **(w_getridof0s / 10.),axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            w_temp = 10. * np.ma.log10(w1/w2)

            #Find weighted STD
            IQR_w2 = np.ma.zeros([w_getridof0s.shape[0]])
            for i in np.arange(w_getridof0s.shape[0]):
                ts = np.ma.zeros(len(w_getridof0s[i,:]))
                for j in np.arange(0,len(w_getridof0s[i,:])):
                    diffs = np.ma.subtract(w_getridof0s[i,j],w_temp[i])
                    diffs = np.ma.power(diffs,2.)
                    ts[j] = diffs
                temporary =  np.ma.sqrt((np.ma.sum(ts)/n_3[i]))
                IQR_w2[i] = temporary



            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * prdistance,axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            dis_temp = w1/w2 

            Ku_gate = ku_temp
            Ka_gate = ka_temp
            W_gate = w_temp

            DFR_gate = ku_temp - ka_temp
            DFR2_gate = ku_temp - w_temp
            DFR3_gate = ka_temp - w_temp
            #
            
        else:
            
            #Eliminate observations that are outliers
            Ku_sub = apr_ku[prind1d]
            Ku_sub = np.ma.masked_where(prind1d == 0,Ku_sub)
            Ka_sub = apr_ka[prind1d]
            Ka_sub = np.ma.masked_where(prind1d == 0,Ka_sub)
            W_sub = apr_w[prind1d]
            W_sub = np.ma.masked_where(prind1d == 0,W_sub)
         
            apr_DFR = apr_ku - apr_ka
            apr_DFR2 = apr_ku - apr_w
            apr_DFR3 = apr_ka - apr_w
        #

            #Barnes weighting
            ku_getridof0s = Ku_sub
            ku_getridof0s = np.ma.masked_where(prind1d == 0,ku_getridof0s)
            ku_getridof0s = np.ma.masked_where(np.isnan(ku_getridof0s),ku_getridof0s)
            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(ku_getridof0s), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * 10. **(ku_getridof0s / 10.),axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            ku_temp = 10. * np.ma.log10(w1/w2)

            ka_getridof0s = Ka_sub
            ka_getridof0s = np.ma.masked_where(prind1d == 0,ka_getridof0s)

            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(ka_getridof0s), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * 10. **(ka_getridof0s / 10.),axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            ka_temp = 10. * np.ma.log10(w1/w2)

            w_getridof0s = W_sub
            w_getridof0s = np.ma.masked_where(prind1d == 0,w_getridof0s) 

            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(w_getridof0s), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * 10. **(w_getridof0s / 10.),axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            w_temp = 10. * np.ma.log10(w1/w2)

            W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
            W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * prdistance,axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            dis_temp = w1/w2 

            Ku_gate = ku_temp
            Ka_gate = ka_temp
            W_gate = w_temp

            DFR_gate = ku_temp - ka_temp
            DFR2_gate = ku_temp - w_temp
            DFR3_gate = ka_temp - w_temp
            #

        #append current lat,lon and alt of the citation plane
        lat_c = cit_lat
        lon_c = cit_lon
        alt_c = cit_alt
        t_c = cit_time2
        T_c = temperature_1
        lwc_c = lwc
        ice_c = ice
        cdp_c = cdp
        twc_c = twc
        iwc_c = iwc
        #

        #Use plane location for barnes averaged radar value
        lat_r = cit_lat
        lon_r = cit_lon
        alt_r = cit_alt
        t_r = cit_time2
        #
        dis_r = dis_temp
        ind_r = np.nan

        #Calculate time difference, weighted the same as everything else
        t_tiled = np.empty([t_c.shape[0],query_k],dtype=object)
        for i in np.arange(0,t_c.shape[0]):
            t_tiled[i,:] = t_c[i]
        diftime = apr_t[prind1d] - t_tiled
        diftime2 = np.empty(diftime.shape)
        for i in np.arange(0,diftime.shape[0]-1):
            for j in np.arange(0,diftime.shape[1]-1):
                diftime2[i,j] = diftime[i,j].total_seconds()

        W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(diftime2), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 *diftime2,axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        dif_temp = w1/w2
        
        dif_t = dif_temp
        #

    else:
        
        #For closest gate: Tested 11/09/17

        #If gate outside sphere will need to remove flaged data == apr_ku.shape[0]
        ind = np.where(prind1d == apr_ku.shape[0])
        if len(ind[0]) > 0:
            print('gate was outside distance upper bound, eliminating those instances')
            #mask ind and distances that are outside the search area
            prind1d[ind] = np.ma.masked
            prdistance[ind] = np.ma.masked
    
        #    
        ku_temp = apr_ku[prind1d]
        ka_temp = apr_ka[prind1d]
        w_temp = apr_w[prind1d]
        
        ku_temp = np.ma.masked_where(prind1d == 0,ku_temp)
        ka_temp = np.ma.masked_where(prind1d == 0,ka_temp)
        w_temp = np.ma.masked_where(prind1d == 0,w_temp)
        
        dfr_temp = ku_temp - ka_temp
        dfr2_temp = ku_temp - w_temp
        dfr3_temp = ka_temp - w_temp
        Ku_gate = ku_temp
        Ka_gate = ka_temp
        W_gate = w_temp
        DFR_gate = dfr_temp
        DFR2_gate = dfr2_temp
        DFR3_gate = dfr3_temp
        #

        #append current lat,lon and alt of the citation plane
        lat_c = cit_lat
        lon_c = cit_lon
        alt_c = cit_alt
        t_c = cit_time2
        T_c = temperature_1
        lwc_c = lwc
        ice_c = ice
        cdp_c = cdp
        twc_c = twc
        iwc_c = iwc
        #

        diftime = apr_t[prind1d] - t_c
        diftime2 = np.empty(diftime.shape)
        for i in np.arange(0,diftime.shape[0]):
                diftime2[i] = diftime[i].total_seconds()

        #Get radar gate info and append it
        lat_r = apr_y[prind1d]
        lon_r = apr_x[prind1d]
        alt_r = apr_alt[prind1d]
        t_r = apr_t[prind1d]
        dis_r = prdistance
        ind_r = prind1d
        dif_t = diftime2

    
    #Make lists full of all the data
    matcher = {}
    Cit = {}
    APR = {}
    matched = {}
    kdtree = {}
    info_c = {}
    info_r = {}
    info_m = {}
    info_k = {}
    
    #Pack values in lists for export
 
    
    info_k['prind1d'] = 'Index in the raveled apr3 array of the selected gate/gates. Units = None'
    info_k['prdistance'] = 'Cartesian distance between Citation and "matched" radar gate. This will be a barnes average if query_k is greater than 1. Units = meters'
    info_k['query_k'] = 'How many gates were considered to be matched. Units = None'
    
    kdtree['prind1d'] = prind1d
    kdtree['prdistance'] = prdistance
    kdtree['query_k'] = query_k
    kdtree['info'] = info_k
    
    info_c['lat'] = 'Latitude of the citation aircraft. Units = Degrees'
    info_c['lon'] = 'Longitude of the Citation aircraft. Units = Degrees'
    info_c['alt'] = 'Altitude above sea level of Citation aircraft. Units = meters'
    info_c['time'] = 'Time of observation in the Citation aircraft. Units = datetime'
    info_c['temperature'] = 'Temperature observed on the Citation aircraft. Units = Degrees C'
    info_c['lwc'] = 'Liquid water content measured using the King hot wire probe. Units = grams per meter cubed'
    info_c['iwc'] = 'Ice water content estimated from the Nevzorov probe. Units = grams per meter cubed'
    info_c['ice'] = 'Frequency from Rosemount Icing detector. Units = Hz'
    info_c['cdp'] = 'Cloud droplet concentration measured from the CDP. Units = Number per cc'
    info_c['twc'] = 'Nevzorov total water content measured by deep cone. Units = grams per meter'
    info_c['td'] = 'Dewpoint temperature, Units = Degrees Celcius'
    info_c['w'] = 'Vertical velocity, Units = meters per second'
    info_c['P'] = 'static pressure, Units = ?'
    info_c['mix'] = 'mixing ratio, Units = none (i.e. kg/kg)'
    info_c['U'] = 'U componate of the wind, Units = meters per second'
    info_c['V'] = 'V componate of the wind, Units = meters per second'
    
    
    info_r['lat'] = 'Latitude of the center of the radar gate. Units = Degrees'
    info_r['lon'] = 'Longitude of the center of the radar gate. Units = Degrees'
    info_r['alt'] = 'Altitude above sea level of the radar gate. Units = meters'
    info_r['time'] = 'Time of observation at the start of the ray. Units = datetime'
    info_r['Ku'] = 'Ku band measured reflectivity at the gate. Units = dBZ'
    info_r['Ka'] = 'Ka band measured reflectivity at the gate. Units = dBZ'
    info_r['W'] = 'W band measured reflectivity at the gate. Units = dBZ'
    info_r['DFR'] = 'Ku - Ka band measured reflectivity at the gate. Units = dB'
    info_r['DFR2'] = 'Ku - W band measured reflectivity at the gate. Units = dB'
    info_r['DFR3'] = 'Ka - W band measured reflectivity at the gate. Units = dB'
    
    info_m['lat_r'] = 'Latitude of the center of the matched radar gates. Units = Degrees'
    info_m['lon_r'] = 'Longitude of the center of the matched radar gates. Units = Degrees'
    info_m['alt_r'] = 'Altitude above sea level of the matched radar gates. Units = meters'
    info_m['time_r'] = 'Time of the matched observation at the start of the ray. Units = datetime'
    info_m['lat_c'] = 'Latitude of the citation aircraft. Units = Degrees'
    info_m['lon_c'] = 'Longitude of the Citation aircraft. Units = Degrees'
    info_m['alt_c'] = 'Altitude above sea level of Citation aircraft. Units = meters'
    info_m['time_c'] = 'Time of observation in the Citation aircraft. Units = datetime'
    info_m['Ku'] = 'Ku band measured reflectivity matched to Citation location. Units = dBZ'
    info_m['Ka'] = 'Ka band measured reflectivity matched to Citation location. Units = dBZ'
    info_m['W'] = 'W band measured reflectivity matched to Citation location. Units = dBZ'
    info_m['DFR'] = 'Ku - Ka band measured reflectivity matched to Citation location. Units = dB'
    info_m['DFR2'] = 'Ku - W band measured reflectivity matched to Citation location. Units = dB'
    info_m['DFR3'] = 'Ka - W band measured reflectivity matched to Citation location. Units = dB'
    info_m['dist'] = 'Cartesian distance between Citation and "matched" radar gate. This will be a barnes average if query_k is greater than 1. Units = meters'
    info_m['dif_t'] = 'Time difference between the radar gate and the citation observation. Units = Seconds'
    info_m['PSD'] = 'N(D) for the matched points. Units = meteres ^ -4'
    info_m['dD'] = 'Binwidths for the N(D). Units = meters'
    info_m['midpoints'] = 'Bin midpoints for the N(D). Units= meters'
    info_m['rho_BF'] = 'Effective density of the particles using the N(D), a and b from Brown and Francis 1995 and assuming a ellipsoidal fit of 0.6'
    info_m['rho_HY'] = 'Effective density of the particles using the N(D), a and b from Heymsfield et al. 2004 and assuming a ellipsoidal fit of 0.6'
    info_m['rho_NV'] = 'Effective density of the particles using the N(D), mass from Nev TWC, volume of ellip sphere'
    info_m['Dmm_BF'] = 'Two types: Dmm, and Dmm_interp. Interp uses a simple interpolation, while Dmm is the Bin that exceeds 50% of the accumulated mass.Median mass dimension using N(D) and a-b from Brown and Francis 1995'
    info_m['Dmm_HY'] = 'Two types: Dmm, and Dmm_interp. Interp uses a simple interpolation, while Dmm is the Bin that exceeds 50% of the accumulated mass.Median mass dimension using N(D) and a-b from Heymsfield et al. 2004'
    
    Cit['info'] = info_c
    Cit['lat'] = cit_lat
    Cit['lon'] = cit_lon
    Cit['alt'] = cit_alt
    Cit['time'] = cit_time2
    Cit['temperature'] = T_c
    Cit['lwc'] = lwc_c
    Cit['ice'] = ice_c
    Cit['cdp'] = cdp_c
    Cit['twc'] = twc_c
    Cit['iwc'] = iwc_c
    Cit['td'] = td
    Cit['w'] = w
    Cit['P'] = P
    Cit['mix'] = mix
    Cit['U'] = U
    Cit['V'] = V
    
    APR['info'] = info_r
    APR['lat'] = apr_y
    APR['lon'] = apr_x
    APR['alt'] = apr_alt
    APR['Ku'] = apr_ku
    APR['Ka'] = apr_ka
    APR['W'] = apr_w
    APR['DFR'] = apr_ku - apr_ka
    APR['DFR2'] = apr_ku - apr_w
    APR['DFR3'] = apr_ka - apr_w
    APR['time'] = apr_t

    matched['info'] = info_m
    matched['Ku'] = Ku_gate
    matched['Ka'] = Ka_gate
    matched['W'] = W_gate
    matched['DFR'] = DFR_gate
    matched['DFR2'] = DFR2_gate
    matched['DFR3'] = DFR3_gate
    matched['lat_r'] = lat_r
    matched['lon_r'] = lon_r
    matched['alt_r'] = alt_r
    matched['lat_c'] = lat_c
    matched['lon_c'] = lon_c
    matched['alt_c'] = alt_c
    matched['time_r'] = t_r
    matched['time_c'] = t_c
    matched['dist'] = dis_r
    matched['dif_t'] = dif_t
    matched['PSD'] = ND_aver*1e8 #convert to m
    matched['dD'] = dD /1000. #convert to m
    matched['midpoints'] = midpoints / 1000. #convert to m
    matched['rho_BF'] = rho_tot3
    matched['rho_HY'] = rho_tot2
    matched['rho_NV'] = rho_tot4
    matched['Dmm_BF'] = dmm_BF
    matched['Dmm_HY'] = dmm_HY
    matched['iwc_BF'] = iwc_BF
    matched['iwc_HY'] = iwc_HY
    
    if attenuation_correct:
        matched['maxchange'] = maxchange
        matched['lwc_prof'] = apr['lwc_prof']
        matched['altbins_prof']= apr['altbins_prof']
        matched['k_store'] = apr['k_store']
    if attenuation_correct and BB:
        matched['gas_w'] = apr['gas_w']
        matched['gas_ku'] = apr['gas_ku']
        matched['gas_ka'] = apr['gas_ka']
        matched['liquid_w'] = apr['liquid']
        matched['ice_w'] = apr['ice']
    
    if return_indices:
        matched['prind1d'] = prind1d
        matched['APR_dim'] = apr['Ku'].shape
        matched['time'] = apr['timedates']
        matched['APR_lat'] = apr['lat_gate']
        matched['APR_lon'] = apr['lon_gate']
        matched['APR_alt'] = apr['alt_gate']
        matched['APR_Ku'] = apr['Ku']
        matched['APR_Ka'] = apr['Ka']
        matched['APR_W'] = apr['W']
        matched['R'] = R
        matched['R_c'] = R_c
        matched['echo_c'] = echo_c
        matched['echo'] = echo
        matched['R_echo'] = R_echo
        matched['bb_long'] = bb_long
        
    if query_k > 1 and QC:
        matched['IQR_ku'] = IQR_ku
        matched['IQR_ka'] = IQR_ka
        matched['IQR_w'] = IQR_w
        
        matched['n_1'] = n_1
        matched['n_2'] = n_2
        matched['n_3'] = n_3

        matched['IQR_w_w'] = IQR_w2
        matched['IQR_ka_w'] = IQR_ka2
        matched['IQR_ku_w'] = IQR_ku2

    #Not needed currently (RJC May 31 2017)
    #matched['array index'] = ind_r
    #matched['conc_hvps3'] = conc_hvps3
    
    if slimfast:
        matcher['matched'] = matched
        matcher['Cit'] = Cit
    else:
        matcher['Cit'] = Cit
        matcher['APR'] = APR
        matcher['matched'] = matched
        matcher['kdtree'] = kdtree
    
    #Several plots to visualize data
    if plotson:
        fontsize=fontsize
        matched = matcher
        
        if query_k <= 1:
            diftime = matched['matched']['time_r'] - matched['matched']['time_c']
            diftime2 = np.array([])
            for i in np.arange(0,diftime.shape[0]):
                diftime2 = np.append(diftime2,diftime[i].total_seconds())
        else:
            diftime2= matched['matched']['dif_t']
            

        fig1,axes = plt.subplots(1,2,)
        
        #ax1 is the histogram of times
        ax1 = axes[0]
        ax1.hist(diftime2/60.,facecolor='b',edgecolor='k')
        ax1.set_xlabel('$t_{gate} - t_{Cit}, [min]$')
        ax1.set_ylabel('Number of gates')
        ax1.set_title(matched['matched']['time_r'][0])
        #ax2 is the histogram of distances
        ax2 = axes[1]
        distances = matched['matched']['dist']
        ax2.hist(distances,facecolor='r',edgecolor='k')
        ax2.set_xlabel('Distance, $[m]$')
        ax2.set_ylabel('Number of gates')
        ax2.set_title(matched['matched']['time_r'][0])

        plt.tight_layout()

        #Print some quick stats
        print(distances.shape[0],np.nanmean(diftime2)/60.,np.nanmean(distances))
        #
        
        fig = plt.figure()
        #ax3 is the swath plot to show radar and plane location
        ax3 = plt.gca()
        apr = apr3read(apr3filename)
        lat3d = apr['lat_gate']
        lon3d = apr['lon_gate']
        alt3d = apr['alt_gate']
        radar_n = apr['Ku']

        lon_s = np.empty(alt3d.shape[1:])
        lat_s = np.empty(alt3d.shape[1:])
        swath = np.empty(alt3d.shape[1:])
        for i in np.arange(0,alt3d.shape[2]):
            for j in np.arange(0,alt3d.shape[1]):
                ind = np.where(alt3d[:,j,i]/1000. > 3.5)
                ind2 = np.where(alt3d[:,j,i]/1000. < 3.6)
                ind3 = np.intersect1d(ind,ind2)
                ind3= ind3[0]
                l1 = lat3d[ind3,j,i]
                l2 = lon3d[ind3,j,i]
                k1 = radar_n[ind3,j,i]
                lon_s[j,i] = l2
                lat_s[j,i] = l1
                swath[j,i] = k1

        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
                                          {'a': '6378144.0', 'b': '6356759.0',
                                           'lat_0': '47.7998', 'lat_ts': '47.7998','lon_0': '-123.7066', 'proj': 'stere'},
                                          400, 400,
                                          [-70000., -70000.,
                                           70000., 70000.])
        bmap = pr.plot.area_def2basemap(area_def,resolution='l',ax=ax3)
        bmap.drawcoastlines(linewidth=2)
        bmap.drawstates(linewidth=2)
        bmap.drawcountries(linewidth=2)
        parallels = np.arange(-90.,90,4)
        bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
        meridians = np.arange(180.,360.,4)
        bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
        bmap.drawmapboundary(fill_color='aqua')
        bmap.fillcontinents(lake_color='aqua')

        x,y = bmap(lon_s,lat_s)
        swath[np.where(swath < 0)] = np.nan
        pm1 = bmap.pcolormesh(x,y,swath,vmin=0,vmax=40,zorder=11,cmap='seismic')
        cbar1 = plt.colorbar(pm1,label='$Z_m, [dBZ]$')

        x2,y2 = bmap(matched['matched']['lon_c'],matched['matched']['lat_c'])
        pm2 = bmap.scatter(x2,y2,c=diftime2/60.,marker='o',zorder=12,cmap='PuOr',edgecolor=[],vmin=-10,vmax=10)
        cbar2 = plt.colorbar(pm2,label = '$\Delta{t}, [min]$')

        ax3.set_ylabel('Latitude',fontsize=fontsize,labelpad=20)
        ax3.set_xlabel('Longitude',fontsize=fontsize,labelpad=20)

        plt.tight_layout()
        plt.show()
        
        #Plot timeseries of barnes averaged or closest gate.
        plt.figure()
        plt.plot(matched['matched']['time_c'],matched['matched']['Ku'],'b',label='Ku',lw=3)
        plt.plot(matched['matched']['time_c'],matched['matched']['Ka'],'r',label='Ka',lw=3)
        plt.plot(matched['matched']['time_c'],matched['matched']['W'],'g',label='W',lw=3)
        
        #plt.plot(matched['matched']['time_c'],matched['matched']['DFR'],'--b',label='Ku-Ka')
        #plt.plot(matched['matched']['time_c'],matched['matched']['DFR2'],'--r',label='Ku-W')
        #plt.plot(matched['matched']['time_c'],matched['matched']['DFR3'],'--g',label='Ka-W')
        
        plt.xlabel('Time')
        plt.ylabel('Z, [dBZ]')
        plt.legend()
        plt.show()
       
    print('done')
    return matcher

def apr3read(filename):
    
    """
    ===========
    
    This is for reading in apr3 hdf (HDF5 updated 2/21/18) files from OLYMPEX and return them all in one dictionary
    
    ===========
    
    filename = filename of the apr3 file
    """
        
        
    apr = {}
    flag = 0

    ##Radar varibles in hdf file found by hdf.datasets
    radar_freq = 'zhh14' #Ku
    radar_freq2 = 'zhh35' #Ka
    radar_freq3 = 'z95s' #W
    radar_freq4 = 'ldr14' #LDR
    vel_str = 'vel14' #Doppler
    ##



    hdf = h5py.File(filename,"r")

    listofkeys = hdf['lores'].keys()
    alt = hdf['lores']['alt3D'][:]
    lat = hdf['lores']['lat'][:]
    lon = hdf['lores']['lon'][:]
    time = hdf['lores']['scantime'][:]
    surf = hdf['lores']['surface_index'][:]
    isurf =  hdf['lores']['isurf'][:]
    plane =  hdf['lores']['alt_nav'][:]
    radar = hdf['lores'][radar_freq][:]
    radar2 = hdf['lores'][radar_freq2][:]
    radar4 = hdf['lores'][radar_freq4][:]
    vel = hdf['lores']['vel14c'][:]
    lon3d = hdf['lores']['lon3D'][:]
    lat3d = hdf['lores']['lat3D'][:]
    alt3d = hdf['lores']['alt3D'][:]

    #see if there is W band
    if 'z95s' in listofkeys:
        if 'z95n' in listofkeys:
            radar_nadir = hdf['lores']['z95n']
            radar_scanning = hdf['lores']['z95s']
            radar3 = radar_scanning
            ##uncomment if you want high sensativty as nadir scan (WARNING, CALIBRATION)
            #radar3[:,12,:] = radar_nadir[:,12,:]
        else:
            radar3 = hdf['lores']['z95s']
            print('No vv, using hh')
    else:
        radar3 = np.ma.array([])
        flag = 1
        print('No W band')

    ##convert time to datetimes
    time_dates = np.empty(time.shape,dtype=object)
    for i in np.arange(0,time.shape[0]):
        for j in np.arange(0,time.shape[1]):
            tmp = datetime.datetime.utcfromtimestamp(time[i,j])
            time_dates[i,j] = tmp

    #Create a time at each gate (assuming it is the same down each ray, there is a better way to do this)      
    time_gate = np.empty(lat3d.shape,dtype=object)
    for k in np.arange(0,550):
        for i in np.arange(0,time_dates.shape[0]):
            for j in np.arange(0,time_dates.shape[1]):
                time_gate[k,i,j] = time_dates[i,j]       

    #Quality control (masked where invalid)
    radar = np.ma.masked_where(radar <= -99,radar)
    radar2 = np.ma.masked_where(radar2 <= -99,radar2)
    radar3 = np.ma.masked_where(radar3 <= -99,radar3)
    radar4 = np.ma.masked_where(radar4 <= -99,radar4)
    
    #Get rid of nans, the new HDF has builtin
    radar = np.ma.masked_where(np.isnan(radar),radar)
    radar2 = np.ma.masked_where(np.isnan(radar2),radar2)
    radar3 = np.ma.masked_where(np.isnan(radar3),radar3)
    radar4 = np.ma.masked_where(np.isnan(radar4),radar4)


    apr['Ku'] = radar
    apr['Ka'] = radar2
    apr['W'] = radar3
    apr['DFR_1'] = radar - radar2 #Ku - Ka

    if flag == 0:
        apr['DFR_3'] = radar2 - radar3 #Ka - W
        apr['DFR_2'] = radar - radar3 #Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward]'
    else:
        apr['DFR_3'] = np.array([]) #Ka - W
        apr['DFR_2'] = np.array([]) #Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward], Note No W band avail'

    apr['ldr'] = radar4
    apr['vel'] = vel
    apr['lon'] = lon
    apr['lat'] = lat
    apr['alt_gate'] = alt3d
    apr['alt_plane'] = plane
    apr['surface'] = isurf 
    apr['time']= time
    apr['timedates']= time_dates
    apr['time_gate'] = time_gate
    apr['lon_gate'] = lon3d
    apr['lat_gate'] = lat3d

    # fileheader = hdf.select('fileheader')
    roll = hdf['lores']['roll']
    pitch = hdf['lores']['pitch']
    drift = hdf['lores']['drift']

    ngates = alt.shape[0] 

    apr['ngates'] = ngates
    apr['roll'] = roll
    apr['pitch'] = pitch
    apr['drift'] = drift

    _range = np.arange(15,550*30,30)
    _range = np.asarray(_range,float)
    ind = np.where(_range >= plane.mean())
    _range[ind] = np.nan
    apr['range'] = _range

    return apr


def atten_cor2(filename1,fl,percentile,matlab_g,lwc_alt=True):
    
    """
    ========
    
    This is a first order attenuation correction algorithm for Ku,Ka and W band radar.
    
    filename1: string, filename of apr3 file
    filename2: string, filename of citation file
    percentile: threshold for lwc prof
    matlab_g: dict, from matlab calc of gaseous attenuation
   
    ========
    
    
    """
    
    #Read in APR data
    apr = apr3read(filename1)
    
    #Determine altitude bins for profile of LWC
    altbins = np.arange(1000,9000,500)
    altmidpoints = np.arange(1500,9000,500) 
    coldcloud = True
    
    #lwc1 = King probe
    #twc = Nevzorov total water content

    
    lwc1 = fl['lwc1']['data']
    twc = fl['twc']['data']
    
    #Get rid negative values
    lwc1 = np.ma.masked_where(lwc1 <=0,lwc1)
    twc = np.ma.masked_where(lwc1 <=0,twc)
    
    
    
    T = fl['temperature']['data']
    
    if coldcloud:
        lwc1 = np.ma.masked_where(T > -5,lwc1)
        twc = np.ma.masked_where(T > -5,twc)
    
    #Correct for ice response on hot wire probe, Cober et al. 2001 
    if lwc_alt:
        lwc1 = lwc1 - twc *.15
        lwc1 = np.ma.masked_where(lwc1 <=0,lwc1)
        
    alt = fl['altitude']['data']
    
    #Get rid of masked values before doing percentiles
    ind = lwc1.mask
    lwc1 = lwc1[~ind]
    alt = alt[~ind]
    T = T[~ind]
    
    #Create top and bottom for interp (fit gets weird outside this range)
    fit_min = np.min(alt)
    fit_max = np.max(alt)
    
    #Do percentiles for each bin
    q = np.array([percentile])
    Q = np.zeros([altbins.shape[0]-1])
    for i in np.arange(1,altbins.shape[0]):
        bottom = altbins[i-1]
        top = altbins[i]

        ind1 = np.where(alt>=bottom)
        ind2 = np.where(alt<top)
        ind3 = np.intersect1d(ind1,ind2)
        if len(ind3) < 1:
            Q[i-1] = np.nan
        else:
            Q[i-1] = np.nanpercentile(lwc1[ind3],q)
    #get rid of any nans    
    ind = np.isnan(Q)
    Q_temp = Q[~ind]
    altmidpoints_temp = altmidpoints[~ind]
    
    #lwc profile func
    lwc_func = interp1d(altmidpoints_temp,Q_temp,kind='cubic',bounds_error=False)
    #
    
    
    #W_lwc_coeff_func
    t_ks = np.array([-20,-10,0])
    ks = np.array([5.41,5.15,4.82])
    k_coeff = np.polyfit(t_ks,ks,deg=1)
    k_func = np.poly1d(k_coeff)
    
    #Ka_lwc_coeff_func
    t_ks2 = np.array([-20,-10,0])
    ks2 = np.array([1.77,1.36,1.05])
    k_coeff2 = np.polyfit(t_ks2,ks2,deg=1)
    k_func2 = np.poly1d(k_coeff2)
    
    #Ku_lwc_coeff_func
    t_ks3 = np.array([-20,-10,0])
    ks3 = np.array([0.45,0.32,0.24])
    k_coeff3 = np.polyfit(t_ks3,ks3,deg=1)
    k_func3 = np.poly1d(k_coeff3)
    
    #temperature function
    t_coef = np.polyfit(alt,T,deg=1)
    t_func = np.poly1d(t_coef)
    
    #Functions for O2 and H2O atten
    alt = np.squeeze(matlab_g['alt'])
    L =np.squeeze(matlab_g['L'])
    L3 = np.squeeze(matlab_g['L3'])
    L5 = np.squeeze(matlab_g['L5'])
    
    k_func4 = interp1d(alt,L,kind='cubic',bounds_error=False)
    k_func5 = interp1d(alt,L3,kind='cubic',bounds_error=False)
    k_func6 = interp1d(alt,L5,kind='cubic',bounds_error=False)
    
    #function to correct for ice scattering (kulie et al. 2014 fig 7)
    k = pd.read_csv('Kulie_specific_attenuation.csv')
    x = k['x'].values
    y = k['y'].values
    x_min = x.min()
    x_max = x.max()
    #fit function so we can crank out any value of Ku
    k_coef7 = np.polyfit(x,y,deg=1)
    k_func7 = np.poly1d(k_coef7)
    
    #Make new data arrays
    w_new_new = np.ma.zeros(apr['W'].shape)
    ka_new_new = np.ma.zeros(apr['Ka'].shape)
    ku_new_new = np.ma.zeros(apr['Ku'].shape)
    
    k_store2 = np.ma.zeros(apr['W'].shape)
    
    #Main loop for correcting the profile
    for j in np.arange(0,apr['alt_gate'].shape[1]):

        alt = np.squeeze(apr['alt_gate'][:,j,:])
        w = np.squeeze(apr['W'][:,j,:])
        ka = np.squeeze(apr['Ka'][:,j,:])
        ku = np.squeeze(apr['Ku'][:,j,:])
        
        #need ku in linear units for ice scatter correction
        ku_lin = 10**(ku/10.)
        ind = np.ma.where(ku_lin > x_max)
        ku_lin[ind] = x_max

        w_new = np.ma.zeros(w.shape)
        ka_new = np.ma.zeros(ka.shape)
        ku_new = np.ma.zeros(ku.shape)
        k_store = np.ma.zeros(w.shape)
        for i in np.arange(0,alt.shape[1]):

            a1 = alt[:,i]
            w1 = w[:,i]
            ka1 = ka[:,i]
            ku1 = ku[:,i]
            ku_lin1 = ku_lin[:,i]
            
            #Create a function to get T from alt
            ts = t_func(a1)
            
            #Get the right coeffs for the right alts (based on T)
            ks = k_func(ts)
            ks2 = k_func2(ts)
            ks3 = k_func3(ts)
            #
            
            #Get the right attenuation from atmospheric gases
            ks4 = k_func4(a1)
            ks5 = k_func5(a1)
            ks6 = k_func6(a1)
            #
            
            #get db/m caused by ice from ku following Kulie et al 2014
            ks7 = k_func7(ku_lin1)
            #zero where ref is masked...
            ks7[ku_lin1.mask] = 0.
            #
            
            #Get lwc prof
            ls = lwc_func(a1)
            #
            
            coeff = ls*ks
            coeff2 = ls*ks2
            coeff3 = ls*ks3
            coeff4 = ks4
            coeff5 = ks5
            coeff6 = ks6
            coeff7 = ks7
            coeff[a1 <= fit_min+500] = 0
            coeff[a1 >= fit_max-500] = 0
            coeff2[a1 <= fit_min+500] = 0
            coeff2[a1 >= fit_max-500] = 0
            coeff3[a1 <= fit_min+500] = 0
            coeff3[a1 >= fit_max-500] = 0
            
#This is an error, was only applying the gaseous attenuation to -5 deg C. Now it goes to the surface (12/13/17)             
#            coeff4[a1 <= fit_min+500] = 0
#            coeff4[a1 >= fit_max-500] = 0
#            coeff5[a1 <= fit_min+500] = 0
#            coeff5[a1 >= fit_max-500] = 0
#            coeff6[a1 <= fit_min+500] = 0
#            coeff6[a1 >= fit_max-500] = 0


            #Convert to dB/gate
            coeff = (coeff/1000.)*30.
            coeff2 = (coeff2/1000.)*30.
            coeff3 = (coeff3/1000.)*30.
            coeff4 = (coeff4/1000.)*30.
            coeff5 = (coeff5/1000.)*30.
            coeff6 = (coeff6/1000.)*30.
            coeff7 = coeff7 * 30.
            #
            
            #get rid of nans so cumsum works right, nans are inserted if radar is masked
            ind = np.isnan(coeff)
            coeff[ind] = 0.
            ind = np.isnan(coeff2)
            coeff2[ind] = 0.
            ind = np.isnan(coeff3)
            coeff3[ind] = 0.
            ind = np.isnan(coeff4)
            coeff4[ind] = 0.
            ind = np.isnan(coeff5)
            coeff5[ind] = 0.
            ind = np.isnan(coeff6)
            coeff6[ind] = 0.
            ind = np.isnan(coeff7)
            coeff7[ind] = 0.
            
            #path integrate
            k = np.cumsum(coeff)*2
            k2 = np.cumsum(coeff2)*2
            k3 = np.cumsum(coeff3)*2
            k4 = np.cumsum(coeff4)*2
            k5 = np.cumsum(coeff5)*2
            k6 = np.cumsum(coeff6)*2
            k7 = np.cumsum(coeff7)*2
            #

            #correct
            w1 =  w1+k+k4+k7
            
            #uncomment if you wish to correct Ka and Ku
            #ka1 = ka1+k2+k5
            #ku1 = ku1+k3+k6
            
            #correcting just for gases 
            ka1 = ka1+k5
            ku1 = ku1+k6

            w_new[:,i] = w1
            ka_new[:,i] = ka1
            ku_new[:,i] = ku1
            #
            k_store[:,i] = k + k4 + k7
            
        w_new_new[:,j,:] = w_new
        ka_new_new[:,j,:] = ka_new
        ku_new_new[:,j,:] = ku_new
        k_store2[:,j,:] = k_store
    
    #mask the attenuation field to where the ref. field is masked (i.e. BB algo) (12/13/17)
    k_store2 = np.ma.masked_where(w_new_new.mask,k_store2)
    
    #Find max correction values for table 
    wmax = np.ma.max(w_new_new - apr['W'])
    kamax = np.ma.max(ka_new_new - apr['Ka'])
    kumax = np.ma.max(ku_new_new - apr['Ku'])
    maxchange = np.array([wmax,kamax,kumax])
    
    #Pack data back into dict
    data_corrected = {}
    data_corrected['Ku'] = ku_new_new
    data_corrected['Ka'] = ka_new_new
    data_corrected['W'] = w_new_new
    data_corrected['Ku_uc'] = apr['Ku']
    data_corrected['Ka_uc'] =apr['Ka']
    data_corrected['W_uc'] = apr['W']
    data_corrected['lwc_prof'] = Q_temp
    data_corrected['altbins_prof'] = altmidpoints_temp
    data_corrected['timedates'] = apr['timedates']
    data_corrected['alt_gate'] =  apr['alt_gate']
    data_corrected['lat'] = apr['lat']
    data_corrected['lon'] = apr['lon']
    data_corrected['lat_gate'] = apr['lat_gate']
    data_corrected['lon_gate'] = apr['lon_gate']
    data_corrected['surface'] = apr['surface']
    data_corrected['time_gate'] = apr['time_gate']
    data_corrected['maxchange'] = maxchange
    data_corrected['k_store'] = k_store2
    data_corrected['roll'] = apr['roll']
    
    return data_corrected

def atten_cor3(apr,fl,percentile,matlab_g,lwc_alt=True):
    
    """
    ========
    
    This is a first order attenuation correction algorithm for Ku,Ka and W band radar.
    
    filename1: string, filename of apr3 file
    filename2: string, filename of citation file
    percentile: threshold for lwc prof
    matlab_g: dict, from matlab calc of gaseous attenuation
   
    ========
    
    
    """
    
    #Determine altitude bins for profile of LWC
    altbins = np.arange(1000,9000,500)
    altmidpoints = np.arange(1500,9000,500) 
    coldcloud = True
    
    #lwc1 = King probe
    #twc = Nevzorov total water content

    
    lwc1 = fl['lwc1']['data']
    twc = fl['twc']['data']
    
    #Get rid negative values
    lwc1 = np.ma.masked_where(lwc1 <=0,lwc1)
    twc = np.ma.masked_where(lwc1 <=0,twc)
    
    T = fl['temperature']['data']
    
    if coldcloud:
        lwc1 = np.ma.masked_where(T > -5,lwc1)
        twc = np.ma.masked_where(T > -5,twc)
    
    #Correct for ice response on hot wire probe, Cober et al. 2001 
    if lwc_alt:
        lwc1 = lwc1 - twc *.15
        lwc1 = np.ma.masked_where(lwc1 <=0,lwc1)
        
    alt = fl['altitude']['data']
    
    #Get rid of masked values before doing percentiles
    ind = lwc1.mask
    lwc1 = lwc1[~ind]
    alt = alt[~ind]
    T = T[~ind]
    
    #Create top and bottom for interp (fit gets weird outside this range)
    fit_min = np.min(alt)
    fit_max = np.max(alt)
    
    #Do percentiles for each bin
    q = np.array([percentile])
    Q = np.zeros([altbins.shape[0]-1])
    for i in np.arange(1,altbins.shape[0]):
        bottom = altbins[i-1]
        top = altbins[i]

        ind1 = np.where(alt>=bottom)
        ind2 = np.where(alt<top)
        ind3 = np.intersect1d(ind1,ind2)
        if len(ind3) < 1:
            Q[i-1] = np.nan
        else:
            Q[i-1] = np.nanpercentile(lwc1[ind3],q)
    #get rid of any nans    
    ind = np.isnan(Q)
    Q_temp = Q[~ind]
    altmidpoints_temp = altmidpoints[~ind]
    
    #lwc profile func
    lwc_func = interp1d(altmidpoints_temp,Q_temp,kind='cubic',bounds_error=False)
    #
    
    
    #W_lwc_coeff_func
    t_ks = np.array([-20,-10,0])
    ks = np.array([5.41,5.15,4.82])
    k_coeff = np.polyfit(t_ks,ks,deg=1)
    k_func = np.poly1d(k_coeff)
    
    #Ka_lwc_coeff_func
    t_ks2 = np.array([-20,-10,0])
    ks2 = np.array([1.77,1.36,1.05])
    k_coeff2 = np.polyfit(t_ks2,ks2,deg=1)
    k_func2 = np.poly1d(k_coeff2)
    
    #Ku_lwc_coeff_func
    t_ks3 = np.array([-20,-10,0])
    ks3 = np.array([0.45,0.32,0.24])
    k_coeff3 = np.polyfit(t_ks3,ks3,deg=1)
    k_func3 = np.poly1d(k_coeff3)
    
    #temperature function
    t_coef = np.polyfit(alt,T,deg=1)
    t_func = np.poly1d(t_coef)
    
    #Functions for O2 and H2O atten
    alt = np.squeeze(matlab_g['alt'])
    L =np.squeeze(matlab_g['L'])
    L3 = np.squeeze(matlab_g['L3'])
    L5 = np.squeeze(matlab_g['L5'])
    
    k_func4 = interp1d(alt,L,kind='cubic',bounds_error=False)
    k_func5 = interp1d(alt,L3,kind='cubic',bounds_error=False)
    k_func6 = interp1d(alt,L5,kind='cubic',bounds_error=False)
    
    #function to correct for ice scattering (kulie et al. 2014 fig 7)
    k = pd.read_csv('Kulie_specific_attenuation.csv')
    x = k['x'].values
    y = k['y'].values
    x_min = x.min()
    x_max = x.max()
    #fit function so we can crank out any value of Ku
    k_coef7 = np.polyfit(x,y,deg=1)
    k_func7 = np.poly1d(k_coef7)
    
    #Make new data arrays
    w_new_new = np.ma.zeros(apr['W'].shape)
    ka_new_new = np.ma.zeros(apr['Ka'].shape)
    ku_new_new = np.ma.zeros(apr['Ku'].shape)
    
    k_store2 = np.ma.zeros(apr['W'].shape)
    gas2_w =  np.ma.zeros(apr['W'].shape)
    gas2_ku =  np.ma.zeros(apr['W'].shape)
    gas2_ka =  np.ma.zeros(apr['W'].shape)
    liquid2 =  np.ma.zeros(apr['W'].shape)
    ice2 =  np.ma.zeros(apr['W'].shape)
    
    #Main loop for correcting the profile
    for j in np.arange(0,apr['alt_gate'].shape[1]):

        alt = np.squeeze(apr['alt_gate'][:,j,:])
        w = np.squeeze(apr['W'][:,j,:])
        ka = np.squeeze(apr['Ka'][:,j,:])
        ku = np.squeeze(apr['Ku'][:,j,:])
        
        #need ku in linear units for ice scatter correction
        ku_lin = 10**(ku/10.)
        ind = np.ma.where(ku_lin > x_max)
        ku_lin[ind] = x_max
        
        w_new = np.ma.zeros(w.shape)
        ka_new = np.ma.zeros(ka.shape)
        ku_new = np.ma.zeros(ku.shape)
        k_store = np.ma.zeros(w.shape)
        gas_w = np.ma.zeros(w.shape)
        gas_ku = np.ma.zeros(w.shape)
        gas_ka = np.ma.zeros(w.shape)
        ice = np.ma.zeros(w.shape)
        liquid = np.ma.zeros(w.shape)
        for i in np.arange(0,alt.shape[1]):

            a1 = alt[:,i]
            w1 = w[:,i]
            ka1 = ka[:,i]
            ku1 = ku[:,i]
            ku_lin1 = ku_lin[:,i]
            
            #Create a function to get T from alt
            ts = t_func(a1)
            
            #Get the right coeffs for the right alts (based on T)
            ks = k_func(ts)
            ks2 = k_func2(ts)
            ks3 = k_func3(ts)
            #
            
            #Get the right attenuation from atmospheric gases
            ks4 = k_func4(a1)
            ks5 = k_func5(a1)
            ks6 = k_func6(a1)
            #
            
            #get db/m caused by ice from ku following Kulie et al 2014
            ks7 = k_func7(ku_lin1)
            #zero where ref is masked...
            ks7[ku_lin1.mask] = 0.
            #
            
            #Get lwc prof
            ls = lwc_func(a1)
            #
            
            coeff = ls*ks
            coeff2 = ls*ks2
            coeff3 = ls*ks3
            coeff4 = ks4
            coeff5 = ks5
            coeff6 = ks6
            coeff7 = ks7
            coeff[a1 <= fit_min+500] = 0
            coeff[a1 >= fit_max-500] = 0
            coeff2[a1 <= fit_min+500] = 0
            coeff2[a1 >= fit_max-500] = 0
            coeff3[a1 <= fit_min+500] = 0
            coeff3[a1 >= fit_max-500] = 0
#This is an error, was only applying the gaseous attenuation to -5 deg C. Now it goes to the surface (12/13/17) 
#             coeff4[a1 <= fit_min+500] = 0
#             coeff4[a1 >= fit_max-500] = 0
#             coeff5[a1 <= fit_min+500] = 0
#             coeff5[a1 >= fit_max-500] = 0
#             coeff6[a1 <= fit_min+500] = 0
#             coeff6[a1 >= fit_max-500] = 0

            #Convert to dB/gate
            coeff = (coeff/1000.)*30.
            coeff2 = (coeff2/1000.)*30.
            coeff3 = (coeff3/1000.)*30.
            coeff4 = (coeff4/1000.)*30.
            coeff5 = (coeff5/1000.)*30.
            coeff6 = (coeff6/1000.)*30.
            coeff7 = coeff7 * 30.
            #
            
            #get rid of nans so cumsum works right, nans are inserted if radar is masked
            ind = np.isnan(coeff)
            coeff[ind] = 0.
            ind = np.isnan(coeff2)
            coeff2[ind] = 0.
            ind = np.isnan(coeff3)
            coeff3[ind] = 0.
            ind = np.isnan(coeff4)
            coeff4[ind] = 0.
            ind = np.isnan(coeff5)
            coeff5[ind] = 0.
            ind = np.isnan(coeff6)
            coeff6[ind] = 0.
            ind = np.isnan(coeff7)
            coeff7[ind] = 0.
            
            #path integrate
            k = np.cumsum(coeff)*2
            k2 = np.cumsum(coeff2)*2
            k3 = np.cumsum(coeff3)*2
            k4 = np.cumsum(coeff4)*2
            k5 = np.cumsum(coeff5)*2
            k6 = np.cumsum(coeff6)*2
            k7 = np.cumsum(coeff7)*2
            #

            #correct
            w1 =  w1+k+k4+k7
            
            #uncomment if you want to correct 
            #ka1 = ka1+k2+k5
            #ku1 = ku1+k3+k6
            
            #correcting just for gases 
            ka1 = ka1+k5
            ku1 = ku1+k6

            w_new[:,i] = w1
            ka_new[:,i] = ka1
            ku_new[:,i] = ku1
            #
            k_store[:,i] = k + k4 + k7
            gas_w[:,i] = k4
            gas_ku[:,i] = k6
            gas_ka[:,i] = k5
            liquid[:,i] = k
            ice[:,i] = k7
            
        w_new_new[:,j,:] = w_new
        ka_new_new[:,j,:] = ka_new
        ku_new_new[:,j,:] = ku_new
        k_store2[:,j,:] = k_store
        gas2_w[:,j,:] = gas_w
        gas2_ku[:,j,:] = gas_ku
        gas2_ka[:,j,:] = gas_ka
        liquid2[:,j,:] = liquid
        ice2[:,j,:] = ice
        
    #mask attenuation to where w is masked    
    gas2_w = np.ma.masked_where(w_new_new.mask,gas2_w)  
    gas2_ka = np.ma.masked_where(ka_new_new.mask,gas2_ka)
    gas2_ku = np.ma.masked_where(ku_new_new.mask,gas2_ku)
    
    #mask the attenuation field to where the ref. field is masked (i.e. BB algo) (12/13/17)
    k_store2 = np.ma.masked_where(w_new_new.mask,k_store2)
    
    #Find max correction values for table 
    wmax = np.ma.max(w_new_new - apr['W'])
    kamax = np.ma.max(ka_new_new - apr['Ka'])
    kumax = np.ma.max(ku_new_new - apr['Ku'])
    maxchange = np.array([wmax,kamax,kumax])
    
    #Pack data back into dict
    data_corrected = {}
    data_corrected['Ku'] = ku_new_new
    data_corrected['Ka'] = ka_new_new
    data_corrected['W'] = w_new_new
    data_corrected['Ku_uc'] = apr['Ku']
    data_corrected['Ka_uc'] =apr['Ka']
    data_corrected['W_uc'] = apr['W']
    data_corrected['lwc_prof'] = Q_temp
    data_corrected['altbins_prof'] = altmidpoints_temp
    data_corrected['timedates'] = apr['timedates']
    data_corrected['alt_gate'] =  apr['alt_gate']
    data_corrected['lat'] = apr['lat']
    data_corrected['lon'] = apr['lon']
    data_corrected['lat_gate'] = apr['lat_gate']
    data_corrected['lon_gate'] = apr['lon_gate']
    data_corrected['surface'] = apr['surface']
    data_corrected['time_gate'] = apr['time_gate']
    data_corrected['maxchange'] = maxchange
    data_corrected['k_store'] = k_store2
    data_corrected['liquid'] = liquid2
    data_corrected['ice'] = ice2
    data_corrected['gas_w'] = gas2_w
    data_corrected['gas_ku'] = gas2_ku
    data_corrected['gas_ka'] = gas2_ka
    data_corrected['roll'] = apr['roll']
    
    return data_corrected

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def BB_alt(aprdata,bbguess=1000):
    data = aprdata
    altbin = np.arange(bbguess-500,10001,100)
    t = np.ravel(10**(data['Ku'][:,12,:]/10.))
    alt = np.ravel(data['alt_gate'][:,12,:])
    Q = np.zeros([5,len(altbin)-1])
    for i in np.arange(0,len(altbin)-1):
        ind = np.where(alt >= altbin[i])
        ind2 = np.where(alt < altbin[i+1])
        ind3 = np.intersect1d(ind,ind2)
        t1 = t[ind3]
        ind = t1.mask
        t1 = t1[~ind]
        if len(t1) < 1:
            Q[:,i] = np.nan*np.zeros([5])
            continue
        Q[:,i] = np.nanpercentile(t1,[10,25,50,75,90])

    mids = np.arange(bbguess-450,10001,100)
    BB = np.where(np.diff(Q[2,:]) <= np.nanmin(np.diff(Q[2,:])))
    
    ku = data['Ku']
    ka = data['Ka']
    w = data['W']
    alt = data['alt_gate']
    
    adj = 250
    ku = np.ma.masked_where(alt <= mids[1:][BB]+adj,ku)
    ka = np.ma.masked_where(alt <= mids[1:][BB]+adj,ka)
    w = np.ma.masked_where(alt <= mids[1:][BB]+adj,w)
    
    
    for j in np.arange(0,alt.shape[1]-1):
        #Look for cocktail spikes,and fix them
        ku_ray = ku[:,j,:]
        alt_ray = alt[:,j,:]
        bump = adj
        for i in np.arange(0,alt_ray.shape[1]):
            ku_ray2 = ku_ray[:,i]
            alt_ray2 = alt_ray[:,i]
            alt_search = mids[1:][BB]+bump
            ind = find_nearest(alt_ray2,alt_search)
            while np.ma.is_masked(ku_ray2[ind]):
                ind = ind - 1
            while ku_ray2[ind] >= 28:
                ku_ray2[ind] = np.ma.masked
                ku[ind,j,i] = np.ma.masked
                ka[ind,j,i]= np.ma.masked
                w[ind,j,i] = np.ma.masked
                ind = ind-1
            ind = np.where(ku_ray2 == np.ma.max(ku_ray2))
            ind2 = np.where(ku_ray2 > 45)
            ind = np.intersect1d(ind,ind2)
            if len(ind) < 1:
                continue
            elif len(ind) > 1:
                ind = ind[0]

            ku[ind:549,j,i] = np.ma.masked
            ka[ind:549,j,i]= np.ma.masked
            w[ind:549,j,i] = np.ma.masked
            ind = ind - 1

            while ku_ray2[ind] >= 28:
                ku_ray2[ind] = np.ma.masked
                ku[ind,j,i] = np.ma.masked
                ka[ind,j,i]= np.ma.masked
                w[ind,j,i] = np.ma.masked
                ind = ind-1
    
    data['Ku'] = ku
    data['Ka'] = ka
    data['W'] = w
    print('BB found at:',mids[1:][BB]+bump)
    return data


def cloudtopmask(apr3data,sigma=1):
    ku_temp = apr3data['Ku']
    ku_temp[ku_temp.mask] = -99.99
    #Maskes Ku cloudtop noise
    for i in np.arange(0,ku_temp.shape[1]-1):
        temp = ku_temp[:,i,:]
        a = scipy.ndimage.filters.gaussian_filter(temp,sigma)
        a = np.ma.masked_where(a<10,a)
        temp = np.ma.masked_where(a.mask,temp)
        ku_temp[:,i,:] = temp
        
    ku_temp = np.ma.masked_where(ku_temp < 10,ku_temp)
    #new data has some weird data near plane. Delete manually
    ku_temp = np.ma.masked_where(apr3data['alt_gate'] > 10000, ku_temp)
    apr3data['Ku'] = ku_temp
    
    return apr3data


def PSD_load(psd_filename_2DS,psd_filename_HVPS,day=1,month=12):
    
    """
    
    This is a function to load and combine the PSD into 1 matrix with dimensions (time,bin)
    
    Units are ND: cm**-4, bindwidth: cm, midpoints: mm
    
    """
    ##Load in UIOPS
    #Select bins for each probe
    
    #2DS
    index1 = 4
    index2 = 18
    #
    #HVPS
    index3 = 5
    index4 = 28
    #
    
    #2DS
    data = netCDF4.Dataset(psd_filename_2DS,'r')
    time1 = data['time'][:]
    ND1 = data['conc_minR'][:,index1:index2]
    midpoints1 = data['bin_mid'][index1:index2]
    bin_width1 = data['bin_dD'][index1:index2]

    #HVPS
    data2 = netCDF4.Dataset(psd_filename_HVPS,'r')
    time2 = data2['time'][:]
    ND2 = data2['conc_minR'][:,index3:index4]
    midpoints2 = data2['bin_mid'][index3:index4]
    bin_width2 = data2['bin_dD'][index3:index4]
    
    #Combine the 2 probes.
    ND = np.zeros([ND1.shape[0],ND1.shape[1]+ND2.shape[1]])
    ND[:,:ND1.shape[1]] = ND1
    ND[:,ND1.shape[1]:] = ND2
    
    binwidth = np.append(bin_width1,bin_width2)
    midpoints = np.append(midpoints1,midpoints2)

    #Use 2ds time (should be the same)
    time = time1
    dtime = np.array([],dtype=object)
    flag1 = 0 
    for i in np.arange(0,time.shape[0]):
        
        if len(str(int(time[i]))) < 6:
            hour = int(str(int(time[i]))[0:1])
            mins = int(str(int(time[i]))[1:3])
            sec = int(str(int(time[i]))[3:5])
        else:
            hour = int(str(int(time[i]))[0:2])
            mins = int(str(int(time[i]))[2:4])
            sec = int(str(int(time[i]))[4:6])
        
        if sec >= 60:
            sec = sec - 60 
            mins = mins + 1
        if mins >= 60:
            mins = mins - 60
            hour = hour + 1  
        if hour >= 24:
            hour = hour - 24
            if flag1 == 0:
                day = day + 1
            flag1 = 1
        
        try:
            temp = datetime.datetime(2015,month,day,hour,mins,sec)
        except:
            print('error in dtime maker')
            print(month,day,hour,mins,sec)
            temp = datetime.datetime(2015,month,day,hour,mins,sec)
            
        dtime = np.append(dtime,temp)
        
    return dtime,ND,binwidth,midpoints

def rho_e(midpoints,binwidth,ND,MD,aspect,mass,twc,return_ice=False):
    
    """
    ===========
    
    Calculate the density of the ice-sphere/spheriod 
    
    
    return_ice = bool, returns iwc with the rho 
    midpoints: cm
    binwidth: cm   
    ND: Number of particles per bin, cm^-4
    MD: Habit dependent MD, see Jackson et al. 2012, g/cm^4 (Habit classification is junk in warm [T > -15])
    aspect: 1,2 or 3.
        1: Spherical volume
        2: 0.6 Ellipsoid 
        3: 0.7 Ellipsoid
    mass: 1,2 or 3
        1: habit
        2: Heymsfield et al. 2004
        3: Brown and Francis 1995
        4: Nev

    ===========
    """
    flag1 = False
    ##Determine Volume function based on input 
    if aspect == 1:
        def volume(D,N,dD):
            vol = vol_1(D,N,dD)
            return vol 
    elif aspect == 2:
        def volume(D,N,dD):
            vol = vol_2(D,N,dD)
            return vol 
    elif aspect == 3:
        def volume(D,N,dD):
            vol = vol_3(D,N,dD)
            return vol
    ##
    
    ##Determine Mass function following m=aD^b and based on input
    if  mass == 1:
        def Mass(D,N,M,dD):
            m = M*dD #g/cm^3
            return m 
    elif mass == 2:
        def Mass(D,N,M,dD):
            #Heymsfield 2004
            a = 0.0061
            b = 2.05
            m = N*dD*a*D**b #g/cm^3
            return m 
    elif mass == 3:
        def Mass(D,N,M,dD):
            #BF 1995 (Hogan adaption)
            a = 0.0121
            b = 1.9
            D = D / 100. #convert cm to m
            m = N*dD*1e3*(a *(D)**b); #g/cm^3
            return m
    elif mass ==4:
        flag1 = True
    ##
    
    rho_array = np.array([],dtype=float)
    ice_array = np.array([],dtype=float)
    for j in np.arange(0,ND.shape[0]):
        NumD = ND[j,:]
        MasD = np.zeros(NumD.shape)
        rho_tot = 0.
        bucket=0.
        bucket2 =0.
        for i in np.arange(3,midpoints.shape[0]):
            if flag1:
                iwc = twc[j]/1e6 #convert g/m^3 to g/cm^3
                bucket = iwc
                vol = volume(midpoints[i],NumD[i],binwidth[i]) #cm^3/cm^3
                bucket2 = np.ma.sum([bucket2,vol])
            else:
                if np.ma.is_masked(NumD[i]):
                    continue
                else:
                    iwc = Mass(midpoints[i],NumD[i],MasD[i],binwidth[i]) # g/cm^3
                    bucket = np.ma.sum([bucket,iwc])
        
                    vol = volume(midpoints[i],NumD[i],binwidth[i]) #cm^3/cm^3
                    bucket2 = np.ma.sum([bucket2,vol])
                
        if bucket==0. and bucket2 == 0.:
            den_b = np.nan
        else:
            den_b = bucket/bucket2 #bulk density of particles (g/cm^3)
        
        #adjust den in the event it exceeds the maxium density of solid ice
        if den_b >= 0.9167:
            den_b = 0.9167
            
        rho_array = np.ma.append(rho_array,den_b)
        ice_array = np.ma.append(ice_array,bucket)
        
    if return_ice:
        return rho_array,ice_array*100**3
    else:
        return rho_array


def vol_1(D,N,dD):
    """
    
    =====
    Volume of ice using sphere 
    =====
    
    """
    vol = (np.pi/6.) * D**3 *N*dD
    
    return vol

def vol_2(D,N,dD):
    
    """
    
    =====
    Volume of ice using ellipsoid with alpha=0.6 
    =====
    
    """
    
    vol = 0.6 * (np.pi/6.) * D**3 *N*dD
    
    return vol

def vol_3(D,N,dD):
    
    """
    
    =====
    Volume of ice using ellipsoid with alpha=0.7 
    =====
    
    """
    
    vol = 0.7 * (np.pi/6) * D**3 *N*dD
    
    return vol

def Dmm(ND,midpoints,binwidth,massflag):
    
    """
    =======
    
    Calculate median mass diameter using BF95 with Hogan 2012 conversion
    flag1: use which m-D? 0: BF95, 1: HY04
    =======
    """
    if massflag == 0:
        a = 0.0121; #Kg/m^-b
        b = 1.9;
    elif massflag == 1:
        #do conversions to get same unit as BF95
        a = 0.0061 #g/cm^-b
        b = 2.05
        a = a / 1000. 
        midpoints = midpoints*100 #convert from m to cm 
    
    #Reset Dmm (Will iterate across flights eventually)
    Dmm = np.ma.array([])
    Dmm_interp = np.ma.array([])
    row = ND.shape
    row = row[0]
    for j in np.arange(0,row):
    #reset Total mass cuml. 
        TMC = np.ma.array([])
        ND_row = np.ma.array(ND[j,:])
        
    #logicals to test if ND is all zeros/Nans before entering second loop.
        temp_ND = np.ma.nonzero(ND_row) # test to see if TMC is all zeros
        shap_ND = np.asarray(temp_ND)
#         temp_isnan_ND = np.isnan(ND_row) #test if TMC is all nans 
#         yaynay = np.all(temp_isnan_ND) #bool
        if shap_ND[0].shape[0] < 1:
            Dmm = np.append(Dmm,np.ma.masked)
            Dmm_interp = np.append(Dmm_interp,np.ma.masked)            
            continue
    #loop across bins/cols
        for i in np.arange(3,len(midpoints)):
            NDi = ND_row[i]
            Ni = np.ma.multiply(NDi,binwidth[i])
            Mi = np.ma.multiply(Ni,(a*midpoints[i]**b))
            TMC = np.ma.append(TMC,Mi)
        TMC_tots = np.ma.sum(TMC)
        Dmm_mass = 0.5*TMC_tots
        y = np.ma.array([])
        bootstrap = 0
        z = -1
        flag1 = 0
        while bootstrap <= Dmm_mass :
            if z >= len(midpoints)-1:
                print('itmax met',j)
                Dmm= np.ma.append(Dmm,np.ma.masked)
                Dmm_interp = np.ma.append(Dmm_interp,np.ma.masked)                
                flag1 = 1
                break
            z = z + 1
            bootstrap = np.ma.sum(TMC[0:z])
            y = np.ma.append(y,bootstrap)
        if flag1 == 1:
            flag1 = 0
            continue
    #By convention in the group choose Dmm to be the bin midpoint when acuum > 0.5*TMC
        Dmm = np.ma.append(Dmm,midpoints[z-1])
    #Quick weighting between two closes bins 
        under = np.ma.sum(TMC[0:z-1])
        over = np.ma.sum(TMC[0:z])
        massbtw = over - under
    #		hundredperadj = massbtw
        dis1 = np.ma.abs(Dmm_mass-under) 
        dis2 = np.ma.abs(Dmm_mass-over) 
        per1 = 1. - dis1/massbtw
        per2 = 1. - dis2/massbtw
        adjmm = per1 * midpoints[z-1] + per2*midpoints[z]
        disper1 = np.ma.abs(Dmm_mass-under) / np.ma.sum(TMC)
        disper2 = np.ma.abs(Dmm_mass-over) / np.ma.sum(TMC)
        Dmm_interp = np.ma.append(Dmm_interp,adjmm)
    Dmms = {}
    if massflag == 0:
        Dmms['Dmm'] = Dmm
        Dmms['Dmm_interp'] = Dmm_interp
    elif massflag == 1:
        Dmms['Dmm'] = Dmm/100.
        Dmms['Dmm_interp'] = Dmm_interp/100.
        
    
    return Dmms

def precip_echo_filt(ku):
    """
    
    This function provides a filter for APR3 data to determine if there exists a precip echo in the column. Only preforms it 
    on one scan at a time. Could easily go up to all scans.
    
    """
    precip_yn = np.zeros(ku.shape[1])
    for j in np.arange(0,ku.shape[1]):
        flag1 = 0 
        c1 = -1
        i = -1
        start_flag = 0
        while flag1 == 0:
            i = i + 1

            if c1 >= 3:
                precip_yn[j] = 1
                break
            if i == 550: 
                precip_yn[j] = 0
                break

            t = ku[i,j]

            if start_flag ==0:
                if np.ma.is_masked(t):
                    continue
                else:
                    start_flag = 1
                    c1 = c1 + 1
            else:
                if np.ma.is_masked(t):
                    start_flag = 0
                    c1 = -1
                    continue
                else:
                    c1 = c1 + 1
    return precip_yn

def find_echo(ku_filt,alt_filt,n=10,thresh=0.5):
    
    """
    
    This function finds the echo top using the filtered Ku from the precip_echo_filt function
    
    """
    ku_mean = np.ma.zeros(ku_filt.shape)
    n_mask =  np.ma.zeros(ku_filt.shape)
    for i in np.arange(0,ku_filt.shape[0]):
        n = 10
        for j in np.arange(n,ku_filt.shape[1]+n,n):
            t = ku_filt[i,j-n:j]
            ku_mean[i,j-n:j] = np.ma.mean(t)
            masks = t.mask
            no_mask = t[~masks]
            n_mask[i,j-n:j] = len(no_mask)/float(t.shape[0])

    echo = np.ma.zeros(ku_filt.shape[1])
    for j in np.arange(0,n_mask.shape[1]):
        t = n_mask[:,j]
        ind = np.where(t > thresh)
        echo[j] = alt_filt[ind[0][0],j]
        
    return echo

def mask_surf(apr):
    
    k = 12
    ku_orig = apr['Ku'][:,k,:]
    ku_out = np.ma.zeros(apr['Ku'].shape)
    ku = scipy.ndimage.gaussian_filter(ku_orig,1)
    for i in np.arange(0,ku.shape[1]):
        ku_prof = 10**(ku[:,i]/10.)
        alt_prof = apr['alt_gate'][:,k,i]
        surf1 = apr['surface'][k,i]
        filt = scipy.ndimage.sobel(ku_prof)
        #from the surface go up until Der < some value
        filt2 = np.abs(filt)
        tol = filt2[int(surf1)]/filt2[int(surf1)]
        j = 0 
        while tol > 0.01:
            j = j - 1
            ii = int(surf1)+j
            tol = filt2[ii]/filt2[int(surf1)]
        val = 2500
        if alt_prof[ii] >= val:
            ku_prof = np.ma.masked_where(alt_prof <= val,ku_orig[:,i])
            ku_orig[:,i] = ku_prof
        else:
            ku_orig[ii:,i] = np.ma.masked
        
    ku_out[:,k,:] = ku_orig
    
    for k in np.arange(0,24):
        if k == 12:
            continue
        ku_out[:,k,:] = np.ma.masked_where(ku_out[:,12,:].mask,apr['Ku'][:,k,:])
    
    apr['Ku'] = ku_out
    
    return apr

def find_bb(apr,ind1):
    ku = np.squeeze(apr['Ku'][:,12,ind1])
    alt = np.squeeze(apr['alt_gate'][:,12,ind1])
    surf = apr['surface'][12,ind1]
    ind = ku.mask 
    ku[ind] = 0
    ku = scipy.ndimage.gaussian_filter(ku,5)
    top_a = np.zeros(ku.shape[1])
    for i in np.arange(0,ku.shape[1]):
        a = i
        ku_prof = 10**(ku[:,a]/10.)
        alt_prof = alt[:,a]

        filt = scipy.ndimage.sobel(ku_prof)
        filtfilt = scipy.ndimage.sobel(filt)
        filtfiltfilt = scipy.ndimage.sobel(filtfilt)
        k_func1 = interp1d(alt_prof,ku_prof,kind='cubic',bounds_error=False)
        k_func2 = interp1d(alt_prof,filtfilt,kind='cubic',bounds_error=False) 


        alt_new = np.linspace(alt_prof.min(),alt_prof.max(),1000)

        ku_interp = k_func1(alt_new)
        der2 = k_func2(alt_new)
        bbmax = np.where(ku_interp == ku_interp.max())

        ind = np.where(alt_new >= alt_new[bbmax])
        ind2 = np.where(alt_new <= alt_new[bbmax]+500)
        ind3 = np.intersect1d(ind,ind2)

        der2_sub = der2[ind3]
        alt_sub = alt_new[ind3]

        k_func3 =interp1d(alt_sub,der2_sub,kind='cubic',bounds_error=False)
        try:
            top  = scipy.optimize.bisect(k_func3,alt_sub[0],alt_sub[-1])
        except:
            top  = scipy.optimize.bisect(k_func3,alt_sub[0],alt_sub[-1]+100)

        top_a[a] = top

    #clean up top 
    ind = np.where(np.abs(top_a-np.percentile(top_a,50)) >= 300)
    top_a[ind] = np.percentile(top_a,50)
    
    return top_a

def extend_bb(ind,R,bb_center):

    tots = np.arange(0,len(R))
    ex = np.setdiff1d(tots,ind)
    both = np.intersect1d(ind,tots)

    bb_long = np.ma.zeros(len(R))
    bb_long[ind] = bb_center
    for i in np.arange(0,len(ex)):
        t = ex[i]
        index = find_nearest(ind[0],t)
        bb_long[t] = bb_long[ind[0][index]]
    
    return bb_long

def precip_echo_filt3D(ku,thresh=5):
    """
    
    This function provides a filter for APR3 data to determine if there exists a precip echo in the column. Only preforms it 
    on one scan at a time. Could easily go up to all scans. 
    
    MAKE SURE TO APPLY masking routines first... 
    
    """
    precip_yn = np.zeros([ku.shape[1],ku.shape[2]])
    for k in np.arange(0,ku.shape[1]):
        for j in np.arange(0,ku.shape[2]):
            flag1 = 0 
            c1 = -1
            i = -1
            start_flag = 0
            while flag1 == 0:
                i = i + 1

                if c1 >= thresh:
                    precip_yn[k,j] = 1
                    break
                if i == 550: 
                    precip_yn[k,j] = 0
                    break

                t = ku[i,k,j]

                if start_flag ==0:
                    if np.ma.is_masked(t):
                        continue
                    else:
                        start_flag = 1
                        c1 = c1 + 1
                else:
                    if np.ma.is_masked(t):
                        start_flag = 0
                        c1 = -1
                        continue
                    else:
                        c1 = c1 + 1
    return precip_yn
