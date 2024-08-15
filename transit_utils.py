import os
import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
import pandas


#TRANSIT GENERATION
def make_transit(t0, period, duration, depth, ratio, resolution=1/192):
    #t0: the coordinate of the transit with respect to the period window.
    #period: the period of the planet, in days.
    #duration: the length of the transit, in days.
    #depth: the depth of the transit with respect to the star's luminosity, from 0-1.
    #ratio: the ratio of sizes between the planet and the star, from 0-1.
    #resolution: the number of points per day
    
    
    #final 30min resolution (0.25x)
    one_period_time = np.arange(0, period, resolution)

    ab = duration * (1 - ratio) / 2
    bc = duration * ratio

    # Make x,y (time, flux) positions for each trapezoid marker
    a = [0, 1]
    b = [ab, 1 - depth]
    c = [ab + bc, 1 - depth]
    d = [duration, 1]

    x = np.array([a[0], b[0], c[0], d[0]])
    y = [a[1], b[1], c[1], d[1]]

    #reshape the array, take the mean on an axis
    one_period_flux = np.interp(one_period_time, x+t0, y)   
    
    #one_period_flux: an array containing the generated flux values.
    return one_period_flux   
    
def transit_full(time, period, dur, depth, ratio, t0_spread=5, dur_spread=0.002, depth_spread=0.002,
                 tprob=0.7, t2prob=0.4, binsize=4, snr=10, fluxratio=1):
    #time: the size of the simulated lightcurve, in days.
    #period: the period of the planet, in days.
    #dur: the duration of the transit, in days.
    #depth: the depth of the transit with respect to the normal luminosity, from 0-1.
    #ratio: the ratio between the sizes of the planet and star, from 0-1.
    #t0_spread: the scatter in t0 between transits, in days.
    #dur_spread: the scatter in transit duration.
    #depth_spread: the scatter in transit depth.
    #tprob: the probability of a successful transit.
    #t2prob: the probability of a transit of the secondary star.
    #binsize: the size of bins, in datapoints.
    #snr: the signal-to-noise ratio, from 0-1.
    #fluxratio: the luminosity ratio between the primary and secondary stars.
    
    #Setup output and determine the number of periods.
    fulltransitcurve = []
    twindow = np.zeros(int(np.ceil(time/period)))
    np.random.default_rng()

    for i in range(0,int(np.ceil(time/period))):
        t_insert = np.random.uniform(0,1)
        
        #Generate randomised parameters for a transit injection
        if t_insert < tprob:
            t0_instance = period/2 + np.random.normal(0,t0_spread)
            dur_instance = np.random.normal(dur,dur_spread)
            depth_instance = np.random.normal(depth,depth_spread)
            tinstance = make_transit(t0_instance,period,dur_instance,depth_instance,ratio)
            
            #Generate parameters for a secondary transit
            if t_insert < t2prob:
                t2t0 = t0_instance + np.random.uniform(-5.0,5.0)
                t2 = make_transit(t2t0,period,dur_instance,depth_instance*fluxratio,ratio)
                tinstance = tinstance+t2-1
            
            fulltransitcurve += tinstance.tolist()

        #Skip a failed transit (precession shifts planet out of transit window)
        else:
            fulltransitcurve += [1.0]*int(period*192-1)

    #Add noise with the specified SNR
    sigma = depth * np.sqrt(dur) / snr #mark for review
    noise = np.random.normal(0.0, sigma, len(fulltransitcurve))
            
    fulltransitcurve += noise
    
    #Bin the synthesized light curve to the specified resolution
    n = -(len(fulltransitcurve)%binsize)
    if n == 0:
        tcarray = np.array(fulltransitcurve)
    else:
        tcarray = np.array(fulltransitcurve[:n])
    tcarray = tcarray.reshape(np.int(len(tcarray)/binsize),binsize)
    finallc = np.mean(tcarray,axis=1)
    
#     #Fold the simulated light curve
#     finalt = np.arange(0,len(finallc))
#     flc = lk.LightCurve(time=finalt,flux=finallc)
#     flcfolded = flc.fold(epoch_time=t0_instance,period=period).bin(bins=1000) #mark for review
    
    
    #finallc: the final simulated lightcurve, with 1 point each 30 minutes.
    return finallc

def make_negative(time, sigma):
    #time: the size of the light curve array desired, in days.
    #sigma: the spread of the gaussian noise distribution.
    
    noisearray = np.random.normal(1.0, sigma, int(time*48)-1)
    
    #noisearray: the specified gaussian noise-only array. Assume 1 point = 30 minutes (it doesn't actually matter what the resolution is).
    return noisearray




#ANTICLIPPING / POST-FOLD DATA PROCESSING & EXTRACTION
def anticlip_data(view, low_sigma, high_sigma):
    # view: the original lightcurve array to be investigated.
    # low_sigma: the significance of a point needed to exceed the anticlip threshold.
    # high_sigma: the significance of a point needed to pass normal sigma clipping.
    

    median = np.median(view)
    sigma = np.std(view)
    indexes = []
    clipped_view = []
    
    upper_limit = median + high_sigma * sigma

    for i in range(len(view)):
        if (view[i] <= upper_limit):
            clipped_view.append(view[i])
        else:
            clipped_view.append(1)

    median_new = np.median(clipped_view)
    sigma_new = np.std(clipped_view)
    lower_limit = median - low_sigma * sigma
    
    for i in range(len(clipped_view)):
        if (clipped_view[i] <= lower_limit):
            indexes.append(i)
            
    # indexes: the positions of points >n_sigma * sigma below the median.
    # clipped_view: the values of points >n_sigma * sigma below the median.
    return indexes, clipped_view


def find_clusters(view, indexes, spacing, min_cluster):
    #view: the original lightcurve array to be investigated.
    #indexes: the array of all outliers identified by anticlip_data.
    #spacing: the maximum allowed datapoint gap size within a cluster.
    #min_cluster: the minimum size of a cluster of significant outliers, in datapoints.
    
    cl = []
    clusters = []
    clusterflux = np.full((len(view)), None)
    
    i = 0
    while(i < len(indexes)):
        cl.append(int(indexes[i]))
        i+=1
        if (i < len(indexes)):
            while(indexes[i] <= indexes[i-1] + spacing):
                cl.append(int(indexes[i]))
                i+=1
                if (i >= len(indexes)):
                    break
        if (len(cl) >= int(min_cluster)):
            clusters = np.append(clusters,cl)
        cl = []
        
        
    for index in clusters:
        clusterflux[int(index)] = view[int(index)]
    
    #clusterflux: a full-length array of NoneType with the correct flux values placed at the correct indices.
    return clusters, clusterflux

def fold_anticlip(t, f, clusters, p, t0, nbins, cliplength):
    # t is the time array.
    # f is the flux value array.
    #p is the period in DATAPOINTS.
    # nbins is the number of desired bins across the period.
    # clusters is an array containing only cluster flux values.
    #cliplength is the radius around t0 in which points are to be retained.
    
    assert len(t) == len(f) == len(clusters), 'time, flux, clusters arrays must have the same lengths'

    folded_lc = []
    phi = ((t - t0 + cliplength*0.5) / p) % 1
    bin_idxs = np.digitize(phi,np.linspace(0,1,nbins)) #Numpy digitize
    
    bin_id_clip = bin_idxs < cliplength / (p/nbins)
    
    bin_idxs = bin_idxs[bin_id_clip]
    f = f[bin_id_clip]
    clusters = clusters[bin_id_clip]
    
    fluxassigned = np.column_stack((bin_idxs,f,clusters))
    
    df = pandas.DataFrame(fluxassigned,columns=['Index','Flux','Cluster']).groupby('Index').mean()
    
    df.Cluster.fillna(df.Flux, inplace=True)
    
    fold_flux=df.iloc[:,1:].values
    
    fold_flux = fold_flux.transpose()[0]
    
    return fold_flux

def transit_extraction(view, cluster_list, rangelength):
    #view: the full lightcurve array to be processed.
    #clist: the array of cluster indices extracted with find_clusters.
    #rangelength: the selection DIAMETER of each cluster, in days.
    
    cluster_array = []
    cindices = []
    clist = cluster_list.astype(int)
    
    for i in range(len(clist)-1):
        if clist[i - 1] != clist[i]-1 and clist[i] > rangelength*24 and clist[i] < len(view)-rangelength*24:
            cluster = []
            for j in range(clist[i]-rangelength*24, clist[i]+rangelength*24):
                cluster = np.append(cluster, view[j])
            cluster_array.append(cluster)
            cindices.append(clist[i]-rangelength*24)
        
    if cluster_array != []:
        cluster_extract = np.vstack(cluster_array)
    else:
        cluster_extract = []
    
    return cluster_extract, cindices