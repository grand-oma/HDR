# Computing antenna response to sky noise
# OMH 26/03/2020

import os
import sys
import numpy as np
import matplotlib.pyplot as pl

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

lfdir = "/home/martineau/GRAND/data/LFMap/"
lobedir = "/home/martineau/GRAND/data/TREND/modelAntenna/"

def get_map(freq):
    # Retrieve LFMap data
    strlfmap= lfdir + 'LFmapshort'+str(freq)+'.npy'
    ralf,declf,temp=np.load(strlfmap)
    declf = np.pi/2-declf  # Switch to proper range
    ralf = ralf - np.pi # Switch to proper range
    return ralf, declf,temp

def get_TRENDlobe(freq):
    # Retrieve antenna lobe
    lobefile = lobedir + 'trendEWButterfly'+str(freq)+'MHz112Ohm.npy'
    lobe = np.load(lobefile)
    zenith,azimuth,Vr,Vt,Vp=np.load(lobefile)
    # zenith and azimuth are here in NEC antenna/local referential conventions. Transform them to AltAz conventions
    # Theta > 0deg <=> coming from above horizon
    alt = zenith - np.pi/2
    Vt = np.fliplr(Vt)
    Vt[alt<0] = 0  # Kill all emission from ground
    Vp = np.fliplr(Vp)
    Vp[alt<0] = 0
    # Azimuth = 0 <=> coming from North
    nl = np.shape(Vt)[0]
    Vt = np.roll(Vt,int(nl/4),axis=0)
    Vp = np.roll(Vp,int(nl/4),axis=0)
    aeff = (Vt*Vt+Vp*Vp)*120*np.pi/112
    return azimuth, alt, Vt, Vp, aeff

def build_gal_map(ra_v,dec_v):
    # Build tools for Gal coordinates
    steplat = round(360./np.shape(ra_v)[0])
    steplong = round(180./np.shape(dec_v)[0])
    lat_v = np.arange(-180,180,steplat)
    nl = np.shape(lat_v)[0]
    lon_v = np.arange(-90,90,steplong)
    nb = np.shape(lon_v)[0]
    lat = np.repeat(lat_v,nb)
    lat = np.reshape(lat,(nl,nb))
    lon = np.tile(lon_v,(nl,1))
    temp_gal = np.zeros(shape=np.shape(lat))

    # Now loop on RaDec map and extract temp values
    for i, ra  in enumerate(ra_v):
      print("***",i,"/",len(ra_v))
      for j, dec in enumerate(dec_v):
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.radian, u.radian))
        gal = c.galactic
        l = gal.l.wrap_at(180 * u.deg).degree # Shift map to [-180, +180] range
        b = gal.b.degree
        ii = np.argmin(abs(l-lat_v))
        jj = np.argmin(abs(b-lon_v))
        temp_gal[ii,jj] = temp[i,j]
        if abs(l)<1 and abs(b)<1:
          print(ra,dec)
          print(c)
          print(gal)
          print("Lat (deg):",ii,l,lat_v[ii])
          print("Long (deg):",jj,b,lon_v[jj])
          print("Temp = ",temp_gal[ii,jj])

    return lat,lon,temp_gal

def build_hor_map(ra_v,dec_v,site,timestr):
    # Build tools for Horizontal coordinates
    if site == "lenghu":
        location = EarthLocation(lat=38.4*u.deg, lon=+93.3*u.deg, height=2650*u.m)
    obs_time = Time(timestr) # UTC time
    str_time = Time.strftime(obs_time,"%b-%d-%Y %H:%M")
    stepaz = round(360./np.shape(ra_v)[0])
    stepalt = round(180./np.shape(dec_v)[0])
    az_v = np.arange(0,360,stepaz)
    nl = np.shape(az_v)[0]
    alt_v = np.arange(-90,90,stepalt)
    nb = np.shape(alt_v)[0]
    az = np.repeat(az_v,nb)
    az = np.reshape(az,(nl,nb))
    alt = np.tile(alt_v,(nl,1))
    temp_hor = np.zeros(shape=np.shape(alt))

    # Now loop on RaDec map and extract temp values
    for i, ra  in enumerate(ra_v):
      print("***",i,"/",len(ra_v))
      for j, dec in enumerate(dec_v):
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.radian, u.radian))
        altaz = c.transform_to(AltAz(obstime=obs_time,location=location))
        th = altaz.alt.degree
        phi = altaz.az.degree
        ii = np.argmin(abs(phi-az_v))
        jj = np.argmin(abs(th-alt_v))
        temp_hor[ii,jj] = temp[i,j]

    return az,alt,temp_hor

def build_hor_map2(ra,dec,temp,site,timestr):
    # First transfer RaDec map to Horizontal coordinates
    if site == "lenghu":
        location = EarthLocation(lat=38.4*u.deg, lon=+93.3*u.deg, height=2650*u.m)
    obs_time = Time(timestr, scale='utc',location=location) # UTC time
    str_time = Time.strftime(obs_time,"%b-%d-%Y %H:%M")
    LST = obs_time.sidereal_time('mean')
    LST_hour = str(LST).split('h')[0]
    print("Local Sideral Time:",LST)

    print("Building LFMap in Hor coordinates for LST hour =",LST_hour,"h.")

    c = SkyCoord(ra, dec, frame='icrs', unit=(u.radian, u.radian))
    altaz = c.transform_to(AltAz(obstime=obs_time,location=location))

    # Now build final (ordered) map in Horizonatl coordinates
    nl = np.shape(ra)[0]
    nb = np.shape(ra)[1]
    stepaz = round(360./nl)
    stepalt = round(180./nb)
    az_v = np.arange(0,360,stepaz) * u.degree
    alt_v = np.arange(-90,90,stepalt) * u.degree
    az = np.repeat(az_v,nb)
    az = np.reshape(az,(nl,nb))
    alt = np.tile(alt_v,(nl,1))
    altaz_fin = SkyCoord(az, alt, frame = 'altaz', obstime=obs_time,location=location)

    # Now find corresponding cells and fill temperature matrix accordingly
    sep = altaz.separation(altaz_fin)
    temp_hor = np.zeros(shape=np.shape(alt))
    for i in range(nl):
          print('***',i,'/',nl)
          for j in range(nb):
              sep = altaz.separation(altaz_fin[i,j]).degree
              sepi = np.where(sep == np.amin(sep))
              a = sepi[0][0]
              b = sepi[1][0]
              #print('Minimal distance =',np.amin(sep),'deg in cell',a,b)
              #print('(',altaz_fin[i,j].az.degree,altaz_fin[i,j].alt.degree,') vs (',altaz[a,b].az.degree,altaz[a,b].alt.degree,')')
              temp_hor[i,j] = temp[a,b]

    return np.array(az),np.array(alt),temp_hor

def loop_freq():
    # Loop on all freqs to build LFmaps
    for freq in range(50,305,5):
        print("Now processing frequency =",freq,"MHz.")
        loop_LST(freq)

def loop_LST(freq):
    import scipy.io
    # Build maps at every LST hour for given frequency and save them to disk
    ralf, declf,temp = get_map(freq)
    LST_hours = range(0,24)
    for LST_hour in LST_hours:
        # Rather ugly way to compute LST/UTC time, but OK if timestr is Jan 1st, 2020 and site is LengHu.
        if LST_hour>=13:
            UTC_hour = LST_hour-13
        else:
            UTC_hour = LST_hour+11
        timestr = '2020-1-1 '+str(UTC_hour)+':06:00'
        mapfile = "LFMap_"+str(freq)+"MHz_"+str(LST_hour)+"h.npz"
        if os.path.isfile(lfdir+mapfile) is False:
            az,alt,temp_hor = build_hor_map2(ralf,declf,temp,"lenghu",timestr)
            np.savez(lfdir+"LFMap_"+str(freq)+"MHz_"+str(LST_hour)+"h.npz",az,alt,temp_hor)
            scipy.io.savemat(lfdir+"LFMap_"+str(freq)+"MHz_"+str(LST_hour)+"h.mat", mdict={'az': az, 'alt':alt,'temp_hor':temp_hor})

def compute_power(az_temp,alt_temp,B,az_ant,alt_ant,aeff,obs_time,location):

    az_v = az_temp[:,0]*np.pi/180
    alt_v = alt_temp[0,:]*np.pi/180
    nl = np.shape(az_v)[0]
    nb = np.shape(alt_v)[0]
    dtheta = float(2*np.pi/nl)
    dphi = float(np.pi/nb)
    cosAlt = np.cos(alt_v)

    aza_v = az_ant[:,0]
    alta_v = alt_ant[0,:]
    power = 0
    for i in range(nl):  # Loop on azimuth
      a = np.argmin(abs(aza_v - az_v[i]))
      for j in range(nb):
          b = np.argmin(abs(alta_v - alt_v[j]))
          #print("** Cell (",i,j,")=(",az_v[i], alt_v[j],"): B=",B[i,j])
          #print("Cell (",a,b,")=(",aza_v[a], alta_v[b],"): Aeff=",aeff[a,b])
          power = power + aeff[a,b]*B[i,j]*cosAlt[j]*dtheta*dphi/2
    print("Power = ",power)
    return power.value

def plot_lobe(azimuth,zenith,aeff):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm, colors
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    freq = sys.argv[1]
    fig = pl.figure()
    ax = fig.gca(projection='3d')

    # Switch from Altitude back to Zenith
    zenith = zenith + np.pi/2
    aeff = np.fliplr(aeff)
    X=aeff*np.cos(azimuth)*np.sin(zenith)
    Y=aeff*np.sin(azimuth)*np.sin(zenith)
    Z=aeff*np.cos(zenith)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    #for xb, yb, zb in zip(Xb, Yb, Zb):
    #   ax.plot([xb], [yb], [zb], 'w')

    norm = colors.Normalize(vmin=0, vmax=np.amax(aeff))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.hot, facecolors=cm.hot(norm(aeff)), linewidth=0, antialiased=False)
    ax.set_title('Effective area - '+str(freq) +' MHz')
    #In EW mode, antenna is along the x axis, in NS mode, antenna is along the y axis
    ax.set_xlabel('Northing')
    ax.set_ylabel('Easting')
    ax.set_zlabel('Up')
    m = cm.ScalarMappable(cmap=cm.hot,norm=norm)
    m.set_array(aeff)
    a = fig.colorbar(m)
    a.ax.set_ylabel("m$^2$")
    pl.savefig("lobe_"+str(freq)+"MHz.png")

def plot_map(x,y,z,tit,site=None,timestr=None,lab='Temperature (K)'):
    freq = sys.argv[1]
    lst = sys.argv[2]
    title = tit+'_'+ str(freq) + 'MHz'
    if tit == "LFMap_Horizontal":
        title = title + '_' +lst+'hLST'

    pl.figure()
    #pl.subplot(111, projection="aitoff")
    CS = pl.contourf(x,y,z, 100, cmap=pl.cm.hot)
    pl.title(title)
    pl.xlabel('Latitude (deg)')
    pl.ylabel('Altitude (deg)')
    cbar = pl.colorbar(CS)
    cbar.ax.set_ylabel(lab)

    pl.savefig(title+'.png')

if __name__ == '__main__':
    from astropy import constants as const

    #freq = sys.argv[1]
    dfreq = 5
    freqs = range(50,105,dfreq)
    LST_hours = range(0,24)
    v_rms = np.zeros(shape = np.shape(LST_hours))
    R_load = 75 # Not really clear in the case of TREND but...
    pl.figure(1)
    for j,freq in enumerate(freqs):

        azimuth, altitude, Vt, Vp, aeff = get_TRENDlobe(freq)
        #plot_lobe(azimuth,altitude,aeff)
        # plot_map(azimuth*180./np.pi,altitude*180./np.pi,Vt,"l_eff_theta",lab='$l_{eff}$ (m)')
        # plot_map(azimuth*180./np.pi,altitude*180./np.pi,Vp,"l_eff_phi",lab='$l_{eff}$ (m)')
        #plot_map(azimuth*180./np.pi,altitude*180./np.pi,aeff,"Effective area",lab='$A_{eff}$ (m2)')
        #pl.show()

        #ralf, declf,temp = get_map(freq)
        # ralf_d = np.rad2deg(ralf)
        # declf_d = np.rad2deg(declf)
        # plot_map(ralf_d,declf_d,temp,"LFMap_Equatorial")
        #
        # ra_v = ralf[:,0]
        # dec_v = declf[0,:]
        #lat,lon,temp_gal = build_gal_map(ra_v,dec_v)
        #plot_map(lat,lon,temp_gal,"LFMap_Galactic")

        power = np.zeros(shape = np.shape(LST_hours))
        for i,LST_hour in enumerate(LST_hours):
            #Rather ugly way to compute LST/UTC time, but OK if timestring is Jan 1st, 2020 and site is LengHu.
            if LST_hour>=13:
                UTC_hour = LST_hour-13
            else:
                UTC_hour = LST_hour+11
            timestr = '2020-1-1 '+str(UTC_hour)+':06:00'
            print("UTC time:",timestr)
            print("Approx cooresponding LST (@ Lenghu on Jan 1st 2020):",str(LST_hour)+"h00mn.")
            # Now load temperature map
            mapfile = "LFMap_"+str(freq)+"MHz_"+str(LST_hour)+"h.npz"
            if os.path.isfile(lfdir+mapfile):
                print("Opening file",mapfile)
                b = np.load(mapfile)
                az = b.f.arr_0
                alt = b.f.arr_1
                temp_hor = b.f.arr_2
            else:
                print(mapfile,"was not found. Skip this time slot.")
                continue

            B = 2*pow(freq*1e6/const.c,2)*temp_hor*const.k_B  # Sky radiation power spectral density (Rayleigh-Jeans approx)
            power[i] = compute_power(az,alt,B,azimuth,altitude,aeff,"lenghu",timestr)
        v_rms = v_rms + np.sqrt(power*dfreq*1e6*R_load)*1e6

        pl.plot(LST_hours,power,label = str(freq)+"MHz")

    pl.xlabel("LST time (hours)")
    pl.ylabel("Power density (W/Hz)")
    pl.title("Sky noise density on antenna")
    pl.legend()
    pl.xlim(min(LST_hours), max(LST_hours))

    pl.figure(2)
    pl.plot(LST_hours,v_rms)
    pl.xlabel("LST time (hours)")
    pl.ylabel("V$_{rms}$ ($\mu$V)")
    pl.title("Sky noise on antenna")
    pl.xlim(min(LST_hours), max(LST_hours))

    pl.show()

    # Rather ugly way to compute LST/UTC time, but OK if timestring is Jan 1st, 2020 and site is LengHu.
    # LST_hour = int(sys.argv[2])
    # if LST_hour>=13:
    #     UTC_hour = LST_hour-13
    # else:
    #     UTC_hour = LST_hour+11
    # timestr = '2020-1-1 '+str(UTC_hour)+':06:00'
    # print("UTC time:",timestr)
    # try:
    #     mapfile = "LFMap_"+freq+"MHz_"+str(LST_hour)+"h.npz"
    #     f = open(mapfile)
    #     b = np.load(mapfile)
    #     az = b.f.arr_0
    #     alt = b.f.arr_1
    #     temp_hor = b.f.arr_2
    #     f.close()
    # except IOError:
    #     print("File "+mapfile+" does not exist. Creating it.")
    #     az,alt,temp_hor = build_hor_map2(ralf,declf,"lenghu",timestr)
    #
    # plot_map(az,alt,temp_hor,"LFMap_Horizontal",site,timestr)
    # compute_power(az,alt,temp_hor,azimuth,alt,aeff,site,timestr)
    # pl.show()
