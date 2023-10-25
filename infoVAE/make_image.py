# This file is from Zehao and Tobias
# sys.argv[1]: name_file_path variable
# sys.argv[2]: imsave folder path
# sys.argv[3]: folder_path variable (data source)


# This code makes mock observation images using Faucher's SKIRT radiative transfer result.
# includes: add PSF, add poisson noise, add guassian sky, and lupton rgb

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, sys
from astropy.visualization import make_lupton_rgb
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel
from tqdm import tqdm

### Faucher data params
# Broadband photometric bandpass names
band_names = ['FUV', 'NUV', 'u', 'g', 'r', 'i', 'z', '2MASS_J', '2MASS_H', '2MASS_KS', 'W1',
              'W2', 'W3', 'W4', 'PACS70', 'PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']


# NIHAO galaxy names
# names = ['g1.88e10','g1.89e10','g1.90e10','g2.34e10','g2.63e10','g2.64e10','g2.80e10','g2.83e10','g2.94e10',
#             'g3.44e10','g3.67e10','g3.93e10','g4.27e10','g4.48e10','g4.86e10','g4.94e10','g4.99e10',
#             'g5.05e10','g6.12e10','g6.37e10','g6.77e10','g6.91e10','g7.12e10','g8.89e10','g9.59e10','g1.05e11',
#             'g1.08e11','g1.37e11','g1.52e11','g1.57e11','g1.59e11','g1.64e11','g2.04e11','g2.19e11','g2.39e11',
#             'g2.41e11','g2.42e11','g2.54e11','g3.06e11','g3.21e11','g3.23e11','g3.49e11','g3.55e11','g3.59e11',
#             'g3.71e11','g4.90e11','g5.02e11','g5.31e11','g5.36e11','g5.38e11','g5.46e11','g5.55e11','g6.96e11',
#             'g7.08e11','g7.44e11','g7.55e11','g7.66e11','g8.06e11','g8.13e11','g8.26e11','g8.28e11','g1.12e12',
#             'g1.77e12','g1.92e12','g2.79e12']

# names = ['397866','17009','99155','324171','599084']
# name_file_path = '../sdss/snapnum_135/subfind_ids.txt'
names = []
name_file_path = sys.argv[1]
with open(name_file_path, 'r') as txt_file:
    for line in txt_file:
        num = line.strip()
        names.append(num)



### Realsim params
#https://github.com/cbottrell/RealSim/blob/master/ObsRealism.py
airmass  = {'u':1.178, 'g':1.178, 'r':1.177, 'i':1.177, 'z':1.178}
aa       = {'u':-23.80,'g':-24.44,'r':-24.03,'i':-23.67,'z':-21.98}
kk       = {'u':0.5082,'g':0.1898,'r':0.1032,'i':0.0612,'z':0.0587}
gain     = {'u':1.680, 'g':3.850, 'r':4.735, 'i':5.111, 'z':4.622}
exptime  = 53.907456 # seconds

# statistics on sky noise (obtained from averages over all Legacy galaxies)
skySig = {'u':23.872, 'g':24.880, 'r':24.384, 'i':23.820, 'z':22.356}
# standard deviation in sky noise (sky noise level is drawn from this distribution)
SigskySig = {'u':0.147, 'g':0.137, 'r':0.109, 'i':0.119, 'z':0.189}
# statistics on seeing (obtained from averages over all Legacy galaxies)
seeing = {'u':1.551, 'g':1.469, 'r':1.356, 'i':1.286, 'z':1.308}
# standard deviation in seeing (seeing is drawn from this distribution)
Sigseeing = {'u':0.243, 'g':0.221, 'r':0.221, 'i':0.222, 'z':0.204}


### some unit conversion
# https://www.cs.princeton.edu/~rpa/pubs/regier2019approximate.pdf
# 1 nanomaggy is equivalent to 3.631×10^−6 Jansky
nanomaggy_to_Jy = 3.631e-6

# kpc/arcsec
# mean,min,max,median of SDSS sample redshift (0.10777628, 0.00500689, 0.843643, 0.1015725)
# kpc_per_arcsec=((cosmo.angular_diameter_distance(0.10777628)*u.arcsec.to(u.rad)).to(u.kpc)).value
kpc_per_arcsec=((cosmo.angular_diameter_distance(0.0485)*u.arcsec.to(u.rad)).to(u.kpc)).value








def makeImages(name,g, r, i, kpc_per_pixel):  
    # for i in tqdm(range(len(flux[:,0,0,0])),desc='rotation',leave=False): # loop over orientations
    # RGB = i, r, g:
    # img = {'i':flux[i,5,:,:], 'r':flux[i,4,:,:], 'g':flux[i,3,:,:]}
    img = {'i':i, 'r':r, 'g':g}

    # kpc/pixel
    #kpc_per_pixel = width*u.pc.to(u.kpc)/img['i'].shape[0]
    print("kpc_per_pixel", kpc_per_pixel)

    for band in ['i','r','g']:
        # convert to nanomaggies
        #img[band] = img[band]/nanomaggy_to_Jy

        # draw a random PSF size from the distribution of typical PSF sizes SDSS
        #psfsize = np.random.normal(seeing[band],Sigseeing[band])
        psfsize = seeing[band] # use mean seeing

        # in kpc
        print("kpc_per_arcsec", kpc_per_arcsec)
        psf_FWHM=kpc_per_arcsec*psfsize
        psf_sigma=psf_FWHM/2.35482004503 #https://en.wikipedia.org/wiki/Full_width_at_half_maximum

        # Gaussian PSF
        kernel = Gaussian2DKernel(psf_sigma/kpc_per_pixel)
        print("Num of pixels: ", psf_sigma/kpc_per_pixel)
        kernel.normalize()
        kernel = kernel.array  # we only need the numpy array
        # convolve with PSF
        img[band] = convolve(img[band], kernel)
        
        ### add poisson noise to image ###
        # conversion factor from nanomaggies to counts
        counts_per_nanomaggy = exptime*10**(-0.4*(22.5+aa[band]+kk[band]*airmass[band]))
        print("counts_per_nanomaggy", counts_per_nanomaggy)
        # image in counts for given field properties
        img_counts = np.clip(img[band] * counts_per_nanomaggy,a_min=0,a_max=None)
        #img_counts = np.clip(img[band],a_min=0,a_max=None) #np.copy(img[band])
        # poisson noise [adu] computed accounting for gain [e/adu]
        img_counts = np.random.poisson(lam=img_counts*gain[band])/gain[band]
        # convert back to nanomaggies
        img[band] = img_counts / counts_per_nanomaggy
         
        ### add gaussian sky to image ###
        # sky sig in AB mag/arcsec2
        # draw a random sky noise from the distribution of typical skies in SDSS
        #false_sky_sig = np.random.normal(skySig[band],SigskySig[band])
        false_sky_sig = skySig[band] # use mean sky
        # conversion from mag/arcsec2 to nanomaggies/arcsec2
        false_sky_sig = 10**(0.4*(22.5-false_sky_sig))
        # account for pixel scale in final image
        arcsec_per_pixel = kpc_per_pixel / kpc_per_arcsec
        false_sky_sig *= arcsec_per_pixel**2
        # create false sky image
        sky = false_sky_sig*np.random.randn(*img[band].shape)
        # add false sky to image in nanomaggies
        img[band] += sky
        
        # back to Jansky
        img[band] *= nanomaggy_to_Jy
        
    # make rgb image
    image = make_lupton_rgb(img['i'],img['r'],img['g'],Q=20,stretch=np.array([img['i'].mean(),img['r'].mean(),img['g'].mean()]).mean()/0.5)
    # plt.imsave('mockobs_0915/'+galaxy+'_'+str(i).zfill(2)+'.png',image,origin='lower')
    # plt.imsave('../mock_illustris_1_sdss_135/'+'broadband_' + name+ '.png',image,origin='lower')
    plt.imsave(sys.argv[2] +'broadband_' + name + '.png',image,origin='lower')

    return None

        







### main
# folder_path = '../sdss/snapnum_135/data/'
folder_path = sys.argv[3]

for name in tqdm(names,total=len(names),desc='names'):
    # with fits.open('NIHAO_SKIRT/'+galaxy+'_nihao-resolved-photometry.fits') as hdulist:

    with fits.open(folder_path +'broadband_' + name +'.fits') as hdulist:
        # Read the SUMMARY extension
        # summary = hdulist['SUMMARY'].data
        g_r_i_z = hdulist[0].data
    # conversion to nanomaggies: https://www.illustris-project.org/data/forum/topic/304/converting-mock-image-counts-to-flux-magnitudes/    
    # 
    counts_per_nanomaggy = exptime*10**(-0.4*(22.5+aa['g']+kk['g']*airmass['g']))
    g = g_r_i_z[0] / counts_per_nanomaggy / nanomaggy_to_Jy #/ 2.40238e+10 #/ 3.631e-6 
    counts_per_nanomaggy = exptime*10**(-0.4*(22.5+aa['r']+kk['r']*airmass['r']))
    r = g_r_i_z[1] / counts_per_nanomaggy / nanomaggy_to_Jy #/ 2.3826e+10 #/ 3.631e-6 
    counts_per_nanomaggy = exptime*10**(-0.4*(22.5+aa['i']+kk['i']*airmass['i']))
    i = g_r_i_z[2] / counts_per_nanomaggy / nanomaggy_to_Jy #/ 1.59001e+10 #/ 3.631e-6 
    # flux = summary['flux']
    # width = summary['size'].mean()
    #width = hdulist[0].header["CDELT1"]
    #print("width", width)
    kpc_per_pixel = hdulist[0].header["CDELT1"]/1000.
    #arcsec_per_pixel = kpc_per_pixel / kpc_per_arcsec
    #print("arcsec_per_pixel", arcsec_per_pixel)
    #print("image shape: ", g.shape)
    #makeImages(name,g/arcsec_per_pixel**2, r/arcsec_per_pixel**2, i/arcsec_per_pixel**2,kpc_per_pixel)
    makeImages(name, g, r, i, kpc_per_pixel)




"""
with fits.open('../sdss/snapnum_095/data/' +'broadband_120.fits') as hdulist:
    header = hdulist[0].header
    data = hdulist[0].data
    for keyword, value in header.items():
        print(f"{keyword}: {value}")
    print(data.shape)
"""

