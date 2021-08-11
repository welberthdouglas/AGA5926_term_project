import numpy as np
import requests
import getpass
import splusdata
import glob
import random
from tqdm import tqdm

from astropy.io import fits


def get_fits_legacy(ra:float,dec:float,save_path:str,bands:str="grz") -> None:
    """
    downloads 256X256 fits files from legacy survey
    """
    r_link = f"http://legacysurvey.org/viewer/fits-cutout/?ra={ra}&dec={dec}&layer=dr8&pixscale=0.277&bands={bands}"    
    r = requests.get(r_link)
    open(f'{save_path}RA{ra}_DEC{dec}_LEGACY.fits', 'wb').write(r.content)
    
def splus_conn() -> None:
    """
    connection with splus
    """
    username = input(prompt="Login: ")
    password = getpass.getpass("Password: ")
    return splusdata.connect(username, password)

def get_fits_splus(ra:float,dec:float,conn,save_path:str,size:int=128,bands:list=["G","R","Z"]) -> None:
    """
    downloads fits files from splus
    """
    
    fits_data = [conn.get_cut(ra, dec, 128, band)[1].data for band in bands]
    hdu = fits.PrimaryHDU(np.stack(fits_data, axis=0))
    hdu.writeto(f'{save_path}RA{ra}_DEC{dec}_SPLUS.fits', overwrite=True)
    
def add_random_offset(coords:list, offset:float=0.005) -> list:
    """
    randomly adds a offset in RA and DEC in a list of coordinates
    """
    coords_offset_RA = [[round(i[0]+offset,4),i[1]] if random.random()<0.25 else i for i in coords]
    coords_offset_RA_DEC = [[i[0],round(i[1]-offset,4)] if random.random()<0.25 else i for i in coords_offset_RA]  
    
    return coords_offset_RA_DEC

def download_data(coords:list,save_path:str) -> None:
    """
    download fits files for grz bands for splus and legacy survey
    """
    print("establishing connection to splus ...")
    conn = splus_conn()
    
    coords = add_random_offset(coords)
    
    print("downloading splus data ...")
    for ra,dec in tqdm(coords):
        get_fits_splus(ra,dec,conn,save_path)
        
    print("downloading legacy survey data ...")
    for ra,dec in tqdm(coords):
        get_fits_legacy(ra,dec,save_path)
    
    print("Done!")
    
def sample_fits(data_dir:str, batch_size:int) -> tuple:
    """
    gets fits data with size batch_size from directory data_dir and returns a tuple of lists 
    with data from splus and legacy 
    """
    
    # Make a list of all fits files inside the data directory
    all_splus_fits = glob.glob(data_dir + "*SPLUS*")

    # Choose a random batch of SPLUS files and get its LEGACY counterparts
    fits_splus_batch = np.random.choice(all_splus_fits, size=batch_size, replace=False)
    fits_legacy_batch = np.array([i.replace("SPLUS","LEGACY") for i in fits_splus_batch])

    splus_fits_data = []
    legacy_fits_data = []

    for s_fits,l_fits in zip(fits_splus_batch,fits_legacy_batch):
        splus_fits_data.append(fits.open(s_fits)[0].data.swapaxes(0,2))
        legacy_fits_data.append(fits.open(l_fits)[0].data.swapaxes(0,2))
        
    return np.stack(splus_fits_data),np.stack(legacy_fits_data)


def data_augmentation(fits_data:list, augmentation_factor = 4) -> np.array:
    """
    performs data augmentation in a list of np.arrays and return a np.array with the original data plus the
    following transformations: flipup, fliplr and rotation 90 degrees counter clock wise
    augmentation factor: [0,2,3,4]
    """
    
    fits_data = np.stack(fits_data)
    
    if len(fits_data.shape)==3:
        fits_data = np.expand_dims(fits_data, axis=0)
    
    if augmentation_factor == 0:
        return fits_data
    
    fits_UD = np.flip(fits_data, axis=1)
    fits_LR = np.flip(fits_data, axis=2)
    fits_ROT90 = np.rot90(fits_data, axes=[1,2])
    
    augmentated_data = {2:[fits_data,fits_UD],
                        3:[fits_data,fits_UD,fits_LR],
                        4:[fits_data,fits_UD,fits_LR,fits_ROT90]}
        
    return np.concatenate(augmentated_data[augmentation_factor],axis=0)    


def plot_GRZ_histogram(fits_data:np.array, xscale:str="log", figsize = (15,5), bins=1000)->None:
    """
    plot histogram of a np.array with G, R and Z channels 
    """
    
    fig,ax = plt.subplots(1,figsize=figsize)
    for i,band,color in zip(range(3),["G","R","Z"],["blue","green","red"]):
        histogram, bin_edges = np.histogram(fits_data[:,:,i], bins=bins)
        ax.plot(bin_edges[0:-1], histogram, label=band, color=color)
    ax.set_xscale(xscale)
    ax.legend(loc="upper right",prop={'size': 15}, title="Band");

def asinh_shrinkage(fits_data:np.array, mult:float=10,den:float=3)->np.array:
    """
    Performs asinh shrinkage in a np.array
    """
    
    return np.arcsinh((mult*fits_data)/den)

def normalize(fits_data:np.array)->np.array:
    """
    takes a np.array of shape = (n,n,3) and return its normalized form in all channels
    """
    
    num = fits_data - fits_data.min(axis=(0,1,2), keepdims=True)
    den = fits_data.max(axis=(0,1,2), keepdims=True) - fits_data.min(axis=(0,1,2), keepdims=True)
    
    return num/den

def normalize_individual_channels(fits_data:np.array)->np.array:
    """
    takes a np.array of shape = (K,n,n,3) and return its normalized form in all chan
    where K is the number of images
    n is the number of pixels
    3 is the number of channels
    """
    if len(fits_data.shape)==3:
        fits_data = np.expand_dims(fits_data, axis=0)
    
    K,n,*_ = fits_data.shape
    
    # reshape to facilitate operation
    data_reshaped = fits_data.swapaxes(0,2).reshape((n**2,K*3), order="F").T
    
    # normalize
    norm_f = lambda x :(x-x.min())/(x.max()-x.min())
    data_normalized = np.apply_along_axis(norm_f, 1, data_reshaped)
    
    return data_normalized.T.reshape((K,n,n,3), order="F")
    
def fits_processing(fits_data:np.array, mult:float=10)->np.array:
    """
    performs asinh shrinkage and normalization to data
    """
    
    if fits_data.shape[1] == 256:
        mult = 100
    shrinked = asinh_shrinkage(fits_data,mult=mult)
    
    return normalize(shrinked)

def save_images(splus_image, legacy_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    fig = plt.figure(20,20)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(splus_image)
    ax.axis("off")
    ax.set_title("SPLUS")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(legacy_image)
    ax.axis("off")
    ax.set_title("LEGACY")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path) 
    