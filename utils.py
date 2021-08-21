import numpy as np
import matplotlib.pyplot as plt
import requests
import getpass
import splusdata
import glob
import random

from tqdm import tqdm
from astropy.io import fits

from config import *

random.seed(SEED)

def get_fits_legacy(ra:float,dec:float,save_path:str,bands:str="grz") -> None:
    """
    downloads 256X256 fits files from legacy survey
    """
    r_link = f"http://legacysurvey.org/viewer/fits-cutout/?ra={ra}&dec={dec}&layer=dr8&pixscale=0.277&bands={bands}"    
    r = requests.get(r_link)
    open(f'{save_path}RA_{ra}_DEC_{dec}_LEGACY.fits', 'wb').write(r.content)
    
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
    hdu.writeto(f'{save_path}RA_{ra}_DEC_{dec}_SPLUS.fits', overwrite=True)
    
def add_random_offset(coords:list, offset:float=0.005) -> list:
    """
    randomly adds a offset in RA and DEC in a list of coordinates
    """
    
    add_coord = [[0,offset],[offset,0],[offset,offset],[offset,-offset]]
    add_coord.extend([[-1*i[0],-1*i[1]] for i in add_coord])

    indices = np.random.choice(len(add_coord), len(coords), replace=True)
    offsets = [add_coord[i] for i in indices]
    
    coords_offsetted = [[round(i[0]+j[0],4),round(i[1]+j[1],4)] for i,j in zip(coords,offsets)]
    
    return coords_offsetted

def download_data(coords:list,save_path:str,train_samples:int=125, offset_aug:bool=True) -> None:
    """
    download fits files for grz bands for splus and legacy survey
    """
    print("establishing connection to splus ...")
    conn = splus_conn()
    
    random.shuffle(coords)
    train_obj_coords = coords[:train_samples]
    validation_obj_coords = coords[train_samples:]
    
    if offset_aug:
        train_obj_coords = train_obj_coords + add_random_offset(train_obj_coords)
    
    print("downloading splus data ...")
    for ra,dec in tqdm(train_obj_coords):
        get_fits_splus(ra,dec,conn,save_path+"train/")
    
    for ra,dec in tqdm(validation_obj_coords):
        get_fits_splus(ra,dec,conn,save_path+"validation/")
        
    print("downloading legacy survey data ...")
    
    for ra,dec in tqdm(train_obj_coords):
        get_fits_legacy(ra,dec,save_path+"train/")
    
    for ra,dec in tqdm(validation_obj_coords):
        get_fits_legacy(ra,dec,save_path+"validation/")
    print("Done!")
    
def sample_fits(data_dir:str, batch_size:int) -> tuple:
    """
    gets fits data with size batch_size from directory data_dir and returns a tuple of lists 
    with data from splus and legacy 
    """
    
    # Make a list of all splus fits files inside the data directory
    all_splus_fits = glob.glob(data_dir + "*SPLUS*")

    # Choose a random batch of SPLUS files and get its LEGACY counterparts
    fits_splus_batch = np.random.choice(all_splus_fits, size=batch_size, replace=False)
    fits_legacy_batch = np.array([i.replace("SPLUS","LEGACY") for i in fits_splus_batch])

    splus_fits_data = []
    legacy_fits_data = []
    coords = []

    for s_fits,l_fits in zip(fits_splus_batch,fits_legacy_batch):
        splus_fits_data.append(fits.open(s_fits, memmap=False)[0].data.swapaxes(0,2))
        legacy_fits_data.append(fits.open(l_fits, memmap=False)[0].data.swapaxes(0,2))
        coords.append([s_fits.split("_")[1],s_fits.split("_")[3]])
        
    return np.stack(splus_fits_data),np.stack(legacy_fits_data),coords


def data_augmentation(fits_data:list, augmentation_factor = 4) -> np.array:
    """
    performs data augmentation in a list of np.arrays and return a np.array with the original data plus the
    following transformations: flipup, fliplr and rotation 90 degrees counter clock wise
    augmentation factor: [1,2,3,4]
    """
    
    fits_data = np.stack(fits_data)
    
    if len(fits_data.shape)==3:
        fits_data = np.expand_dims(fits_data, axis=0)
    
    if augmentation_factor == 1:
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
    for i,band,color in zip(range(3),["g","r","z"],["royalblue","darkcyan","indianred"]):
        histogram, bin_edges = np.histogram(fits_data[:,:,i], bins=bins)
        ax.plot(bin_edges[0:-1], histogram, label=band, color=color,linewidth=1)
        ax.set_xlabel("pixel value", fontsize=12);
        ax.set_ylabel("count                  ",rotation =0, fontsize=12);
    ax.set_xscale(xscale)
    ax.set_xlim([0, 1]);
    ax.legend(loc="upper right",prop={'size': 12}, title="Band");

def asinh_shrinkage(fits_data:np.array, mult:float=10,den:float=3)->np.array:
    """
    Performs asinh shrinkage in a np.array
    """
    
    return np.arcsinh((mult*fits_data)/den)

def normalize(fits_data:np.array)->np.array:
    """
    takes a np.array of shape = (K,n,n,3) and return its normalized form in all channels
    """
    if len(fits_data.shape)==3:
        fits_data = np.expand_dims(fits_data, axis=0)
    
    num = fits_data - fits_data.min(axis=(1,2,3), keepdims=True)
    den = fits_data.max(axis=(1,2,3), keepdims=True) - fits_data.min(axis=(1,2,3), keepdims=True)
    
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

def fits2images(data_dir, images_dir):
    train_splus_fits, train_legacy_fits, train_coords = sample_fits(data_dir+"train/",250)
    val_splus_fits, val_legacy_fits, val_coords = sample_fits(data_dir+"validation/",25)
    
    train_splus_images,train_legacy_images = fits_processing(train_splus_fits),fits_processing(train_legacy_fits)
    val_splus_images,val_legacy_images = fits_processing(val_splus_fits),fits_processing(val_legacy_fits)

    print("saving training images ...")
    for i in tqdm(range(250)):
        plt.imsave(images_dir+f'train/RA_{train_coords[i][0]}_DEC_{train_coords[i][1]}_SPLUS.png',train_splus_images[i])
        plt.imsave(images_dir+f'train/RA_{train_coords[i][0]}_DEC_{train_coords[i][1]}_LEGACY.png',train_legacy_images[i])
    
    print("saving validation images ...")
    for i in tqdm(range(25)):

        plt.imsave(images_dir+f'validation/RA_{val_coords[i][0]}_DEC_{val_coords[i][1]}_SPLUS.png',val_splus_images[i])
        plt.imsave(images_dir+f'validation/RA_{val_coords[i][0]}_DEC_{val_coords[i][1]}_LEGACY.png',val_legacy_images[i])

def save_images(splus_image, legacy_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    fig,ax = plt.subplots(1,3,figsize=(20,20))
    ax[0].imshow(splus_image)
    ax[0].set_title("SPLUS")
    
    ax[1].imshow(generated_image)
    ax[1].set_title("SPLUS - SR")

    ax[2].imshow(legacy_image)
    ax[2].set_title("LEGACY")

    plt.savefig(path) 
    plt.close('all')
    
def get_data(data_dir:str = DATA_DIR+"train/",
             train_size:float = TRAIN_SIZE,
             augmentation_factor:int = AUGMENTATION_FACTOR)->tuple:
    
    # load fits
    train_splus_fits, train_legacy_fits,_ = sample_fits(data_dir,train_size)
    
    # asinh shrink and normalize
    train_splus_images,train_legacy_images = fits_processing(train_splus_fits),fits_processing(train_legacy_fits)
    
    # perform data augmentation
    augmented_data = (data_augmentation(train_splus_images, augmentation_factor=augmentation_factor),
                      data_augmentation(train_legacy_images, augmentation_factor=augmentation_factor))
    
    return augmented_data