import numpy as np
import requests
import getpass
import splusdata
from tqdm import tqdm

from astropy.io import fits


def get_fits_legacy(ra:float,dec:float,save_path:str,bands:str="grz"):
    """
    downloads 256X256 fits files from legacy survey
    """
    r_link = f"http://legacysurvey.org/viewer/fits-cutout/?ra={ra}&dec={dec}&layer=dr8&pixscale=0.272&bands={bands}"    
    r = requests.get(r_link)
    open(f'{save_path}RA{ra}_DEC{dec}_LEGACY.fits', 'wb').write(r.content)
    
def splus_conn():
    """
    connection with splus
    """
    username = input(prompt="Login: ")
    password = getpass.getpass("Password: ")
    return splusdata.connect(username, password)

def get_fits_splus(ra:float,dec:float,conn,save_path:str,size:int=128,bands:list=["G","R","Z"]):
    """
    downloads fits files from splus
    """
    
    fits_data = [conn.get_cut(ra, dec, 128, band)[1].data for band in bands]
    hdu = fits.PrimaryHDU(np.stack(fits_data, axis=0))
    hdu.writeto(f'{save_path}RA{ra}_DEC{dec}_SPLUS.fits', overwrite=True)

def download_data(coords:list,save_path:str):
    """
    download fits files for grz bands for splus and legacy survey
    """

    print("downloading legacy survey data ...")
    for ra,dec in tqdm(coords):
        get_fits_legacy(ra,dec,save_path)
        
    print("establishing connection to splus ...")
    conn = splus_conn()
    
    print("downloading splus data ...")
    for ra,dec in tqdm(coords):
        get_fits_splus(ra,dec,conn,save_path)
    
    print("Done!")
