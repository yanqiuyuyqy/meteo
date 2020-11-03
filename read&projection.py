import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import math
import matplotlib
from numpy import deg2rad, rad2deg, arctan, arcsin, tan, sqrt, cos, sin
import numpy as np
import matplotlib.colors as colors
import copy
import netCDF4 as nc
from skimage import exposure
from fy4a import FY4A_AGRI_L1
from PIL import Image

BASE_PATH = r"F:\test"

SATELLITE_PATH = os.path.join(BASE_PATH, "Image4000")

ea = 6378.137  # 地球的半长轴[km]
eb = 6356.7523  # 地球的短半轴[km]
h = 42164  # 地心到卫星质心的距离[km]
λD = deg2rad(104.7)  # 卫星星下点所在经度
# 列偏移
COFF = {"0500M": 10991.5,
        "1000M": 5495.5,
        "2000M": 2747.5,
        "4000M": 1373.5}
# 列比例因子
CFAC = {"0500M": 81865099,
        "1000M": 40932549,
        "2000M": 20466274,
        "4000M": 10233137}
LOFF = COFF  # 行偏移
LFAC = CFAC  # 行比例因子

# data directory helper functions
def satellite_path(tile):
    return os.path.join(SATELLITE_PATH, tile + ".HDF")



def visualize_data(data, title, fig_width=15, fig_height=15):
    """Visualize the satellite image data."""
    # visualize only RGB bands
    # ms images: 0:3
    # pan images: 0
    for k in range(data.shape[2]):
#         print(data.shape)
        ds = data[:, :,k]
        ds = ds.astype(np.float)
        # perform stretching for better visualization
        # cannot use it if pan images
    #     for i in range(data.shape[2]):
    #       p2, p98 = np.percentile(data[:, :, i], (2, 98))
    #       data[:, :, i] = exposure.rescale_intensity(data[:, :, i],
    #                                                     in_range=(p2, p98))
#         fig = plt.figure(figsize=(fig_width, fig_height))
        Imax = np.max(ds)
        print(Imax)
        Imin = np.min(ds)
        print(Imin)
        output = (ds - Imin) / (Imax - Imin) * (255 - 0) + 0
#         plt.axis('off')
#         plt.imshow(output,cmap='Greys')
#         plt.savefig('F:/test/Image/'+str(k)+'.jpg',dpi = 200, bbox_inches='tight')
        img = Image.fromarray(output)
        img = img.convert("L")
        img.save('F:/test/Image/'+str(k)+'.jpg')
def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr
def hdf_to_array(file_path):
    """Takes a file path and returns a hdf file as a 3-dimensional numpy array, width x height x bands."""


    fy4a_agri_l1 = FY4A_AGRI_L1(file_path)
    geo_range = '5, 54, 50, 160, 0.04'
    fy4a_agri_l1.extract('Channel01',geo_range)
    fy4a_agri_l1.extract('Channel02',geo_range)
    fy4a_agri_l1.extract('Channel03',geo_range)
    fy4a_agri_l1.extract('Channel04',geo_range)
    fy4a_agri_l1.extract('Channel05',geo_range)
#     fy4a_agri_l1.extract('Channel06', geo_range)
#     fy4a_agri_l1.extract('Channel07', geo_range)
#     fy4a_agri_l1.extract('Channel08', geo_range)
#     fy4a_agri_l1.extract('Channel09', geo_range)
#     fy4a_agri_l1.extract('Channel10', geo_range)
#     fy4a_agri_l1.extract('Channel11', geo_range)
#     fy4a_agri_l1.extract('Channel12', geo_range)
#     fy4a_agri_l1.extract('Channel13', geo_range)
#     fy4a_agri_l1.extract('Channel14', geo_range)

    channel01 = fy4a_agri_l1.channels['Channel01']
    channel02 = fy4a_agri_l1.channels['Channel02']
    channel03 = fy4a_agri_l1.channels['Channel03']
    channel04 = fy4a_agri_l1.channels['Channel04']
    channel05 = fy4a_agri_l1.channels['Channel05']
#     channel06 = fy4a_agri_l1.channels['Channel06']
#     channel07 = fy4a_agri_l1.channels['Channel07']
#     channel08 = fy4a_agri_l1.channels['Channel08']
#     channel09 = fy4a_agri_l1.channels['Channel09']
#     channel10 = fy4a_agri_l1.channels['Channel10']
#     channel11 = fy4a_agri_l1.channels['Channel11']
#     channel12 = fy4a_agri_l1.channels['Channel12']
#     channel13 = fy4a_agri_l1.channels['Channel13']
#     channel14 = fy4a_agri_l1.channels['Channel14']
    print(channel01.shape)
    row,column = channel01.shape
    print(row,column)

    data = np.zeros(shape=(row, column , 5), dtype=np.float32)
    data[:,:,0] = channel01
    data[:,:,1] = channel02
    data[:,:,2] = channel03
    data[:,:,3] = channel04
    data[:,:,4] = channel05
#     data[:,:,5] = channel06
#     data[:,:,6] = channel07
#     data[:,:,7] = channel08
#     data[:,:,8] = channel09
#     data[:,:,9] = channel10
#     data[:,:,10] = channel11
#     data[:,:,11] = channel12
#     data[:,:,12] = channel13
#     data[:,:,13] = channel14

    data = data.astype(np.float)

    data[np.isnan(data)]=0
    data[data==65535]=0
    data[data==65534]=0
    return data



def fetch_tiles():
    """Function for fetching tile data and labels into a dictionary of numpy arrrays."""
    tile_files_image = []

    for f in os.listdir(SATELLITE_PATH):
        file, ext = os.path.splitext(f)
        if ext.lower() == ".hdf":
            tile_files_image.append(file)
    

    
    tile_image = {}
    tiles = {}
    count = 0
    for tile_image in tile_files_image:
        satellite_data = hdf_to_array(satellite_path(tile_image))
        
        print(satellite_data.shape)
        visualize_data(satellite_data,'result')
        
        tiles[tile_image] = {"data": satellite_data}
        count = count + 1
    return tiles

tiles = fetch_tiles()
print(len(list(tiles.keys())))
print("数据读取完成")