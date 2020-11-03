#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import sys
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2

#查看nc数据基本信息
ofam=Dataset(r'F:\nc_ice\NC_H08_20160125_0400_R21_FLDK.06001_06001.nc')

for item in ofam.dimensions:
    print("name: {}, size: {}\n".format(ofam.dimensions[item].name, ofam.dimensions[item].size))


print("\n")
vars = ofam.variables.keys()
for item in vars:
    print('Variable:\t{}'.format(item))
    print('Dimensions:\t{}'.format(ofam[item].dimensions))
    print('Shape:\t{}\n'.format(ofam[item].shape))


# In[2]:


import netCDF4 as nc
import matplotlib.pyplot as plt
import projection
from PIL import Image
dataset = nc.Dataset(r'F:\nc_ice\geocatL1.HIMAWARI-8.2020190.005000.FLDK.R40.nc')
s_l=np.array(dataset.variables['pixel_surface_type'])
lats=np.array(dataset.variables['pixel_latitude'])
lons=np.array(dataset.variables['pixel_longitude'])
geo_range = [32,44,120,135,0.04]
# west = 70.22154999
# east = 211.1784668
# south = -70.386653564
# north = 70.38653564
# s_l[s_l>0]=1
line, column = projection.getlinecolum(geo_range,'4000M')
line = line.astype(np.int)
column = column.astype(np.int)
# print(line.shape)
# print(s_l.shape)
s_l = s_l[line,column]
s_l[s_l>0]=255
# plt.imshow(s_l)
image = Image.fromarray(s_l, "L")

image = image.resize((750, 600),Image.ANTIALIAS)
dis = np.array(image)
dis[dis>0]=255
print(dis.shape)
plt.imshow(dis)
print(np.max(dis))
print(np.min(dis))
print(dis)


# In[3]:


# lat = ofam.variables['latitude'][:]
# lon = ofam.variables['longitude'][:]
# t = ofam.variables['sea_surface_temperature'][:]
B1 = ofam.variables['albedo_01'][:,:]
B2 = ofam.variables['albedo_02'][:,:]
B3 = ofam.variables['albedo_03'][:,:]
B4 = ofam.variables['albedo_04'][:,:]
B5 = ofam.variables['albedo_05'][:,:]
B6 = ofam.variables['albedo_06'][:,:]
row,column = B1.shape
data = np.zeros(shape=(row, column,6), dtype=np.float32)
data[:,:,0] = B1
data[:,:,1] = B2
data[:,:,2] = B3
data[:,:,3] = B4
data[:,:,4] = B5
data[:,:,5] = B6
ds = copy.deepcopy(data[800:1400, 2000:2750])
print(np.max(ds))
print(np.min(ds))
print(ds.shape)


# In[4]:


NDSI = (B2 - B5)/(B2 + B5)
si = NDSI[800:1400, 2000:2750]
print(si.shape)
CH1 = (B5 - B6)/(B5 + B6)
ch = CH1[800:1400, 2000:2750]


# In[56]:



def visualize_result(data, title, fig_width=15, fig_height=12):
    """Visualize the satellite image data."""
    # set color codes of the classes
    _Ci = [0, 204, 255,255]
    _Dc = [5, 5, 200,255]
    _Ns = [204, 204, 0]
    _Cu = [255, 153, 153]
    _St = [255, 0, 0]
    _Other = [255, 255, 255,0]
    ccolors = (_Dc,_Other)
    norm_ccolors = np.array(ccolors)/255.0
    # visualize only RGB bands
    # ms images: 0:3
    # pan images: 0
    data = data[:,:]
    data = data.astype(np.float)
    # perform stretching for better visualization
    # cannot use it if pan images
    # for i in range(data.shape[2]):
    #   p2, p98 = np.percentile(data[:, :, i], (2, 98))
    #   data[:, :, i] = exposure.rescale_intensity(data[:, :, i],
    #                                                 in_range=(p2, p98))
    fig = plt.figure(figsize=(fig_width, fig_height))
   
    cmap = colors.ListedColormap(norm_ccolors)
    bounds=[-0.5,0.5,1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
#     labels = np.argmax(Ytest, axis=2)
    
#     a.set_title(title)
    rectangles = [matplotlib.patches.Rectangle((0, 0), 1, 1, color=norm_ccolors[r]) for r in range(norm_ccolors.shape[0])]
#     classes = ["Ci","Dc",'Other']
    #Create legend from custom artist/label lists
#     a.legend(rectangles, classes,bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
  
    plt.axis('off')
    plt.imshow(data, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
    plt.savefig('F:/nc_ice/'+title+'.png',dpi = 200, bbox_inches='tight',pad_inches=0.0)
# result = copy.deepcopy(NDSI)
# for i in range(NDSI.shape[0]):
#     for j in range(NDSI.shape[1]):
#         if NDSI[i,j]>0.5 and B4[i,j] > 0.1 and B2[i,j]>0.1 and CH1[i,j] > 0.000001:
#             result[i,j]=0
               
#         else:result[i,j]=1
# data[data==-32768]=2
result = np.where((si>0.6)& (B4[800:1400, 2000:2750]>0.11)& (B2[800:1400, 2000:2750]>0.1)& (dis==0),0,1)
print(np.max(result))
print(np.min(result))

visualize_result(result,'sparse_ice')


# In[10]:


from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt
def visualize_data(data, title, fig_width=15, fig_height=15):
    
    """Visualize the satellite image data."""
    # visualize only RGB bands
    # ms images: 0:3
    # pan images: 0

    data = data[800:1400, 2000:2750,0:3]
    data = data.astype(np.float)
        # perform stretching for better visualization
        # cannot use it if pan images
    for i in range(data.shape[2]):
        p2, p98 = np.percentile(data[:, :, i], (2, 98))
        data[:, :, i] = exposure.rescale_intensity(data[:, :, i],
                                                        in_range=(p2, p98))
    fig = plt.figure(figsize=(fig_width, fig_height))
#         Imax = np.max(ds)
#         print(Imax)
#         Imin = np.min(ds)
#         print(Imin)
#         output = (ds - Imin) / (Imax - Imin) * (255 - 0) + 0
    plt.axis('off')
    plt.imshow(data)
#     print(data.shape)
#     plt.savefig('F:/nc_ice/'+'1.jpg',dpi = 200, bbox_inches='tight',pad_inches=0.0)
#     im = Image.fromarray(data)
#     print(im.format, im.size, im.mode)
#         img = img.convert("L")
#         
#         img.save('F:/test/Image/'+str(k)+'.jpg')
visualize_data(data,'test')


# In[ ]:




