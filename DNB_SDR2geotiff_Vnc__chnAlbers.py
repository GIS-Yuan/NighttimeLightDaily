# -*- coding: utf-8 -*-

import h5py
import numpy as np
from pyresample import geometry
import os
import xarray as xr
import dask.array as da
from satpy import Scene
from satpy.utils import debug_on
import argparse
from pyresample import get_area_def
# import matplotlib.pyplot as plt
# from pylab import *
from pyresample import create_area_def
from PIL import Image
import datetime
import warnings
import os
import gc

warnings.filterwarnings("ignore")

# 读取SDR文件中的Radiance、QF1_VIIRSDNBSDR以及SDR_GEO中的Longitude_TC、Latitude_TC、QF2_VIIRSSDRGEO、
# SolarZenithAngle、QF1_SCAN_VIIRSSDRGEO、LunarZenithAngle
def read_h5(sdr_data_path, SDR_names, SDR_GEO_names):
    with h5py.File(sdr_data_path, 'r') as sdr_file:
        GROUP_DNB_SDR = dict()
        GROUP_DNB_SDR_GEO = dict()

        if len(SDR_names) != 0:
            for SDR_name in SDR_names:
                temp_subdataset = sdr_file.get(SDR_name)
                if temp_subdataset is None:
                    print("The subdataset:%s don't exist." % (SDR_name))
                    continue
                GROUP_DNB_SDR[SDR_name] = temp_subdataset[()]
                del temp_subdataset

        if len(SDR_GEO_names) != 0:
            for SDR_GEO_name in SDR_GEO_names:
                temp_subdataset = sdr_file.get(SDR_GEO_name)
                if temp_subdataset is None:
                    print("The subdataset:%s don't exist." % (SDR_GEO_name))
                    continue
                GROUP_DNB_SDR_GEO[SDR_GEO_name] = temp_subdataset[()] # temp_subdataset.value
                del temp_subdataset

    return GROUP_DNB_SDR, GROUP_DNB_SDR_GEO


# 对SDR进行质量控制，剔除受边缘噪声、阳光、月光等影响的数据，输出数据还还未进行云掩膜
def sdr_radiance_filter(SDR_GEO_path, SDR_names, SDR_GEO_names, sdr_out_dir):
    GROUP_DNB_SDR, GROUP_DNB_SDR_GEO = read_h5(SDR_GEO_path, SDR_names, SDR_GEO_names)
    sdr_output_name = os.path.basename(SDR_GEO_path)
    sdr_output_name = sdr_output_name.split('.')[0]

    # 1 VIIRS Fill Values
    cloud_radiance = GROUP_DNB_SDR[SDR_names[0]]
    r_fillvalue = np.array([-999.3, -999.5, -999.8, -999.9])
    radiance_mask = np.isin(cloud_radiance, r_fillvalue)

    # 2 Edge-of-swath pixels
    edge_of_swath_mask = np.zeros_like(cloud_radiance, dtype='bool')
    edge_of_swath_mask[:, 0:230] = 1
    edge_of_swath_mask[:, 3838:] = 1

    # 3 QF1_VIIRSDNBSDR_flags
    qf1_viirsdnbsdr = GROUP_DNB_SDR[SDR_names[1]]
    # &符号后面的3、12、48、64由二进制计算，如3=1+2， 12=4+8， 48=16+32，加数均为2的倍数
    SDR_Quality_mask = (qf1_viirsdnbsdr & 3) > 0
    Saturated_Pixel_mask = ((qf1_viirsdnbsdr & 12) >> 2) > 0
    Missing_Data_mask = ((qf1_viirsdnbsdr & 48) >> 4) > 0
    Out_of_Range_mask = ((qf1_viirsdnbsdr & 64) >> 6) > 0
    #
    # 4 QF2_VIIRSSDRGEO_flags
    qf2_viirssdrgeo = GROUP_DNB_SDR_GEO[SDR_GEO_names[2]]
    qf2_viirssdrgeo_do0_mask = (qf2_viirssdrgeo & 1) > 0
    qf2_viirssdrgeo_do1_mask = ((qf2_viirssdrgeo & 2) >> 1) > 0
    qf2_viirssdrgeo_do2_mask = ((qf2_viirssdrgeo & 4) >> 2) > 0
    qf2_viirssdrgeo_do3_mask = ((qf2_viirssdrgeo & 8) >> 3) > 0

    # 5 QF1_SCAN_VIIRSSDRGEO
    qf1_scan_viirssdrgeo = GROUP_DNB_SDR_GEO[SDR_GEO_names[4]]
    within_south_atlantic_anomaly = ((qf2_viirssdrgeo & 16) >> 4) > 0

    # 6 SolarZenithAngle
    solarZenithAngle = GROUP_DNB_SDR_GEO[SDR_GEO_names[3]]
    solarZenithAngle_mask = (solarZenithAngle < 118.5)  # np.where(solarZenithAngle >= 101, 0, 1)

    # 7 LunarZenithAngle
    lunar_zenith = GROUP_DNB_SDR_GEO[SDR_GEO_names[5]]
    moon_illuminance_mask = (lunar_zenith <= 90)

    # 8 Combine pixel level flags
    viirs_sdr_geo_mask = np.logical_or.reduce((
        radiance_mask,
        edge_of_swath_mask,
        solarZenithAngle_mask,
        moon_illuminance_mask,
        SDR_Quality_mask,
        Saturated_Pixel_mask,
        Missing_Data_mask,
        Out_of_Range_mask,
        qf2_viirssdrgeo_do0_mask,
        qf2_viirssdrgeo_do1_mask,
        qf2_viirssdrgeo_do2_mask,
        qf2_viirssdrgeo_do3_mask
    ))

    viirs_sdr_geo_mask_temp = np.logical_or.reduce((
        radiance_mask,
        solarZenithAngle_mask,
        moon_illuminance_mask,
        SDR_Quality_mask,
        Saturated_Pixel_mask,
        Missing_Data_mask,
        Out_of_Range_mask,
        qf2_viirssdrgeo_do0_mask,
        qf2_viirssdrgeo_do1_mask,
        qf2_viirssdrgeo_do2_mask,
        qf2_viirssdrgeo_do3_mask
    ))

    nan_count = np.sum(viirs_sdr_geo_mask_temp == True)
    nan_count_fraction = (nan_count / np.size(viirs_sdr_geo_mask_temp)) * 100

    # 如果数据受月光或者阳光影响太大，导致有效数据占比很小，那么这部分数据被忽略，不保存结果
    if nan_count_fraction == 101:
        print(sdr_output_name + " ignored.")
        del viirs_sdr_geo_mask, radiance_mask, edge_of_swath_mask, solarZenithAngle_mask, moon_illuminance_mask
        del SDR_Quality_mask, Saturated_Pixel_mask, Missing_Data_mask, Out_of_Range_mask, qf2_viirssdrgeo_do0_mask
        del qf2_viirssdrgeo_do1_mask, qf2_viirssdrgeo_do2_mask, qf2_viirssdrgeo_do3_mask, viirs_sdr_geo_mask_temp
        del lunar_zenith
        gc.collect()
    else:
        # 多定义的两个
        del viirs_sdr_geo_mask_temp, GROUP_DNB_SDR

        del radiance_mask, solarZenithAngle_mask, moon_illuminance_mask, edge_of_swath_mask
        del SDR_Quality_mask, Saturated_Pixel_mask, Missing_Data_mask, Out_of_Range_mask, qf2_viirssdrgeo_do0_mask
        del qf2_viirssdrgeo_do1_mask, qf2_viirssdrgeo_do2_mask, qf2_viirssdrgeo_do3_mask
        del lunar_zenith
        gc.collect()

        fill_value = np.nan
        scalefactor = np.float32(pow(10, 9))
        cloud_radiance = cloud_radiance * scalefactor  # convert Watts to nanoWatts
        cloud_radiance[viirs_sdr_geo_mask] = fill_value  # set fill value for masked pixels in DNB
        # del viirs_sdr_geo_mask

        sdr_lon_data = GROUP_DNB_SDR_GEO[SDR_GEO_names[0]]
        sdr_lon_data[viirs_sdr_geo_mask] = np.nan
        sdr_lat_data = GROUP_DNB_SDR_GEO[SDR_GEO_names[1]]
        sdr_lat_data[viirs_sdr_geo_mask] = np.nan
        del viirs_sdr_geo_mask
        gc.collect()
        sdr_swath_def = geometry.SwathDefinition(
            xr.DataArray(da.from_array(sdr_lon_data, chunks=4096), dims=('y', 'x')),
            xr.DataArray(da.from_array(sdr_lat_data, chunks=4096), dims=('y', 'x'))
        )
        sdr_metadata_dict = {'name': 'dnb', 'area': sdr_swath_def}

        sdr_scn = Scene()
        sdr_scn['Radiance'] = xr.DataArray(
            da.from_array(cloud_radiance, chunks=4096),
            attrs=sdr_metadata_dict,
            dims=('y', 'x')  # https://satpy.readthedocs.io/en/latest/dev_guide/xarray_migration.html#id1
        )

        sdr_scn.load(['Radiance'])
        proj_str = '+proj=aea +lat_1=25 +lat_2=48 +lat_0=35 +lon_0=-90 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'  # aea坐标
        sdr_custom_area = create_area_def('aea', proj_str, resolution=750, units='meters', area_extent=[-2328784.272, -1643532.283764, 4125227.112702 , 1511684.573]) # China Aea Extent
        sdr_proj_scn = sdr_scn.resample(sdr_custom_area, resampler='nearest')

        # sdr_proj_shape = sdr_proj_scn.datasets['Radiance'].shape

        sdr_out_path = sdr_out_dir + "\\" + sdr_output_name + '.tif'
        # 必须将enhancement_config设为False，不然输出的值会变的很小
        sdr_proj_scn.save_dataset('Radiance', sdr_out_path, writer='geotiff', dtype=np.float32, enhancement_config=False, fill_value=fill_value)
        print(sdr_output_name + ' processed.')

        # release memory
        sdr_proj_scn = None
        del r_fillvalue
        del fill_value, sdr_proj_scn, sdr_lon_data, sdr_lat_data, sdr_swath_def, sdr_metadata_dict
        gc.collect()

# input_dir代表存放需要处理的sdr文件的文件夹路径
def batch_pro(sdr_input_dir, SDR_out_dir):
    file_list = os.listdir(sdr_input_dir)
    h5_file_list = []
    # 防止出现非h5文件，所以对读出来的文件过滤一下
    for temp_file in file_list:
        if temp_file.endswith('.h5'):
            h5_file_list.append(sdr_input_dir + "\\" + temp_file)

    # 用于在h5文件中提取相应数据的关键字
    SDR_names = ["/All_Data/VIIRS-DNB-SDR_All/Radiance", "/All_Data/VIIRS-DNB-SDR_All/QF1_VIIRSDNBSDR"]
    SDR_GEO_names = ["/All_Data/VIIRS-DNB-GEO_All/Longitude_TC", "/All_Data/VIIRS-DNB-GEO_All/Latitude_TC",
                     "/All_Data/VIIRS-DNB-GEO_All/QF2_VIIRSSDRGEO", "/All_Data/VIIRS-DNB-GEO_All/SolarZenithAngle",
                     '/All_Data/VIIRS-DNB-GEO_All/QF1_SCAN_VIIRSSDRGEO', '/All_Data/VIIRS-DNB-GEO_All/LunarZenithAngle']

    for h5_file in h5_file_list:
        sdr_radiance_filter(h5_file, SDR_names, SDR_GEO_names, SDR_out_dir)

if __name__ == "__main__":
    input_sdr_dir = r"F:\Data\US_Hurricane\SDR\SDR_2021_11" # sdr的存储文件夹
    output_sdr_dir = r"F:\Data\US_Hurricane\SDR\SDR_2021_11_Geotiff" # 输出文件夹；输出是剔除了边缘噪声、阳光、月光等影响的Radiance数据，格式为geotiff
    batch_pro(input_sdr_dir, output_sdr_dir)

    print("Complete")
