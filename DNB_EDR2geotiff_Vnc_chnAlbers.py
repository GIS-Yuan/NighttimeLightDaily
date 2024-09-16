import numpy as np
from pyresample import geometry
import xarray as xr
import dask.array as da
from satpy import Scene
from pyresample import create_area_def
import os
import netCDF4 as nc
import gc
import re
import warnings
warnings.filterwarnings("ignore")

# get filename without suffix from absolute path
def get_filename_without_suffix(inputPath):
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(inputPath)
    if data:
        return data[0]

def read_nc(edr_data_path, EDR_names):
    with nc.Dataset(edr_data_path, 'r') as edr_file:
        day_night_flag = edr_file.day_night_data_flag
        if day_night_flag == "day": # just make nighttime data
            GROUP_DNB_EDR = dict()
            edr_file_name = get_filename_without_suffix(edr_data_path)
            print(edr_file_name + " in day time")
        else:
            GROUP_DNB_EDR = dict()
            for EDR_name in EDR_names:
                temp_subdataset = edr_file.variables[EDR_name]
                if temp_subdataset is None:
                    print("The subdataset:%s don't exist." % (EDR_name))
                    continue
                temp_subdataset = np.array(temp_subdataset)  # to numpy array
                GROUP_DNB_EDR[EDR_name] = temp_subdataset
                del temp_subdataset

    return GROUP_DNB_EDR

def edr_cloud_mask2geotiff(EDR_GEO_path, EDR_names, edr_out_dir):
    GROUP_DNB_EDR = read_nc(EDR_GEO_path, EDR_names)
    if len(GROUP_DNB_EDR) == 0: # skip daytime data
        return
    edr_output_name = os.path.basename(EDR_GEO_path)
    edr_output_name = edr_output_name.split('.')[0]

    # read cloud mask
    cloudPro = GROUP_DNB_EDR[EDR_names[0]]

    # export clouds=1, noclouds=0
    # cloudPro[cloudPro <= 0.25] = 0
    # cloudPro[cloudPro > 0.25] = 1
    # CloudDetection_ConfidencePixel_mask = np.ma.masked_greater(cloudPro, 0.25)
    # CloudDetection_ConfidencePixel_mask = np.ma.filled(CloudDetection_ConfidencePixel_mask, fill_value=np.nan)

    edr_lon_data = GROUP_DNB_EDR[EDR_names[1]]
    edr_lat_data = GROUP_DNB_EDR[EDR_names[2]]
    edr_lon_data[edr_lon_data == -999.0] = np.nan
    edr_lat_data[edr_lat_data == -999.0] = np.nan
    lon_min = np.nanmin(edr_lon_data)
    lon_max = np.nanmax(edr_lon_data)
    lat_min = np.nanmin(edr_lat_data)
    lat_max = np.nanmax(edr_lat_data)
    edr_swath_def = geometry.SwathDefinition(
        xr.DataArray(da.from_array(edr_lon_data, chunks=4096), dims=('y', 'x')),
        xr.DataArray(da.from_array(edr_lat_data, chunks=4096), dims=('y', 'x'))
    )
    edr_metadata_dict = {'name': 'cloud_mask', 'area': edr_swath_def}
    del edr_lon_data, edr_lat_data, edr_swath_def
    gc.collect()

    cloudPro = cloudPro.astype(np.uint8)
    # CloudDetection_ConfidencePixel_mask[qf1_viirscmedr == 0] = 255
    edr_scn = Scene()
    edr_scn['Cloud_mask'] = xr.DataArray(
        da.from_array(cloudPro, chunks=4096),
        attrs=edr_metadata_dict,
        dims=('y', 'x')
    )
    del cloudPro, edr_metadata_dict
    gc.collect()

    edr_scn.load(['Cloud_mask'])
    # wgs84:+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs
    # aea:proj=aea +lat_1=29.5 +lat_2=42.5
    # CHN_aea:+proj=aea +lat_1=27 +lat_2=45 +lat_0=35 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=mm +no_defs
    proj_str = '+proj=aea +lat_1=27 +lat_2=45 +lat_0=35 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    edr_custom_area = create_area_def('aea', proj_str, resolution=750, units='degree', area_extent=[-2641644.056319, -3051079.954397, 2222583.910354, 2174272.289243]) # China Aea Extent
    edr_proj_scn = edr_scn.resample(edr_custom_area, resampler='nearest', radius_of_influence=2000)
    del edr_scn
    gc.collect()

    edr_out_path = edr_out_dir + "\\" + edr_output_name + '_CM' + '.tif'  #
    with np.errstate(invalid='ignore'):
        edr_proj_scn.save_dataset('Cloud_mask', edr_out_path, writer='geotiff', dtype=np.uint8, enhancement_config=False, fill_value=255)
    print(edr_output_name + ' processed.')

    # release memory
    del edr_proj_scn, edr_output_name
    gc.collect()

# Search all nc file, including subfolders
def search_nc_file(inDir):
    nc_fileList = []
    for root, dirs, files in os.walk(inDir, topdown=False):
        for name in files:
            if name.endswith(".nc"):
                nc_fileList.append(os.path.join(root, name))

    return nc_fileList

def batch_pro(edr_input_dir, EDR_out_dir):
    # file_list = os.listdir(edr_input_dir)
    nc_file_list = search_nc_file(edr_input_dir)
    # just .nc file
    # for temp_file in nc_file_list:
    #     if temp_file.endswith('.nc'):
    #         nc_file_list.append(edr_input_dir + "\\" + temp_file)

    # key words
    EDR_names = ['CloudMaskBinary', 'Longitude', 'Latitude']

    for nc_file in nc_file_list:
        # edr_cloud_mask2geotiff(nc_file, EDR_names, EDR_out_dir)
        nc_dir = os.path.dirname(nc_file)
        nc_dir = os.path.basename(nc_dir)
        output_dir = EDR_out_dir + "\\" + nc_dir
        if os.path.exists(output_dir):
            edr_cloud_mask2geotiff(nc_file, EDR_names, output_dir)
        else:
            os.mkdir(output_dir)
            edr_cloud_mask2geotiff(nc_file, EDR_names, output_dir)

if __name__ == "__main__":
    print("start")
    input_edr_dir = r"F:\Data\US_Hurricane\VCM_2021"  # folder to restore the vcm nc file
    output_edr_dir = r"F:\Data\US_Hurricane\newVCM_2021_Geotiff"  # output folder
    # if not os.path.exists(output_edr_dir):
      #   os.mkdir(output_edr_dir)
    batch_pro(input_edr_dir, output_edr_dir)
    print("Complete")
