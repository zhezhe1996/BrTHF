import os
import calendar

import numpy as np
import pandas as pd

import rasterio
import xarray as xr

import zipfile

def create_zip(files, output_zip):
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files: 
                zipf.write(file, arcname=os.path.basename(file)) 
                print(f"文件已成功压缩为：{output_zip}")
    except Exception as e:
        print(f"压缩失败: {e}")

product_name = 'BrTHF'
path = r'.\data\tif'

for year in range(1993,2018):
    for month in range(1,13):

        print(year,month)
        days = calendar.monthrange(year,month)[1]
        
        SHF_list = []
        LHF_list = []
        time_list = []
        
        for day in np.arange(days): 
            # try:   
            year_str = str(year)
            month_str = str(month).rjust(2,'0')
            day_str = str(day+1).rjust(2,'0')
            date_str = year_str+month_str+day_str
            sclae_type = 'Daily'

            variable_name = 'SHF'
            read_temp_path = os.path.join(path,product_name+' Conversion',variable_name,year_str,month_str)
            filename = product_name+'_'+variable_name+'_Daily.A'+date_str+'.tif'
            read_file_path = os.path.join(read_temp_path,filename)
            src = rasterio.open(read_file_path)
            SHF = src.read(1)
    
            variable_name = 'LHF'
            read_temp_path = os.path.join(path,product_name+' Conversion',variable_name,year_str,month_str)
            filename = product_name+'_'+variable_name+'_Daily.A'+date_str+'.tif'
            read_file_path = os.path.join(read_temp_path,filename)
            src = rasterio.open(read_file_path)
            LHF = src.read(1)

            SHF_list.append(SHF[::-1])
            LHF_list.append(LHF[::-1])
            time_list.append(date_str)

            beta = SHF/LHF
            
        SHF_array = np.array(SHF_list)
        LHF_array = np.array(LHF_list)

        dates = pd.to_datetime(time_list)
        epoch_time = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        time_values = np.array(epoch_time)

        transform,crs,bounds,width,height = src.transform,src.crs,src.bounds,src.width,src.height

        lon = np.linspace(bounds[0], bounds[2], width)
        lat = np.linspace(bounds[1], bounds[3], height)

        time = xr.DataArray(time_values, dims=["time"], coords={"time": time_values})
        lon = xr.DataArray(lon, dims=["lon"], coords={"lon": lon})
        lat = xr.DataArray(lat, dims=["lat"], coords={"lat": lat})
        
        time.attrs["time"] = "time"            
        time.attrs["units"] = "seconds since 1970-01-01"
        time.attrs["calendar"] = "Gregorian"
        time.attrs["valid_min"] = str(np.nanmin(time_values))
        time.attrs["valid_max"] = str(np.nanmax(time_values))

        lon.attrs["_FillValue"] = np.nan
        lon.attrs["axis"] = "X"
        lon.attrs["standard_name"] = "longitude"
        lon.attrs["units"] = "degrees_east"
        lon.attrs["valid_min"] = str(np.nanmin(lon))
        lon.attrs["valid_max"] = str(np.nanmax(lon))

        lat.attrs["_FillValue"] = np.nan
        lat.attrs["axis"] = "X"
        lat.attrs["standard_name"] = "longitude"
        lat.attrs["units"] = "degrees_east"
        lat.attrs["valid_min"] = str(np.nanmin(lat))
        lat.attrs["valid_max"] = str(np.nanmax(lat))

        SHF_array = SHF_array.astype(float)
        LHF_array = LHF_array.astype(float)

        SHF_array = np.round(SHF_array,2)
        LHF_array = np.round(LHF_array,2)
        
        beta_array = np.round(SHF_array/LHF_array,2)              

        SHF_array = SHF_array*100
        LHF_array = LHF_array*100

        beta_array = SHF_array/LHF_array
        
        SHF_array = np.nan_to_num(SHF_array,nan = -32768)
        LHF_array = np.nan_to_num(LHF_array,nan = -32768)
        
        SHF_array = np.round(SHF_array,0)
        LHF_array = np.round(LHF_array,0)
        
        SHF_array = SHF_array.astype(int)
        LHF_array = LHF_array.astype(int)   

        ds = xr.Dataset(
            {"SHF": (["time", "lat", "lon"], SHF_array.reshape(-1,height,width)),
             "LHF": (["time", "lat", "lon"], LHF_array.reshape(-1,height,width))},
            coords={"time": time, "lat": lat, "lon": lon},
        )
        
        ds.attrs['Title'] = r'Global dataset of air-sea turbulent heat fluxes (sensible heat flux and latent heat flux) (1993–2017)'
        ds.attrs['Summary'] = r'The air-sea turbulent heat fluxes include the sensible heat flux and latent heat flux between the ocean and the atmosphere. These fluxes are important components of the energy and water exchange between the ocean and the atmosphere and are crucial for understanding the air-sea interactions. This dataset was developed using a Bowen ratio-constrained air-sea turbulent heat fluxes model using neural network technique. When compared with observations collected from 197 buoys, the model shows good consistency, with root mean square errors of 6.05 W/m² for sensible heat flux, 23.67 W/m² for latent heat flux, and 0.22 for Bowen ratio. The correlation coefficients are 0.93, 0.91, and 0.25, respectively, outperforming the seven comparison products. This dataset can help improve our understanding of the air-sea interactions and provide more accurate quantification of the water and energy budgets between the ocean and the atmosphere. The dataset contains daily global air-sea sensible heat flux and latent heat flux (in units of W/m²) at a 0.25° spatial resolution for the period 1993-2017.'
        ds.attrs['Usage'] = r'The dataset is stored in NetCDF (.nc) format. The file name follows the format BrTHF_Monthly.Ayyyymm.nc, where yyyy represents the year and mm represents the month. For example, the file BrTHF_Monthly.A199301.nc contains the daily distribution of global air-sea turbulent heat fluxes (sensible heat flux and latent heat flux) from January 1 to January 31, 1993. The files can be opened using ArcGIS or read programmatically through Python and other programming languages. The data is in units of W/m², with an integer data type and a scaling factor of 0.01.'
        ds.attrs['Temporal resolution'] = r'Daily'
        ds.attrs['Spatial resolution'] = r'0.25°'
        ds.attrs['Data time range'] = r'from 1993-01-01 to 2017-12-31'        
        ds.attrs['Reference of data'] = r'Tang, R., Wang, Y. (2025). Global dataset of air-sea turbulent heat fluxes (sensible heat flux and latent heat flux) (1993–2017). National Tibetan Plateau / Third Pole Environment Data Center. https://doi.org/10.11888/Atmos.tpdc.302578. https://cstr.cn/18406.11.Atmos.tpdc.302578.'
        ds.attrs['Article citation'] = r'Wang, Y., Tang, R., Huang, L., Liu, M., Jiang, Y., & Li, Z.-L. (2024). A Bowen ratio-informed method for coordinating the estimates of air–sea turbulent heat fluxes. Environmental Research Letters, *19*, Article 104020. https://doi.org/10.1088/1748-9326/ad9341.'
        ds.attrs['License agreement'] = r'CC BY 4.0'   
        
        ds['SHF'].attrs['_FillValue'] = -32768
        ds['SHF'].attrs['standard_name'] = 'mean surface sensible heat flux'
        ds['SHF'].attrs['units'] = 'W m-2'
        ds['SHF'].attrs['positive'] = 'up'
        ds['SHF'].attrs['scale_factor'] = 0.01
        ds['SHF'].attrs['add_offset'] = 0.0
        
        ds['LHF'].attrs['_FillValue'] = -32768
        ds['LHF'].attrs['standard_name'] = 'mean surface latent heat flux'
        ds['LHF'].attrs['units'] = 'W m-2'
        ds['LHF'].attrs['positive'] = 'up'
        ds['LHF'].attrs['scale_factor'] = 0.01
        ds['LHF'].attrs['add_offset'] = 0.0
        
        # 设置压缩参数
        encoding = {
            "SHF": {
                    "zlib": True,  
                    "complevel": 7,  
                    "chunksizes": (len(SHF_array), 180, 360),  
                
            },
            "LHF": {
                "zlib": True,  
                "complevel": 7,
                "chunksizes": (len(LHF_array), 180, 360),
                
            }
        }

        save_path = os.path.join(r'.\data\nc',
                                 'BrTHF_Monthly.A'+year_str+month_str+'.nc')
        ds.to_netcdf(save_path,encoding=encoding,mode="w")


read_path = r'.\data\nc'
files = os.listdir(read_path)

for year in range(1993,2018):
    year_str = str(year)

    files_selected = [os.path.join(read_path,file) for file in files if 'A'+year_str in file]

    output_zip = os.path.join(".\data\zip",year_str+'.zip')
    create_zip(files_selected, output_zip)