import sys
import snappy
import h5py
import numpy as np
import csv
import os
import glob
import pandas as pd


from snappy import ProductIO
from snappy import jpy
from snappy import GPF
from snappy import Product

snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
HashMap = jpy.get_type('java.util.HashMap')
Coord = jpy.get_type('org.esa.snap.pixex.Coordinate')

snow_dataset=pd.read_csv("/home/simsek/workspace/EDUCATION/Thesis/Scripts/turku_SnowCover_insitu.csv", names=["year","month","day","lat","lon","sd"])
nosnow_dataset=pd.read_csv("/home/simsek/workspace/EDUCATION/Thesis/Scripts/turku_NoSnow_insitu.csv", names=["year","month","day","lat","lon","sd"])
snow_coords=np.asarray(nosnow_dataset[["lat","lon"]].drop_duplicates())
n=np.asarray(nosnow_dataset[["lat","lon"]].drop_duplicates()).shape[0]


## Parameters for resampling
resampling_parameters = HashMap()
resampling_parameters.put('targetResolution', 60)
## Parameters for Pixel extraction
insitu_coordinates = jpy.array('org.esa.snap.pixex.Coordinate',n)
for i in range(n):
    insitu_coordinates[i] = Coord('station'+str(i), snow_coords[i,0], snow_coords[i,1], None)
#insitu_coordinates[0] = Coord('station0', 61.00, 24.5, None)
# insitu_coordinates[1] = Coord('station1', 60.82, 23.5, None)
# insitu_coordinates[2] = Coord('station2', 60.45, 23.65, None)
# insitu_coordinates[3] = Coord('station3', 60.65, 23.81, None)
# insitu_coordinates[4] = Coord('station4', 60.52, 24.65, None)
# insitu_coordinates[5] = Coord('station5', 61.12, 24.33, None)
# insitu_coordinates[6] = Coord('station6', 60.37, 23.12, None)

print(insitu_coordinates[8])
pixex_parameters = HashMap()
pixex_parameters.put('PexportBands', 1)
pixex_parameters.put('PexportExpressionResult', 0)
pixex_parameters.put('PexportMasks', 0)
pixex_parameters.put('PexportTiePoints', 0)
pixex_parameters.put('coordinates' ,insitu_coordinates)
pixex_parameters.put('PoutputDir', '/home/simsek/workspace/EDUCATION/Thesis/Scripts/')


##### In-situ datayi okumak icin bir seyler
# with open('turku_list_averaged.csv') as csvfile:
#     turku_insitu_data = csv.reader(csvfile, delimiter=',')
#     lat=[]
#     lon=[]
#     averageSD=[]
#     for col in  turku_insitu_data:
#         lat=col[3]
#         lon=col[4]
#         averageSD=col[5]

##Buraya file path list seklinde ** kullanarak tum klasoru at
#file_path="/media/simsek/1CB4C75FB4C73A52/SummerWork2/SentinelDataForThesis&Validation_Oct&May/2A/S2A_MSIL2A_20171005T095031_N0205_R079_T34VFN_20171005T095027.SAFE"

file_path="/media/simsek/1CB4C75FB4C73A52/SummerWork2/SentinelDataForThesis&Validation_Oct&May/2A/"
##"./ EKLE bulundugun yerden calissin"


for filename in os.listdir(file_path):
    if filename.endswith(".SAFE"):
        product_name = os.path.join(file_path,filename)
        product = ProductIO.readProduct(product_name)
        ## resampling to 60 m to be able to use pixel extraction operator
        print(product_name)
        resampled_product = GPF.createProduct('Resample', resampling_parameters, product)
        ## Pixel extraction
        result = GPF.createProduct('PixEx', pixex_parameters, resampled_product)
        ## Renaming the last txt file to data date
        list_of_text_files=glob.glob('/home/simsek/workspace/EDUCATION/Thesis/Scripts/*.txt')
        latest_text_file = max(list_of_text_files, key=os.path.getctime)
        os.rename(latest_text_file,filename.split("_")[2].split("T")[0]+"_Sentinel_insitu_pixels_snow.csv" )
    else:
        print("There is no Sentinel-2 data in the given folder")
