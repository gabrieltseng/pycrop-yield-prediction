# Data

This directory stores useful files, and is the location to which Earth Engine Exports should be saved.

## Contents:

### 1. [`yield_data.csv`](yield_data.csv)

This is the yield data from the USDA website, which measures soybean yields in bushels per acre. The data can be reproduced
at the following [link](https://quickstats.nass.usda.gov/#5A65CCEF-B75F-366D-AA20-5632E0073EA1).

Click on `Get Data`, and then `Spreadsheet`, to generate a copy of the csv file. This file is analogous to
[`yield_final.csv`](https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/2%20clean%20data/yield_final.csv) in
the original repository.

### 2. [`county_data.csv`](county_data.csv)

This is the county data from a US 2010 Census [fusion table](https://support.google.com/fusiontables/answer/2571232), 
which we also use to delineate the county borders when exporting the MODIS data from the Earth Engine. It is used for 
its latitude and longitude data, which allows the distance between counties to be approximated. The data can be 
reproduced at the following [link](https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1).

Note that fusion tables will [not be available after December 3rd 2019](https://support.google.com/fusiontables/answer/9185417).

Click on `File`, and then `Download`, and download all rows as a CSV file.

### 3. [`counties.svg`](counties.svg)

A map of the U.S., with county delineations. Taken from the 
[original repository](https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/6%20result_analysis/counties.svg), 
and useful for analysis.

### 4. MODIS datasets<a name="MODIS"></a>

The satellite data used comes from the Moderate Resolution Imaging Spectroradiometer 
([MODIS](https://en.wikipedia.org/wiki/Moderate_Resolution_Imaging_Spectroradiometer)), aboard the [Terra](https://en.wikipedia.org/wiki/Terra_(satellite))
satellite.

Once the data has been exported to Google Drive, it can be downloaded to this folder.

Specifically, the following datasets are used:

#### 4.a [MOD09A1: Surface Reflectance](https://modis.gsfc.nasa.gov/data/dataprod/mod09.php)

"An estimate of the surface spectral reflectance of Terra MODIS bands 1 through 7".

Basically, an 'image' of the county as seen from the satellite.

#### 4.b [MCD12Q1: Land Cover Type](https://modis.gsfc.nasa.gov/data/dataprod/mod12.php)

Labels the data according to one of five global land cover classification systems. This is used as a mask, because we only
want to consider pixels associated with farmland.

#### 4.c [MYD11A2: Aqua/Land Surface Temperature](https://modis.gsfc.nasa.gov/data/dataprod/mod11.php)

Two more bands which can be used as input data to our models.
