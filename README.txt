Evaluating a remotely sensed illumination regime as an ecological indicator in species distribution models
====================================================================================================

Repository: https://github.com/sasaho374/Illumination-Indicator-SDM

OVERVIEW
--------
This repository provides Google Earth Engine (GEE) JavaScript and R scripts for the remote-sensing, spatial, and species distribution modelling analyses of Juniperus polycarpos var. turcomanica in the Hezarmasjed Mountains, northeastern Iran.

The workflow covers seasonal Sentinel-2 preprocessing, SCL=2 temporal recurrence, GLCM texture extraction, 47-band raster assembly, shadow–topography analysis, residual variogram analysis, spatial blocking, and the main species distribution models (SDMs).

REPOSITORY STRUCTURE
--------------------
Scripts/
  01_GEE_Sentinel2_preprocessing.js
  02_GEE_SCL2_temporal_stability_validation.js
  03_R_GLCM_and_47band_stack.R
  04_Shadow vs Topography_analysis.R
  05_Variogram.R
  06_blockcv.R
  07_main analysis2.R

data/
  training_points.gpkg       (included; 486 observations)
  external_test_points.gpkg  (included; 94 observations)
  master_points.gpkg         (included; 580 observations)

These three GeoPackage datasets are distributed in the repository.
training_points.gpkg and external_test_points.gpkg provide the fixed
analytical development and external-test partition used by script 07.
master_points.gpkg contains the 580 reference observations used for
spatial diagnostics and the spatial-blocking workflow.

The following GeoTIFFs are generated during the workflow and are not stored in this repository:
  data/hybrid_final_quality_Nov6_Apr4_with_DEM.tif
  data/texture_features_all.tif
  data/predictor_stack_47bands.tif

SCRIPT DESCRIPTIONS
-------------------
01_GEE_Sentinel2_preprocessing.js
  Produces seasonal Sentinel-2 composites for seven November–April periods
  (2018–2019 through 2024–2025), spectral indices, topographic layers,
  and the final 24-band hybrid GeoTIFF.

02_GEE_SCL2_temporal_stability_validation.js
  Evaluates temporal recurrence of the SCL=2 classification. Results are
  displayed in the GEE console; this is a separate validation analysis.

03_R_GLCM_and_47band_stack.R
  Calculates 24 GLCM texture layers from B5, B8, B11, and NDVI and assembles
  the 47-band GeoTIFF. Also exports the band inventory and SCL summaries.

04_Shadow vs Topography_analysis.R
  Evaluates topographic and spatial associations of the binary illumination
  indicator, including spatial modelling and diagnostic outputs.

05_Variogram.R
  Fits a simple binomial GLM and calculates an empirical residual variogram.

06_blockcv.R
  Generates spatial blocks, summarizes fold composition, and exports a
  spatial train/test partition for documentation.

07_main analysis2.R
  Runs the main SDM analyses using the analytical training and external-test
  GeoPackages and the 47-band predictor raster.

SOFTWARE REQUIREMENTS
---------------------
- Google Earth Engine Code Editor for scripts 01 and 02.
- R for scripts 03–07.
- R packages: raster, glcm, sf, terra, blockCV, dplyr, ggplot2,
  data.table, tidyr, tibble, readr, purrr, stringr, ranger, pROC,
  glmnet, mgcv, spdep, and gstat.

Install the R packages using:

install.packages(c(
  'raster', 'glcm', 'sf', 'terra', 'blockCV', 'dplyr', 'ggplot2',
  'data.table', 'tidyr', 'tibble', 'readr', 'purrr', 'stringr',
  'ranger', 'pROC', 'glmnet', 'mgcv', 'spdep', 'gstat'
))

WORKFLOW
--------
1. Run 01_GEE_Sentinel2_preprocessing.js in the GEE Code Editor. Export and
   download the 24-band hybrid GeoTIFF. Place it in the repository's data/
   directory with this filename:

   data/hybrid_final_quality_Nov6_Apr4_with_DEM.tif

   The hybrid includes two processing-control bands, seasonCov and
   sourceSeason, which are omitted from the assembled predictor raster.

2. Run 02_GEE_SCL2_temporal_stability_validation.js in GEE to obtain the
   temporal-recurrence results, if these validation results are required.

3. In R, set the working directory to the repository root and run:

   source('Scripts/03_R_GLCM_and_47band_stack.R')

   This creates:
   data/texture_features_all.tif
   data/predictor_stack_47bands.tif
   data/predictor_stack_47bands_band_inventory.csv
   data/predictor_stack_47bands_SCL_QC.csv

4. The following analytical GeoPackages are already provided in the
   repository under data/; retain these filenames and locations:

   data/training_points.gpkg       (486 observations)
   data/external_test_points.gpkg  (94 observations)
   data/master_points.gpkg         (580 observations)

5. Run the supporting analyses and main SDM analysis from the repository root:

   source('Scripts/04_Shadow vs Topography_analysis.R')
   source('Scripts/05_Variogram.R')
   source('Scripts/06_blockcv.R')
   source('Scripts/07_main analysis2.R')

   Scripts 04–06 are supporting analyses. Script 07 reads the analytical
   training_points.gpkg and external_test_points.gpkg directly. Script 06
   writes its separately generated partition files with '_generated' in
   their filenames.

47-BAND RASTER CONTENT
----------------------
Band 01–10: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
Band 11:    SCL
Band 12–16: NDVI, GNDVI, NDMI, NDWI, NBR
Band 17–22: DEM, Slope, Aspect, TPI, TRI, TWI
Band 23–28: B5_contrast, B5_entropy, B5_homogeneity,
            B5_variance, B5_correlation, B5_dissimilarity
Band 29–34: B8_contrast, B8_entropy, B8_homogeneity,
            B8_variance, B8_correlation, B8_dissimilarity
Band 35–40: B11_contrast, B11_entropy, B11_homogeneity,
            B11_variance, B11_correlation, B11_dissimilarity
Band 41–46: NDVI_contrast, NDVI_entropy, NDVI_homogeneity,
            NDVI_variance, NDVI_correlation, NDVI_dissimilarity
Band 47:    ShadowMask

The SCL layer is retained for data provenance and processing. It is not
included as a model predictor. The RAW configuration contains 45
non-illumination predictors; RAW+SHADOW adds the binary illumination
indicator, for a total of 46 predictors.

TEXTURE AND ILLUMINATION PROCESSING
-----------------------------------
GLCM features are calculated with the R glcm package using a 3 x 3 moving
window, shift (1,1), and six statistics: contrast, entropy, homogeneity,
variance, correlation, and dissimilarity. The script specifies
na_opt='ignore' and na_val=NA.

In the 47-band assembly script, pixels with SCL=2 are assigned
ShadowMask=1, while pixels with SCL=4 or SCL=5 are assigned ShadowMask=0.
Other SCL values are excluded from the assembled raster. The SCL=2
pixels remain represented in the spectral and environmental layers.

MAIN SDM DESIGN
---------------
The main analysis compares three training-sample schemes:
  UR: shadow representation follows landscape prevalence.
  SB: shadow representation follows fitting-pool prevalence.
  SW: the SB observations are weighted toward landscape shadow prevalence.

Within each scheme, RAW and RAW+SHADOW predictor configurations are
compared across the modelling algorithms. A separate calibration subset
is used for threshold selection, and the held-out Fold 2 dataset is used
for external evaluation. The analysis specifies 30 stochastic replicates
and a random seed of 123.

SPATIAL PARTITION
-----------------
The spatial blocking script uses a 2,500 m block size, six folds,
random seed 42, and Fold 2 as the external test fold. The main SDM
script reads the supplied analytical training and external-test files;
newly generated spatial-partition files are saved separately.

DATA AND CODE AVAILABILITY
--------------------------
The analysis scripts are provided in this repository. The large
47-band predictor GeoTIFF is generated from the GEE export using
script 03 rather than distributed directly through GitHub. The three GeoPackage point datasets (master, training, and external
test) are included in the data/ directory and used directly by the
corresponding R scripts.

