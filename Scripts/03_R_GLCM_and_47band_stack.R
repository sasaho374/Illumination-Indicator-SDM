# =============================================================================
# SENTINEL-2 TEXTURE EXTRACTION AND 47-BAND STACK ASSEMBLY
# Hezarmasjed Mountains | Supplementary reproducibility workflow
# =============================================================================
# INPUT: 24-band hybrid composite exported by 01_GEE_Sentinel2_preprocessing.js
# OUTPUT: 24-band GLCM GeoTIFF, 47-band combined GeoTIFF, band inventory, QC
#
# Texture settings are taken from the historical 'final' R GLCM script:
#   sources B5, B8, B11, NDVI; six statistics; 3x3 window; shift (1,1);
#   na_opt='ignore'; na_val=NA. No reflectance scaling or quantization is added.
#
# SCL handling follows manuscript section 2.4: retain SCL 2,4,5 only;
#   SCL 2 -> ShadowMask 1; SCL 4/5 -> ShadowMask 0.
# SCL is kept in the GeoTIFF for provenance, but is NOT a model predictor.
# Thus 47 raster bands = 46 candidate predictors + the SCL QC band.
#
# CAUTION: This is a reconstructed workflow, not yet verified as the exact
# historical workflow. In particular, GEE's seasonal median of categorical
# SCL values can produce fractional values. Those pixels are reported and
# excluded here, NOT rounded or silently reclassified. Compare the output
# against the original modelling raster and sampled training-point data.
# =============================================================================

suppressPackageStartupMessages({
  library(raster)
  library(glcm)
})

# ---- 1. Files and computational settings -----------------------------------
project_dir <- normalizePath('.', winslash = '/', mustWork = TRUE)
data_dir <- file.path(project_dir, 'data')
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

gee_file <- file.path(data_dir, 'hybrid_final_quality_Nov6_Apr4_with_DEM.tif')
texture_file <- file.path(data_dir, 'texture_features_all.tif')
stack_file <- file.path(data_dir, 'predictor_stack_47bands.tif')
bands_csv <- file.path(data_dir, 'predictor_stack_47bands_band_inventory.csv')
qc_csv <- file.path(data_dir, 'predictor_stack_47bands_SCL_QC.csv')

if (!file.exists(gee_file)) stop('Missing GEE export: ', gee_file)

gee_names <- c(
  'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
  'SCL','NDVI','GNDVI','NDMI','NDWI','NBR',
  'seasonCov','sourceSeason','DEM','Slope','Aspect','TPI','TRI','TWI'
)
texture_sources <- c('B5', 'B8', 'B11', 'NDVI')
texture_metrics <- c(
  'contrast', 'entropy', 'homogeneity', 'variance',
  'correlation', 'dissimilarity'
)
texture_names <- unlist(lapply(texture_sources, function(x) {
  paste(x, texture_metrics, sep = '_')
}), use.names = FALSE)

# ---- 2. Read the GEE export -------------------------------------------------
gee <- raster::stack(gee_file)
if (raster::nlayers(gee) != 24L) {
  stop('GEE composite must contain 24 bands; found ', raster::nlayers(gee))
}
# GEE GeoTIFF band descriptions are not always imported as R layer names.
# The order below MUST match the GEE export's finalBandList.
original_names <- names(gee)
names(gee) <- gee_names
cat('GEE bands: ', paste(names(gee), collapse = ', '), '\n')

# ---- 3. Recreate the original 24 GLCM layers -------------------------------
# Using the exact parameter choices in the historical final GLCM block.
# Processing may take substantial time/disk space for a full 10-m landscape.
texture_layers <- vector('list', length(texture_names))
k <- 0L
for (source_name in texture_sources) {
  src <- gee[[source_name]]
  for (metric in texture_metrics) {
    k <- k + 1L
    layer_name <- paste(source_name, metric, sep = '_')
    message('GLCM ', k, '/24: ', layer_name)
    tx <- glcm::glcm(
      src,
      window = c(3, 3),
      statistics = metric,
      shift = c(1, 1),
      na_opt = 'ignore',
      na_val = NA
    )
    names(tx) <- layer_name
    texture_layers[[k]] <- tx
  }
}
texture <- raster::stack(texture_layers)
stopifnot(identical(names(texture), texture_names))

raster::writeRaster(
  texture, texture_file, format = 'GTiff',
  overwrite = TRUE, options = c('COMPRESS=LZW', 'TILED=YES')
)
rm(texture_layers, texture)
gc()
texture <- raster::stack(texture_file)
names(texture) <- texture_names

# ---- 4. Validate alignment --------------------------------------------------
if (!isTRUE(raster::compareRaster(
  gee, texture, extent = TRUE, rowcol = TRUE, crs = TRUE,
  res = TRUE, orig = TRUE, rotation = TRUE, stopiffalse = FALSE
))) stop('GEE and texture rasters have mismatched spatial grids.')

# ---- 5. Derive illumination regime and valid terrestrial mask --------------
scl <- gee[['SCL']]
# The manuscript retains only classes 2, 4 and 5. Do NOT round a
# categorical layer that may have been median-composited in GEE.
valid_land <- raster::calc(scl, fun = function(x) {
  ifelse(!is.na(x) & x %in% c(2, 4, 5), 1, NA_real_)
})
shadow <- raster::calc(scl, fun = function(x) {
  ifelse(is.na(x), NA_real_, ifelse(x == 2, 1, ifelse(x %in% c(4,5), 0, NA_real_)))
})
names(shadow) <- 'ShadowMask'

# ---- 6. Assemble in the precise 47-band order supplied by the authors -----
gee_without_controls <- raster::dropLayer(gee, c('seasonCov','sourceSeason'))
combined <- raster::stack(gee_without_controls, texture, shadow)
expected <- c(gee_names[!gee_names %in% c('seasonCov','sourceSeason')],
              texture_names, 'ShadowMask')
if (raster::nlayers(combined) != 47L || !identical(names(combined), expected)) {
  stop('Unexpected band count or order in assembled stack.')
}
# Keep terrain-shadow pixels intact in all bands; remove only invalid SCL.
final <- raster::mask(combined, valid_land)

# ---- 7. Write results -------------------------------------------------------
raster::writeRaster(
  final, stack_file, format = 'GTiff', overwrite = TRUE,
  options = c('COMPRESS=LZW', 'TILED=YES')
)
write.csv(
  data.frame(band_number = seq_along(expected), band_name = expected,
             role = ifelse(expected == 'SCL', 'QC_only',
                           ifelse(expected == 'ShadowMask', 'illumination_indicator',
                                  'environmental_predictor'))),
  bands_csv, row.names = FALSE
)

# ---- 8. Report SCL classes and the resulting shadow classification ---------
# Frequency is calculated without reading the entire raster into an R vector.
scl_freq <- raster::freq(scl, useNA = 'ifany')
shadow_freq <- raster::freq(final[['ShadowMask']], useNA = 'ifany')
qc <- data.frame(
  category = c(rep('input_SCL', nrow(scl_freq)),
               rep('output_ShadowMask', nrow(shadow_freq))),
  value = c(scl_freq[,1], shadow_freq[,1]),
  count = c(scl_freq[,2], shadow_freq[,2])
)
write.csv(qc, qc_csv, row.names = FALSE)
print(qc)
cat('\nWrote:', texture_file, '\n', stack_file, '\n', bands_csv,
    '\n', qc_csv, '\n')
cat('Package versions: raster=', as.character(packageVersion('raster')),
    '; glcm=', as.character(packageVersion('glcm')), '\n')
cat('IMPORTANT: Compare raster values at training/test points against the',
    ' historical modelling data before claiming exact reproducibility.\n')
