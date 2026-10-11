# ======================================================================
# Residual spatial-range diagnostic
#
# This script reproduces the residual variogram analysis used to
# characterize spatial dependence before spatial partitioning.
#
# ======================================================================


# ============================================================
# 1. Required packages
# ============================================================

required_packages <- c(
  "sf",
  "terra",
  "dplyr",
  "gstat"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    quietly = TRUE,
    FUN.VALUE = logical(1)
  )
]

if(length(missing_packages) > 0){

  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(sf)
  library(terra)
  library(dplyr)
  library(gstat)
})


# ============================================================
# 2. Project paths
# ============================================================

PROJECT_DIR <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

DATA_DIR <- file.path(
  PROJECT_DIR,
  "data"
)

OUTPUT_DIR <- file.path(
  PROJECT_DIR,
  "outputs_publication",
  "spatial_partition"
)

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

MASTER_PATH <- file.path(
  DATA_DIR,
  "master_points.gpkg"
)

RASTER_PATH <- file.path(
  DATA_DIR,
  "predictor_stack_46bands.tif"
)

stopifnot(
  file.exists(MASTER_PATH),
  file.exists(RASTER_PATH)
)


# ============================================================
# 3. Read data
# ============================================================

pts <- sf::st_read(
  MASTER_PATH,
  quiet = TRUE
)

preds <- terra::rast(
  RASTER_PATH
)

stopifnot(
  sf::st_crs(pts)$epsg ==
    terra::crs(
      preds,
      describe = TRUE
    )$code
)


# ============================================================
# 4. Select predictors
# ============================================================

preds_simple <- preds[[
  c(
    "NDVI",
    "DEM",
    "TWI"
  )
]]


# ============================================================
# 5. Extract predictors at master points
# ============================================================

v <- terra::vect(
  pts
)

x <- terra::extract(
  preds_simple,
  v
)

df <- dplyr::bind_cols(
  sf::st_drop_geometry(pts),
  x[, -1, drop = FALSE]
)

df <- na.omit(df)

cat(
  "Rows used in GLM:",
  nrow(df),
  "\n"
)


# ============================================================
# 6. Simple GLM
# ============================================================

glm_simple <- glm(
  presence ~ NDVI + DEM + TWI,
  data = df,
  family = binomial
)

print(
  summary(glm_simple)
)


# ============================================================
# 7. Pearson residuals
# ============================================================

df$resid <- residuals(
  glm_simple,
  type = "pearson"
)


# ============================================================
# 8. Reattach residuals to geometry
# ============================================================

pts_used <- pts[
  as.integer(
    rownames(df)
  ),
]

pts_used$resid <- df$resid

sp_res <- as(
  pts_used,
  "Spatial"
)


# ============================================================
# 9. Empirical variogram
# ============================================================

vg_emp <- gstat::variogram(
  resid ~ 1,
  data = sp_res,
  cutoff = 15000,
  width = 500
)

vg_fit <- gstat::fit.variogram(
  vg_emp,
  gstat::vgm(
    psill = 1,
    model = "Exp",
    range = 5000,
    nugget = 0.1
  )
)

print(
  vg_fit
)


# ============================================================
# 10. Extract spatial range
# ============================================================

range_m <- vg_fit$range[2]

range_km <- range_m / 1000

cat(
  "\nResidual spatial range:",
  round(range_km, 2),
  "km\n"
)


# ============================================================
# 11. Save diagnostic outputs
# ============================================================

write.csv(
  as.data.frame(vg_fit),
  file.path(
    OUTPUT_DIR,
    "residual_variogram_model.csv"
  ),
  row.names = FALSE
)

png(
  file.path(
    OUTPUT_DIR,
    "residual_variogram.png"
  ),
  width = 1800,
  height = 1400,
  res = 200
)

plot(
  vg_emp,
  vg_fit,
  main = paste0(
    "Residual variogram (GLM simple)\n",
    "Range = ",
    round(range_km, 2),
    " km"
  )
)

dev.off()

cat(
  "\nSpatial-range diagnostic completed successfully.\n"
)