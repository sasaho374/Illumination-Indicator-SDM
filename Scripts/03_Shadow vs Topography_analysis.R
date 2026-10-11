# ======================================================================
# Persistent shadow–topography analysis
#
# This script evaluates the topographic and spatial structure of
# persistent low-illumination areas derived from Sentinel-2 imagery.
#
# Analyses include:
#   - topographic predictor derivation
#   - descriptive and correlation diagnostics
#   - binomial GAM/BAM models
#   - spatial block cross-validation
#   - drop-one predictor contribution analysis
#   - predictor-block deviance partitioning
#   - residual spatial autocorrelation
#   - full-region prediction
#
# Persistent shadow is treated as a terrain-associated remotely sensed
# illumination indicator rather than as an independent causal driver.
# ======================================================================


# =============================================================================
# 0) USER SETTINGS
# =============================================================================

# Input raster stack must contain DEM, Aspect, and ShadowMask.

PROJECT_DIR <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

DATA_DIR <- file.path(PROJECT_DIR, "data")

PRED_RASTER_PATH <- file.path(
  DATA_DIR,
  "predictor_stack_46bands.tif"
)

OUT_DIR <- file.path(
  PROJECT_DIR,
  "outputs_publication",
  "shadow_topography"
)

# Number of pixels drawn regularly across the whole valid raster.
# 100,000–250,000 is generally sufficient.
N_SAMPLE <- 200000L

# Maximum number of rows used in spatial CV.
# Set equal to N_SAMPLE to use all sampled rows.
CV_MAX_N <- 150000L

# Spatial block width in map units.
# For projected CRS in metres, 5000 = 5 km.
BLOCK_SIZE <- 5000

# Number of spatial folds.
N_FOLDS <- 5L

# Number of nearest neighbours used for residual Moran's I.
MORAN_K <- 8L

# Maximum residual sample used for Moran's I.
MORAN_N <- 5000L

# BAM basis dimensions.
K_UNIVARIATE <- 8L
K_SPATIAL <- 100L

# Number of processing threads.
N_THREADS <- max(1L, parallel::detectCores(logical = TRUE) - 1L)

# Refit controls.
# FALSE means: reuse a valid saved object when available.
FORCE_REBUILD_RASTERS <- FALSE
FORCE_RESAMPLE <- FALSE
FORCE_REFIT_MODELS <- FALSE
FORCE_REFIT_DROP_ONE <- FALSE
FORCE_REFIT_CV <- FALSE
FORCE_REPREDICT_RASTERS <- FALSE

SEED <- 123L
set.seed(SEED)


# =============================================================================
# 1) PACKAGES
# =============================================================================

required_packages <- c(
  "terra",
  "data.table",
  "dplyr",
  "tidyr",
  "mgcv",
  "ggplot2",
  "pROC",
  "spdep"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    "Install the following packages before running the script:\n",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(terra)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(mgcv)
  library(ggplot2)
  library(pROC)
  library(spdep)
})


# =============================================================================
# 2) OUTPUT STRUCTURE AND LOGGING
# =============================================================================

FIG_DIR <- file.path(OUT_DIR, "figures")
TAB_DIR <- file.path(OUT_DIR, "tables")
MOD_DIR <- file.path(OUT_DIR, "models")
RAS_DIR <- file.path(OUT_DIR, "rasters")
CHK_DIR <- file.path(OUT_DIR, "checkpoints")
LOG_DIR <- file.path(OUT_DIR, "logs")

for (d in c(
  OUT_DIR, FIG_DIR, TAB_DIR, MOD_DIR, RAS_DIR, CHK_DIR, LOG_DIR
)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

LOG_FILE <- file.path(
  LOG_DIR,
  paste0("run_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt")
)

log_message <- function(...) {
  text <- paste0(..., collapse = "")
  line <- paste0(
    "[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
    text
  )
  cat(line, "\n")
  cat(line, "\n", file = LOG_FILE, append = TRUE)
}

safe_save_rds <- function(object, path) {
  temp_path <- paste0(path, ".tmp")
  saveRDS(object, temp_path)
  if (file.exists(path)) {
    file.remove(path)
  }
  ok <- file.rename(temp_path, path)
  if (!ok) {
    stop("Could not safely write checkpoint: ", path)
  }
  invisible(path)
}

load_or_run <- function(path, force = FALSE, fun) {
  if (!force && file.exists(path)) {
    log_message("Recovering checkpoint: ", path)
    return(readRDS(path))
  }

  log_message("Creating checkpoint: ", path)
  object <- fun()
  safe_save_rds(object, path)
  object
}

capture_to_file <- function(expr, file) {
  output <- capture.output(force(expr))
  writeLines(output, con = file)
  invisible(output)
}

options(
  warn = 1,
  stringsAsFactors = FALSE,
  terra.progress = 1
)

config <- list(
  input = PRED_RASTER_PATH,
  output = OUT_DIR,
  n_sample = N_SAMPLE,
  cv_max_n = CV_MAX_N,
  block_size = BLOCK_SIZE,
  n_folds = N_FOLDS,
  moran_k = MORAN_K,
  moran_n = MORAN_N,
  k_univariate = K_UNIVARIATE,
  k_spatial = K_SPATIAL,
  n_threads = N_THREADS,
  seed = SEED,
  started_at = Sys.time()
)

safe_save_rds(config, file.path(CHK_DIR, "00_analysis_config.rds"))
writeLines(
  capture.output(str(config)),
  file.path(LOG_DIR, "analysis_config.txt")
)

if (!file.exists(PRED_RASTER_PATH)) {
  stop("Input raster stack does not exist:\n", PRED_RASTER_PATH)
}

log_message("Analysis started.")
log_message("Input stack: ", PRED_RASTER_PATH)
log_message("Output directory: ", OUT_DIR)


# =============================================================================
# 3) READ AND AUDIT INPUT STACK
# =============================================================================

r <- rast(PRED_RASTER_PATH)

required_layers <- c(
  "DEM",
  "Aspect",
  "ShadowMask"
)

missing_required <- setdiff(required_layers, names(r))

if (length(missing_required) > 0L) {
  stop(
    "Missing required raster layers: ",
    paste(missing_required, collapse = ", ")
  )
}

# The working input is a multiband raster stack.
# DEM, Aspect and ShadowMask are read directly from this same file.
dem <- r[["DEM"]]
aspect <- r[["Aspect"]]
shadow_raw <- r[["ShadowMask"]]

capture_to_file(
  r,
  file.path(LOG_DIR, "01_input_raster_description.txt")
)

geometry_table <- data.frame(
  layer = names(r),
  nrow = nrow(r),
  ncol = ncol(r),
  resolution_x = res(r)[1],
  resolution_y = res(r)[2],
  xmin = xmin(r),
  xmax = xmax(r),
  ymin = ymin(r),
  ymax = ymax(r),
  crs = crs(r)
)

write.csv(
  geometry_table,
  file.path(TAB_DIR, "01_input_geometry.csv"),
  row.names = FALSE
)


# =============================================================================
# 4) SHADOW BINARY AUDIT
# =============================================================================

shadow_frequency <- as.data.frame(freq(shadow_raw))

write.csv(
  shadow_frequency,
  file.path(TAB_DIR, "02_shadow_original_frequency.csv"),
  row.names = FALSE
)

shadow_values <- shadow_frequency$value
already_binary <- all(shadow_values %in% c(0, 1))

if (already_binary) {
  shadow <- shadow_raw
  log_message("ShadowMask is already binary.")
} else {
  shadow <- ifel(shadow_raw >= 0.5, 1, 0)
  log_message(
    "ShadowMask was not strictly binary; threshold 0.5 was applied."
  )
}

names(shadow) <- "Shadow"

writeRaster(
  shadow,
  file.path(RAS_DIR, "01_ShadowMask_10m_binary_checked.tif"),
  overwrite = TRUE
)

global_shadow_fraction <- global(
  shadow,
  fun = "mean",
  na.rm = TRUE
)[1, 1]

global_shadow_sum <- global(
  shadow,
  fun = "sum",
  na.rm = TRUE
)[1, 1]

valid_pixel_count <- global(
  !is.na(shadow),
  fun = "sum",
  na.rm = TRUE
)[1, 1]

shadow_audit <- data.frame(
  valid_pixels = valid_pixel_count,
  shadow_pixels = global_shadow_sum,
  shadow_fraction = global_shadow_fraction,
  originally_binary = already_binary
)

write.csv(
  shadow_audit,
  file.path(TAB_DIR, "03_shadow_binary_audit.csv"),
  row.names = FALSE
)


# =============================================================================
# 5) DERIVE TOPOGRAPHIC VARIABLES
# =============================================================================

derived_stack_path <- file.path(
  RAS_DIR,
  "02_shadow_topography_predictor_stack_reduced_3x3_10m.tif"
)

if (!FORCE_REBUILD_RASTERS && file.exists(derived_stack_path)) {

  log_message("Recovering reduced 3x3 predictor stack.")
  topo_stack <- rast(derived_stack_path)

} else {

  log_message(
    "Deriving Northness, Eastness, TPI 3x3 and Local DEM SD 3x3."
  )

  aspect_rad <- aspect * pi / 180

  northness <- cos(aspect_rad)
  eastness <- sin(aspect_rad)

  names(northness) <- "Northness"
  names(eastness) <- "Eastness"

  w_3x3 <- matrix(
    1,
    nrow = 3,
    ncol = 3
  )

  local_mean_3x3 <- focal(
    dem,
    w = w_3x3,
    fun = mean,
    na.policy = "omit",
    na.rm = TRUE
  )

  tpi_3x3 <- dem - local_mean_3x3
  names(tpi_3x3) <- "TPI_3x3"

  local_dem_sd_3x3 <- focal(
    dem,
    w = w_3x3,
    fun = sd,
    na.policy = "omit",
    na.rm = TRUE
  )

  names(local_dem_sd_3x3) <- "LocalDEM_SD_3x3"
  names(dem) <- "DEM"

  topo_stack <- c(
    shadow,
    dem,
    northness,
    eastness,
    tpi_3x3,
    local_dem_sd_3x3
  )

  names(topo_stack) <- c(
    "Shadow",
    "DEM",
    "Northness",
    "Eastness",
    "TPI_3x3",
    "LocalDEM_SD_3x3"
  )

  # Mask all predictors to valid shadow pixels.
  topo_stack <- mask(
    topo_stack,
    shadow
  )

  writeRaster(
    topo_stack,
    derived_stack_path,
    overwrite = TRUE
  )
}

capture_to_file(
  topo_stack,
  file.path(
    LOG_DIR,
    "02_derived_stack_description.txt"
  )
)


# =============================================================================
# 6) REGULAR SPATIAL SAMPLE WITH COORDINATES
# =============================================================================

sample_checkpoint <- file.path(
  CHK_DIR,
  "01_regular_spatial_sample.rds"
)

sample_df <- load_or_run(
  sample_checkpoint,
  force = FORCE_RESAMPLE,
  fun = function() {

    log_message(
      "Drawing a regular spatial sample of ",
      format(N_SAMPLE, big.mark = ","),
      " pixels."
    )

    sample_points <- spatSample(
      topo_stack,
      size = N_SAMPLE,
      method = "regular",
      na.rm = TRUE,
      as.points = TRUE,
      values = TRUE
    )

    df <- as.data.frame(
      sample_points,
      geom = "XY",
      na.rm = TRUE
    )

    df <- as.data.table(df)

    if (!all(c("x", "y") %in% names(df))) {
      stop("Coordinates x and y were not returned by spatSample.")
    }

    df <- df[Shadow %in% c(0, 1)]

    numeric_columns <- names(df)[
      vapply(df, is.numeric, logical(1))
    ]

    for (v in numeric_columns) {
      set(
        df,
        i = which(!is.finite(df[[v]])),
        j = v,
        value = NA_real_
      )
    }

    df <- na.omit(df)
    df[, Shadow := as.integer(Shadow)]

    df
  }
)

sample_representativeness <- data.frame(
  global_shadow_fraction = global_shadow_fraction,
  sample_shadow_fraction = mean(sample_df$Shadow),
  absolute_difference = abs(
    mean(sample_df$Shadow) - global_shadow_fraction
  ),
  sample_n = nrow(sample_df)
)

write.csv(
  sample_representativeness,
  file.path(TAB_DIR, "04_sample_representativeness.csv"),
  row.names = FALSE
)

fwrite(
  sample_df,
  file.path(TAB_DIR, "05_regular_spatial_sample.csv")
)

log_message(
  "Usable regular sample size: ",
  format(nrow(sample_df), big.mark = ",")
)


# =============================================================================
# 7) DESCRIPTIVE STATISTICS BY SHADOW CLASS
# =============================================================================

all_candidate_predictors <- setdiff(
  names(sample_df),
  c("Shadow", "x", "y")
)

shadow_class_summary <- rbindlist(
  lapply(
    all_candidate_predictors,
    function(v) {
      sample_df[
        ,
        .(
          n = .N,
          mean = mean(get(v), na.rm = TRUE),
          sd = sd(get(v), na.rm = TRUE),
          median = median(get(v), na.rm = TRUE),
          q25 = quantile(get(v), 0.25, na.rm = TRUE),
          q75 = quantile(get(v), 0.75, na.rm = TRUE)
        ),
        by = Shadow
      ][
        ,
        variable := v
      ]
    }
  ),
  use.names = TRUE,
  fill = TRUE
)

setcolorder(
  shadow_class_summary,
  c(
    "variable", "Shadow", "n",
    "mean", "sd", "median", "q25", "q75"
  )
)

fwrite(
  shadow_class_summary,
  file.path(TAB_DIR, "06_topography_by_shadow_class.csv")
)


# =============================================================================
# 8) CORRELATION DIAGNOSTICS AND PREDICTOR SELECTION
# =============================================================================

cor_n <- min(50000L, nrow(sample_df))
set.seed(SEED)

cor_df <- sample_df[
  sample(.N, cor_n),
  ..all_candidate_predictors
]

cor_matrix <- cor(
  cor_df,
  use = "pairwise.complete.obs",
  method = "spearman"
)

write.csv(
  cor_matrix,
  file.path(TAB_DIR, "07_spearman_predictor_correlations.csv")
)

high_cor_index <- which(
  abs(cor_matrix) >= 0.75 &
    upper.tri(cor_matrix),
  arr.ind = TRUE
)

if (nrow(high_cor_index) > 0L) {
  high_cor_pairs <- data.frame(
    variable_1 = rownames(cor_matrix)[high_cor_index[, 1]],
    variable_2 = colnames(cor_matrix)[high_cor_index[, 2]],
    spearman_rho = cor_matrix[high_cor_index]
  )
} else {
  high_cor_pairs <- data.frame(
    variable_1 = character(),
    variable_2 = character(),
    spearman_rho = numeric()
  )
}

write.csv(
  high_cor_pairs,
  file.path(TAB_DIR, "08_high_correlations_abs_rho_ge_075.csv"),
  row.names = FALSE
)

# Final reduced predictor set:
# - DEM: broad elevation gradient
# - Northness and Eastness: circular representation of exposure
# - TPI_3x3: local terrain position
# - LocalDEM_SD_3x3: local microtopographic heterogeneity

primary_predictors <- c(
  "DEM",
  "Northness",
  "Eastness",
  "TPI_3x3",
  "LocalDEM_SD_3x3"
)

missing_primary <- setdiff(primary_predictors, names(sample_df))

if (length(missing_primary) > 0L) {
  stop(
    "Missing primary predictors after raster processing: ",
    paste(missing_primary, collapse = ", ")
  )
}

write.csv(
  data.frame(primary_predictor = primary_predictors),
  file.path(TAB_DIR, "09_primary_predictor_set.csv"),
  row.names = FALSE
)


# =============================================================================
# 9) STANDARDIZE MODEL PREDICTORS AND BUILD SPATIAL BLOCKS
# =============================================================================

model_checkpoint <- file.path(
  CHK_DIR,
  "02_model_dataframe.rds"
)

model_df <- load_or_run(
  model_checkpoint,
  force = FORCE_RESAMPLE,
  fun = function() {

    df <- copy(sample_df)

    scaling_table <- rbindlist(
      lapply(
        primary_predictors,
        function(v) {
          data.table(
            variable = v,
            mean = mean(df[[v]], na.rm = TRUE),
            sd = sd(df[[v]], na.rm = TRUE)
          )
        }
      )
    )

    if (any(!is.finite(scaling_table$sd)) ||
        any(scaling_table$sd <= 0)) {
      stop("At least one predictor has zero or invalid SD.")
    }

    for (v in primary_predictors) {
      mu <- scaling_table[variable == v, mean]
      sigma <- scaling_table[variable == v, sd]
      df[, paste0(v, "_z") := (get(v) - mu) / sigma]
    }

    df[, block_x := floor(x / BLOCK_SIZE)]
    df[, block_y := floor(y / BLOCK_SIZE)]

    df[
      ,
      spatial_block := interaction(
        block_x,
        block_y,
        drop = TRUE
      )
    ]

    block_ids <- unique(df$spatial_block)

    if (length(block_ids) < N_FOLDS) {
      stop(
        "There are fewer spatial blocks than requested folds. ",
        "Reduce BLOCK_SIZE or N_FOLDS."
      )
    }

    set.seed(SEED)

    fold_table <- data.table(
      spatial_block = block_ids,
      fold = sample(
        rep(
          seq_len(N_FOLDS),
          length.out = length(block_ids)
        )
      )
    )

    df <- merge(
      df,
      fold_table,
      by = "spatial_block",
      all.x = TRUE,
      sort = FALSE
    )

    safe_save_rds(
      scaling_table,
      file.path(CHK_DIR, "02_predictor_scaling_table.rds")
    )

    df
  }
)

scaling_table <- readRDS(
  file.path(CHK_DIR, "02_predictor_scaling_table.rds")
)

z_predictors <- paste0(primary_predictors, "_z")

fwrite(
  model_df[
    ,
    c(
      "x", "y", "Shadow",
      "spatial_block", "fold",
      primary_predictors,
      z_predictors
    ),
    with = FALSE
  ],
  file.path(TAB_DIR, "10_model_dataframe.csv")
)


# =============================================================================
# 10) MODEL FORMULAS
# =============================================================================

smooth_terms <- paste0(
  "s(",
  z_predictors,
  ", k = ",
  K_UNIVARIATE,
  ")"
)

formula_null <- Shadow ~ 1

formula_elevation <- as.formula(
  paste0(
    "Shadow ~ s(DEM_z, k = ",
    max(10L, K_UNIVARIATE),
    ")"
  )
)

formula_topography <- as.formula(
  paste(
    "Shadow ~",
    paste(smooth_terms, collapse = " + ")
  )
)

formula_topography_space <- as.formula(
  paste(
    "Shadow ~",
    paste(smooth_terms, collapse = " + "),
    "+ s(x, y, bs = 'tp', k =",
    K_SPATIAL,
    ")"
  )
)

writeLines(
  c(
    paste("Null:", deparse(formula_null)),
    paste("Elevation:", deparse(formula_elevation)),
    paste("Topography:", deparse(formula_topography)),
    paste(
      "Topography + space:",
      deparse(formula_topography_space)
    )
  ),
  file.path(LOG_DIR, "03_model_formulas.txt")
)


# =============================================================================
# 11) RECOVERABLE MODEL FITTING
# =============================================================================

fit_bam_recoverable <- function(
    model_name,
    formula,
    data,
    select = FALSE,
    force = FORCE_REFIT_MODELS
) {

  model_path <- file.path(
    MOD_DIR,
    paste0(model_name, ".rds")
  )

  load_or_run(
    model_path,
    force = force,
    fun = function() {

      log_message("Fitting model: ", model_name)

      formula_text <- paste(
        deparse(formula),
        collapse = " "
      )

      has_smooth <- grepl(
        "\\bs\\s*\\(|\\bte\\s*\\(|\\bti\\s*\\(|\\bt2\\s*\\(",
        formula_text
      )

      if (!has_smooth) {

        glm(
          formula = formula,
          data = data,
          family = binomial(link = "logit"),
          na.action = na.fail
        )

      } else {

        bam(
          formula = formula,
          data = data,
          family = binomial(link = "logit"),
          method = "fREML",
          discrete = TRUE,
          select = select,
          nthreads = N_THREADS,
          na.action = na.fail
        )
      }
    }
  )
}

m0_null <- fit_bam_recoverable(
  "m0_null",
  formula_null,
  model_df
)

m1_elevation <- fit_bam_recoverable(
  "m1_elevation_only",
  formula_elevation,
  model_df
)

m2_topography <- fit_bam_recoverable(
  "m2_topography_only",
  formula_topography,
  model_df,
  select = TRUE
)

m3_topography_space <- fit_bam_recoverable(
  "m3_topography_plus_space",
  formula_topography_space,
  model_df,
  select = TRUE
)

for (object_name in c(
  "m0_null",
  "m1_elevation",
  "m2_topography",
  "m3_topography_space"
)) {

  model_object <- get(object_name)

  capture_to_file(
    summary(model_object),
    file.path(
      MOD_DIR,
      paste0("summary_", object_name, ".txt")
    )
  )

  if (inherits(model_object, "gam")) {
    capture_to_file(
      gam.check(model_object),
      file.path(
        MOD_DIR,
        paste0("gam_check_", object_name, ".txt")
      )
    )
  }
}


# =============================================================================
# 12) MODEL COMPARISON AND IN-SAMPLE DISCRIMINATION
# =============================================================================

model_list <- list(
  Null = m0_null,
  Elevation_only = m1_elevation,
  Topography = m2_topography,
  Topography_plus_space = m3_topography_space
)

model_comparison <- rbindlist(
  lapply(
    names(model_list),
    function(model_name) {
      model <- model_list[[model_name]]
      model_summary <- summary(model)

      deviance_explained <- if (
        inherits(model, "gam") &&
          !is.null(model_summary$dev.expl)
      ) {
        as.numeric(model_summary$dev.expl)
      } else if (
        !is.null(model$deviance) &&
          !is.null(model$null.deviance) &&
          is.finite(model$null.deviance) &&
          model$null.deviance > 0
      ) {
        1 - model$deviance / model$null.deviance
      } else {
        NA_real_
      }

      adjusted_r2 <- if (
        inherits(model, "gam") &&
          !is.null(model_summary$r.sq)
      ) {
        as.numeric(model_summary$r.sq)
      } else {
        NA_real_
      }

      data.table(
        model = model_name,
        AIC = AIC(model),
        deviance_explained = deviance_explained,
        adjusted_R2 = adjusted_r2,
        n = nobs(model)
      )
    }
  )
)

fwrite(
  model_comparison,
  file.path(TAB_DIR, "11_model_comparison.csv")
)

prediction_checkpoint <- file.path(
  CHK_DIR,
  "03_in_sample_predictions.rds"
)

in_sample_predictions <- load_or_run(
  prediction_checkpoint,
  force = FORCE_REFIT_MODELS,
  fun = function() {

    data.table(
      Shadow = model_df$Shadow,
      elevation = predict(
        m1_elevation,
        newdata = model_df,
        type = "response"
      ),
      topography = predict(
        m2_topography,
        newdata = model_df,
        type = "response"
      ),
      topography_space = predict(
        m3_topography_space,
        newdata = model_df,
        type = "response"
      )
    )
  }
)

auc_table <- rbindlist(
  lapply(
    setdiff(names(in_sample_predictions), "Shadow"),
    function(v) {

      roc_object <- pROC::roc(
        response = in_sample_predictions$Shadow,
        predictor = in_sample_predictions[[v]],
        quiet = TRUE
      )

      data.table(
        model = v,
        AUC = as.numeric(pROC::auc(roc_object)),
        Brier = mean(
          (
            in_sample_predictions$Shadow -
              in_sample_predictions[[v]]
          )^2
        )
      )
    }
  )
)

fwrite(
  auc_table,
  file.path(TAB_DIR, "12_in_sample_auc_brier.csv")
)


# =============================================================================
# 13) SMOOTH-TERM TESTS AND CONCURVITY
# =============================================================================

extract_smooth_table <- function(model, model_name) {

  smooth_table <- summary(model)$s.table

  if (is.null(smooth_table)) {
    return(data.table())
  }

  output <- as.data.table(
    as.data.frame(smooth_table),
    keep.rownames = "term"
  )

  output[, model := model_name]
  output
}

smooth_tests <- rbindlist(
  list(
    extract_smooth_table(
      m2_topography,
      "Topography"
    ),
    extract_smooth_table(
      m3_topography_space,
      "Topography + space"
    )
  ),
  use.names = TRUE,
  fill = TRUE
)

fwrite(
  smooth_tests,
  file.path(TAB_DIR, "13_smooth_term_tests.csv")
)

concurvity_m2 <- concurvity(
  m2_topography,
  full = TRUE
)

concurvity_m3 <- concurvity(
  m3_topography_space,
  full = TRUE
)

write.csv(
  as.data.frame(concurvity_m2),
  file.path(TAB_DIR, "14_concurvity_topography.csv")
)

write.csv(
  as.data.frame(concurvity_m3),
  file.path(TAB_DIR, "15_concurvity_topography_space.csv")
)


# =============================================================================
# 14) SPATIAL BLOCK CROSS-VALIDATION
# =============================================================================

cv_checkpoint <- file.path(
  CHK_DIR,
  "04_spatial_cross_validation.rds"
)

cv_results <- load_or_run(
  cv_checkpoint,
  force = FORCE_REFIT_CV,
  fun = function() {

    cv_df <- copy(model_df)

    if (nrow(cv_df) > CV_MAX_N) {
      set.seed(SEED)
      cv_df <- cv_df[
        sample(.N, CV_MAX_N)
      ]
    }

    fold_results <- vector(
      "list",
      length = N_FOLDS
    )

    for (f in seq_len(N_FOLDS)) {

      log_message(
        "Spatial CV fold ",
        f,
        " of ",
        N_FOLDS
      )

      train_df <- cv_df[fold != f]
      test_df <- cv_df[fold == f]

      if (
        length(unique(train_df$Shadow)) < 2L ||
          length(unique(test_df$Shadow)) < 2L
      ) {
        warning(
          "Fold ",
          f,
          " does not contain both classes."
        )
        next
      }

      cv_model_path <- file.path(
        MOD_DIR,
        paste0("cv_fold_", f, "_model.rds")
      )

      cv_model <- load_or_run(
        cv_model_path,
        force = FORCE_REFIT_CV,
        fun = function() {
          bam(
            formula_topography,
            data = train_df,
            family = binomial(link = "logit"),
            method = "fREML",
            discrete = TRUE,
            select = TRUE,
            nthreads = N_THREADS,
            na.action = na.fail
          )
        }
      )

      test_prediction <- predict(
        cv_model,
        newdata = test_df,
        type = "response"
      )

      roc_object <- pROC::roc(
        response = test_df$Shadow,
        predictor = test_prediction,
        quiet = TRUE
      )

      fold_results[[f]] <- data.table(
        fold = f,
        n_train = nrow(train_df),
        n_test = nrow(test_df),
        prevalence_test = mean(test_df$Shadow),
        AUC = as.numeric(pROC::auc(roc_object)),
        Brier = mean(
          (test_df$Shadow - test_prediction)^2
        )
      )

      rm(cv_model)
      gc()
    }

    rbindlist(
      fold_results,
      use.names = TRUE,
      fill = TRUE
    )
  }
)

fwrite(
  cv_results,
  file.path(TAB_DIR, "16_spatial_block_cross_validation.csv")
)

cv_summary <- cv_results[
  ,
  .(
    mean_AUC = mean(AUC, na.rm = TRUE),
    sd_AUC = sd(AUC, na.rm = TRUE),
    mean_Brier = mean(Brier, na.rm = TRUE),
    sd_Brier = sd(Brier, na.rm = TRUE),
    n_valid_folds = sum(is.finite(AUC))
  )
]

fwrite(
  cv_summary,
  file.path(TAB_DIR, "17_spatial_cv_summary.csv")
)


# =============================================================================
# 15) DROP-ONE CONTRIBUTION ANALYSIS
# =============================================================================

# This is the principal answer to:
# "Which topographic characteristics make the largest contribution
# to persistent shadow?"
#
# For each predictor, a reduced model is fitted without that predictor.
#
# Two primary contribution metrics:
# 1) loss_in_deviance_explained:
#      full model deviance explained - reduced model deviance explained
# 2) delta_AIC:
#      reduced-model AIC - full-model AIC
#
# Larger positive values indicate greater unique contribution,
# conditional on the other predictors.

drop_one_checkpoint <- file.path(
  CHK_DIR,
  "05_drop_one_contribution.rds"
)

drop_one_results <- load_or_run(
  drop_one_checkpoint,
  force = FORCE_REFIT_DROP_ONE,
  fun = function() {

    full_deviance <- summary(m2_topography)$dev.expl
    full_aic <- AIC(m2_topography)

    results <- vector(
      "list",
      length(primary_predictors)
    )

    for (i in seq_along(primary_predictors)) {

      removed_predictor <- primary_predictors[i]
      removed_z <- paste0(removed_predictor, "_z")

      remaining_z <- setdiff(
        z_predictors,
        removed_z
      )

      reduced_formula <- as.formula(
        paste(
          "Shadow ~",
          paste0(
            "s(",
            remaining_z,
            ", k = ",
            K_UNIVARIATE,
            ")",
            collapse = " + "
          )
        )
      )

      reduced_path <- file.path(
        MOD_DIR,
        paste0(
          "drop_one_without_",
          removed_predictor,
          ".rds"
        )
      )

      reduced_model <- load_or_run(
        reduced_path,
        force = FORCE_REFIT_DROP_ONE,
        fun = function() {
          log_message(
            "Drop-one model: removing ",
            removed_predictor
          )

          bam(
            reduced_formula,
            data = model_df,
            family = binomial(link = "logit"),
            method = "fREML",
            discrete = TRUE,
            select = TRUE,
            nthreads = N_THREADS,
            na.action = na.fail
          )
        }
      )

      reduced_deviance <- summary(
        reduced_model
      )$dev.expl

      reduced_aic <- AIC(reduced_model)

      results[[i]] <- data.table(
        removed_predictor = removed_predictor,
        full_deviance_explained = full_deviance,
        reduced_deviance_explained = reduced_deviance,
        loss_in_deviance_explained =
          full_deviance - reduced_deviance,
        full_AIC = full_aic,
        reduced_AIC = reduced_aic,
        delta_AIC = reduced_aic - full_aic
      )

      rm(reduced_model)
      gc()
    }

    output <- rbindlist(results)

    output[
      ,
      relative_unique_contribution_percent :=
        100 *
        pmax(loss_in_deviance_explained, 0) /
        sum(
          pmax(loss_in_deviance_explained, 0),
          na.rm = TRUE
        )
    ]

    setorder(
      output,
      -loss_in_deviance_explained
    )

    output
  }
)

fwrite(
  drop_one_results,
  file.path(TAB_DIR, "18_drop_one_variable_contribution.csv")
)


# =============================================================================
# 16) PREDICTOR-BLOCK DEVIANCE PARTITIONING
# =============================================================================

# Predictor blocks represent the four retained topographic processes:
#
# Elevation gradient:
#   DEM
#
# Exposure:
#   Northness + Eastness
#
# Terrain position:
#   TPI_3x3
#
# Microtopographic heterogeneity:
#   LocalDEM_SD_3x3
#
# Northness and Eastness are removed together because they jointly encode
# circular aspect/exposure.

predictor_blocks <- list(
  Elevation_gradient = "DEM_z",
  Exposure = c(
    "Northness_z",
    "Eastness_z"
  ),
  Terrain_position = "TPI_3x3_z",
  Microtopographic_heterogeneity =
    "LocalDEM_SD_3x3_z"
)

block_checkpoint <- file.path(
  CHK_DIR,
  "06_predictor_block_deviance_partitioning.rds"
)

block_partition <- load_or_run(
  block_checkpoint,
  force = FORCE_REFIT_DROP_ONE,
  fun = function() {

    full_deviance <- summary(
      m2_topography
    )$dev.expl

    full_aic <- AIC(m2_topography)

    results <- vector(
      "list",
      length(predictor_blocks)
    )

    for (i in seq_along(predictor_blocks)) {

      block_name <- names(
        predictor_blocks
      )[i]

      removed_terms <- predictor_blocks[[i]]

      remaining_z <- setdiff(
        z_predictors,
        removed_terms
      )

      reduced_formula <- as.formula(
        paste(
          "Shadow ~",
          paste0(
            "s(",
            remaining_z,
            ", k = ",
            K_UNIVARIATE,
            ")",
            collapse = " + "
          )
        )
      )

      model_path <- file.path(
        MOD_DIR,
        paste0(
          "block_without_",
          block_name,
          ".rds"
        )
      )

      reduced_model <- load_or_run(
        model_path,
        force = FORCE_REFIT_DROP_ONE,
        fun = function() {

          log_message(
            "Block partition model: removing ",
            block_name
          )

          bam(
            reduced_formula,
            data = model_df,
            family = binomial(link = "logit"),
            method = "fREML",
            discrete = TRUE,
            select = TRUE,
            nthreads = N_THREADS,
            na.action = na.fail
          )
        }
      )

      reduced_deviance <- summary(
        reduced_model
      )$dev.expl

      reduced_aic <- AIC(reduced_model)

      results[[i]] <- data.table(
        removed_block = block_name,
        block_members = paste(
          sub("_z$", "", removed_terms),
          collapse = " + "
        ),
        full_deviance_explained = full_deviance,
        reduced_deviance_explained = reduced_deviance,
        loss_in_deviance_explained =
          full_deviance - reduced_deviance,
        delta_AIC = reduced_aic - full_aic
      )

      rm(reduced_model)
      gc()
    }

    output <- rbindlist(results)

    output[
      ,
      relative_block_contribution_percent :=
        100 *
        pmax(loss_in_deviance_explained, 0) /
        sum(
          pmax(loss_in_deviance_explained, 0),
          na.rm = TRUE
        )
    ]

    setorder(
      output,
      -loss_in_deviance_explained
    )

    output
  }
)

fwrite(
  block_partition,
  file.path(
    TAB_DIR,
    "19_predictor_block_deviance_partitioning.csv"
  )
)




# =============================================================================
# 18) RESIDUAL SPATIAL AUTOCORRELATION
# =============================================================================

moran_checkpoint <- file.path(
  CHK_DIR,
  "07_residual_morans_I.rds"
)

moran_results <- load_or_run(
  moran_checkpoint,
  force = FORCE_REFIT_MODELS,
  fun = function() {

    moran_n <- min(
      MORAN_N,
      nrow(model_df)
    )

    set.seed(SEED)

    moran_df <- model_df[
      sample(.N, moran_n)
    ]

    moran_df[
      ,
      pred_topography := predict(
        m2_topography,
        newdata = moran_df,
        type = "response"
      )
    ]

    moran_df[
      ,
      pred_topography_space := predict(
        m3_topography_space,
        newdata = moran_df,
        type = "response"
      )
    ]

    moran_df[
      ,
      residual_topography :=
        Shadow - pred_topography
    ]

    moran_df[
      ,
      residual_topography_space :=
        Shadow - pred_topography_space
    ]

    coords <- as.matrix(
      moran_df[
        ,
        .(x, y)
      ]
    )

    knn <- knearneigh(
      coords,
      k = MORAN_K
    )

    neighbours <- knn2nb(knn)

    list_weights <- nb2listw(
      neighbours,
      style = "W",
      zero.policy = TRUE
    )

    moran_topography <- moran.test(
      moran_df$residual_topography,
      list_weights,
      zero.policy = TRUE
    )

    moran_topography_space <- moran.test(
      moran_df$residual_topography_space,
      list_weights,
      zero.policy = TRUE
    )

    data.table(
      model = c(
        "Topography",
        "Topography + space"
      ),
      Moran_I = c(
        unname(
          moran_topography$estimate[
            "Moran I statistic"
          ]
        ),
        unname(
          moran_topography_space$estimate[
            "Moran I statistic"
          ]
        )
      ),
      expectation = c(
        unname(
          moran_topography$estimate[
            "Expectation"
          ]
        ),
        unname(
          moran_topography_space$estimate[
            "Expectation"
          ]
        )
      ),
      p_value = c(
        moran_topography$p.value,
        moran_topography_space$p.value
      ),
      n = moran_n,
      k_neighbours = MORAN_K
    )
  }
)

fwrite(
  moran_results,
  file.path(TAB_DIR, "20_residual_morans_I.csv")
)


# =============================================================================
# 19) FIGURES
# =============================================================================

p_drop_one <- ggplot(
  drop_one_results,
  aes(
    x = reorder(
      removed_predictor,
      loss_in_deviance_explained
    ),
    y = loss_in_deviance_explained
  )
) +
  geom_col() +
  coord_flip() +
  theme_bw(base_size = 12) +
  labs(
    title = paste(
      "Unique topographic contribution",
      "to persistent shadow"
    ),
    subtitle = paste(
      "Drop-one binomial BAM;",
      "larger loss indicates greater unique contribution"
    ),
    x = NULL,
    y = "Loss in deviance explained"
  )

ggsave(
  file.path(
    FIG_DIR,
    "01_drop_one_unique_contribution.png"
  ),
  p_drop_one,
  width = 8,
  height = 6,
  dpi = 400
)

p_blocks <- ggplot(
  block_partition,
  aes(
    x = reorder(
      removed_block,
      loss_in_deviance_explained
    ),
    y = loss_in_deviance_explained
  )
) +
  geom_col() +
  coord_flip() +
  theme_bw(base_size = 12) +
  labs(
    title = paste(
      "Contribution of topographic process blocks",
      "to persistent shadow"
    ),
    subtitle = paste(
      "Block-wise deviance partitioning",
      "conditional on all remaining blocks"
    ),
    x = NULL,
    y = "Loss in deviance explained"
  )

ggsave(
  file.path(
    FIG_DIR,
    "02_topographic_block_deviance_partitioning.png"
  ),
  p_blocks,
  width = 8,
  height = 6,
  dpi = 400
)

plot_n <- min(
  30000L,
  nrow(model_df)
)

set.seed(SEED)

plot_df <- model_df[
  sample(.N, plot_n)
]

plot_df[
  ,
  Shadow_class := factor(
    Shadow,
    levels = c(0, 1),
    labels = c(
      "Non-shadow",
      "Persistent shadow"
    )
  )
]

p_elevation <- ggplot(
  plot_df,
  aes(
    x = Shadow_class,
    y = DEM
  )
) +
  geom_violin(trim = TRUE) +
  geom_boxplot(
    width = 0.16,
    outlier.shape = NA
  ) +
  theme_bw(base_size = 12) +
  labs(
    title = paste(
      "Elevation distribution across",
      "persistent-shadow regimes"
    ),
    x = NULL,
    y = "Elevation"
  )

ggsave(
  file.path(
    FIG_DIR,
    "03_elevation_by_shadow_class.png"
  ),
  p_elevation,
  width = 7,
  height = 6,
  dpi = 400
)

density_df <- data.table(
  Shadow = model_df$Shadow,
  predicted_probability =
    in_sample_predictions$topography
)

if (nrow(density_df) > 50000L) {
  set.seed(SEED)
  density_df <- density_df[
    sample(.N, 50000L)
  ]
}

density_df[
  ,
  Shadow_class := factor(
    Shadow,
    levels = c(0, 1),
    labels = c(
      "Non-shadow",
      "Persistent shadow"
    )
  )
]

p_probability <- ggplot(
  density_df,
  aes(
    x = predicted_probability,
    fill = Shadow_class
  )
) +
  geom_density(alpha = 0.35) +
  theme_bw(base_size = 12) +
  labs(
    title = paste(
      "Topographic predictability",
      "of persistent shadow"
    ),
    x = "Predicted probability",
    y = "Density",
    fill = NULL
  )

ggsave(
  file.path(
    FIG_DIR,
    "04_predicted_probability_by_observed_class.png"
  ),
  p_probability,
  width = 8,
  height = 6,
  dpi = 400
)

png(
  file.path(
    FIG_DIR,
    "05_partial_effects_topography_model.png"
  ),
  width = 3400,
  height = 2800,
  res = 300
)

plot(
  m2_topography,
  pages = 1,
  shade = TRUE,
  scale = 0,
  residuals = FALSE
)

dev.off()


# =============================================================================
# 20) FULL-REGION PREDICTION AND UNEXPLAINED-SHADOW MAP
# =============================================================================

probability_path <- file.path(
  RAS_DIR,
  "03_predicted_shadow_probability_from_topography_10m.tif"
)

residual_path <- file.path(
  RAS_DIR,
  "04_observed_minus_topographic_prediction_10m.tif"
)

if (
  FORCE_REPREDICT_RASTERS ||
    !file.exists(probability_path) ||
    !file.exists(residual_path)
) {

  log_message(
    "Predicting topographic shadow probability across the full raster."
  )

  expected_z_predictors <- paste0(
    primary_predictors,
    "_z"
  )

  actual_model_variables <- setdiff(
    all.vars(
      formula(m2_topography)
    ),
    "Shadow"
  )

  if (!setequal(
    actual_model_variables,
    expected_z_predictors
  )) {
    stop(
      "m2_topography does not match the reduced predictor set.\n",
      "Model variables: ",
      paste(actual_model_variables, collapse = ", "),
      "\nExpected variables: ",
      paste(expected_z_predictors, collapse = ", ")
    )
  }

  missing_raster_predictors <- setdiff(
    primary_predictors,
    names(topo_stack)
  )

  if (length(missing_raster_predictors) > 0L) {
    stop(
      "Missing raster predictors in topo_stack: ",
      paste(missing_raster_predictors, collapse = ", ")
    )
  }

  prediction_stack_raw <- topo_stack[[
    primary_predictors
  ]]

  standardized_layers <- vector(
    mode = "list",
    length = length(primary_predictors)
  )

  for (i in seq_along(primary_predictors)) {

    variable_name <- primary_predictors[[i]]

    scaling_row <- scaling_table[
      variable == variable_name
    ]

    if (nrow(scaling_row) != 1L) {
      stop(
        "Expected exactly one scaling row for ",
        variable_name,
        "; found ",
        nrow(scaling_row),
        "."
      )
    }

    mu <- scaling_row$mean[[1]]
    sigma <- scaling_row$sd[[1]]

    if (
      !is.finite(mu) ||
        !is.finite(sigma) ||
        sigma <= 0
    ) {
      stop(
        "Invalid scaling values for ",
        variable_name,
        "."
      )
    }

    standardized_layers[[i]] <- (
      prediction_stack_raw[[variable_name]] - mu
    ) / sigma

    names(
      standardized_layers[[i]]
    ) <- paste0(
      variable_name,
      "_z"
    )
  }

  prediction_stack_z <- standardized_layers[[1]]

  if (length(standardized_layers) > 1L) {
    for (i in 2:length(standardized_layers)) {
      prediction_stack_z <- c(
        prediction_stack_z,
        standardized_layers[[i]]
      )
    }
  }

  names(prediction_stack_z) <- expected_z_predictors

  prediction_stack_z <- prediction_stack_z[[
    actual_model_variables
  ]]

  shadow_probability <- terra::predict(
    object = prediction_stack_z,
    model = m2_topography,
    type = "response",
    na.rm = TRUE,
    filename = probability_path,
    overwrite = TRUE
  )

  names(shadow_probability) <-
    "Predicted_shadow_probability"

  shadow_probability <- terra::clamp(
    shadow_probability,
    lower = 0,
    upper = 1,
    values = TRUE
  )

  writeRaster(
    shadow_probability,
    probability_path,
    overwrite = TRUE
  )

  if (!isTRUE(
    compareGeom(
      shadow,
      shadow_probability,
      crs = TRUE,
      ext = TRUE,
      rowcol = TRUE,
      res = TRUE,
      stopOnError = FALSE
    )
  )) {
    stop(
      "Observed shadow and predicted probability rasters have ",
      "different geometry."
    )
  }

  shadow_residual <- shadow -
    shadow_probability

  names(shadow_residual) <-
    "Observed_minus_predicted_shadow"

  writeRaster(
    shadow_residual,
    residual_path,
    overwrite = TRUE
  )

} else {

  log_message(
    "Recovering previously predicted full-region rasters."
  )

  shadow_probability <- rast(
    probability_path
  )

  shadow_residual <- rast(
    residual_path
  )
}


# =============================================================================
# 21) FINAL SUMMARY
# =============================================================================

final_summary <- data.table(
  metric = c(
    "Global shadow fraction",
    "Regular-sample shadow fraction",
    "Regular-sample size",
    "Elevation-only deviance explained",
    "Topography deviance explained",
    "Topography plus space deviance explained",
    "Elevation-only in-sample AUC",
    "Topography in-sample AUC",
    "Topography plus space in-sample AUC",
    "Spatial CV mean AUC",
    "Spatial CV SD AUC",
    "Spatial CV mean Brier",
    "Residual Moran I: topography",
    "Residual Moran I: topography plus space"
  ),
  value = c(
    global_shadow_fraction,
    mean(model_df$Shadow),
    nrow(model_df),
    summary(m1_elevation)$dev.expl,
    summary(m2_topography)$dev.expl,
    summary(m3_topography_space)$dev.expl,
    auc_table[
      model == "elevation",
      AUC
    ],
    auc_table[
      model == "topography",
      AUC
    ],
    auc_table[
      model == "topography_space",
      AUC
    ],
    cv_summary$mean_AUC,
    cv_summary$sd_AUC,
    cv_summary$mean_Brier,
    moran_results[
      model == "Topography",
      Moran_I
    ],
    moran_results[
      model == "Topography + space",
      Moran_I
    ]
  )
)

fwrite(
  final_summary,
  file.path(TAB_DIR, "21_FINAL_ANALYSIS_SUMMARY.csv")
)

# Rank variables and blocks for a manuscript-ready table.

manuscript_variable_ranking <- drop_one_results[
  ,
  .(
    rank = seq_len(.N),
    topographic_feature = removed_predictor,
    unique_loss_in_deviance =
      loss_in_deviance_explained,
    relative_unique_contribution_percent,
    delta_AIC
  )
]

fwrite(
  manuscript_variable_ranking,
  file.path(
    TAB_DIR,
    "22_MANUSCRIPT_variable_contribution_ranking.csv"
  )
)

manuscript_block_ranking <- block_partition[
  ,
  .(
    rank = seq_len(.N),
    topographic_process = removed_block,
    variables = block_members,
    unique_loss_in_deviance =
      loss_in_deviance_explained,
    relative_block_contribution_percent,
    delta_AIC
  )
]

fwrite(
  manuscript_block_ranking,
  file.path(
    TAB_DIR,
    "23_MANUSCRIPT_process_block_ranking.csv"
  )
)



# =============================================================================
# 23) REPRODUCIBILITY AND COMPLETION MARKERS
# =============================================================================

config$completed_at <- Sys.time()

safe_save_rds(
  config,
  file.path(CHK_DIR, "99_completed_analysis_config.rds")
)

capture_to_file(
  sessionInfo(),
  file.path(OUT_DIR, "sessionInfo.txt")
)

writeLines(
  c(
    paste("Analysis completed:", Sys.time()),
    paste("Input:", PRED_RASTER_PATH),
    paste("Output:", OUT_DIR),
    paste("Sample size:", nrow(model_df)),
    paste("Threads:", N_THREADS),
    "",
    "To resume after interruption:",
    "1) run this same script again;",
    "2) keep FORCE_* options set to FALSE;",
    "3) existing samples, models, CV folds, drop-one models and",
    "   raster predictions will be recovered from checkpoints."
  ),
  file.path(OUT_DIR, "ANALYSIS_COMPLETED.txt")
)

log_message("Analysis completed successfully.")
log_message("Key contribution table: ",
            file.path(
              TAB_DIR,
              "22_MANUSCRIPT_variable_contribution_ranking.csv"
            ))
log_message("Key block table: ",
            file.path(
              TAB_DIR,
              "23_MANUSCRIPT_process_block_ranking.csv"
            ))
log_message("Output directory: ", OUT_DIR)
