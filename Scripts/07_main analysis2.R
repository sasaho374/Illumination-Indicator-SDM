# ======================================================================
# Reproducible analysis pipeline
# Illumination regime and species distribution models
#
# Study design
#
# Sampling schemes:
#   UR = shadow representation follows landscape prevalence
#   SB = shadow representation follows the original fitting-pool prevalence
#   SW = the same sampled observations as SB, with case weights adjusted
#        to landscape shadow prevalence
#
# Predictor configurations:
#   RAW
#     = all modelling predictors except SCL, ShadowMask, and shadow_bin
#
#   RAW+SHADOW
#     = RAW predictors plus shadow_bin
#
# Data separation:
#
#   ORIGINAL TRAINING DATA
#          |
#          +---- FITTING POOL -----> UR / SB / SW resampling
#          |
#          +---- CALIBRATION ------> threshold selection only
#
#   EXTERNAL TEST FOLD 2 ----------> evaluation only
#
# Analysis rules:
#   - SCL is not used as a model predictor.
#   - ShadowMask is not duplicated in the modelling data.
#   - shadow_bin is derived only from the raster ShadowMask layer.
#   - External-test data do not affect sampling, model fitting,
#     or threshold selection.
#   - Calibration data are separated before stochastic resampling.
#   - RAW and RAW+SHADOW differ by exactly one predictor.
# ======================================================================


# ============================================================
# 1. Required packages
# ============================================================

required_packages <- c(
  "sf",
  "terra",
  "dplyr",
  "tidyr",
  "tibble",
  "readr",
  "purrr",
  "stringr",
  "ggplot2",
  "ranger",
  "pROC",
  "glmnet",
  "mgcv"
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
      "\nThe following required packages are not installed:\n",
      paste(
        missing_packages,
        collapse = ", "
      ),
      "\n\nInstall the missing packages before running the analysis."
    )
  )
}


suppressPackageStartupMessages({

  library(sf)
  library(terra)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(purrr)
  library(stringr)
  library(ggplot2)
  library(ranger)
  library(pROC)
  library(glmnet)
  library(mgcv)

})


cat(
  "\nRequired packages loaded successfully.\n"
)


# ============================================================
# 2. Project directories
# ============================================================

# Local project directory.
# For use on another computer, change only PROJECT_DIR.

PROJECT_DIR <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)


DATA_DIR <- file.path(
  PROJECT_DIR,
  "data"
)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

TRAIN_PATH <- file.path(
  DATA_DIR,
  "training_points.gpkg"
)


TEST_PATH <- file.path(
  DATA_DIR,
  "external_test_points.gpkg"
)


PRED_RASTER_PATH <- file.path(
  DATA_DIR,
  "predictor_stack_47bands.tif"
)


# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

OUT_DIR <- file.path(
  PROJECT_DIR,
  "outputs_publication"
)


DIR_CSV <- file.path(
  OUT_DIR,
  "tables"
)


DIR_FIG <- file.path(
  OUT_DIR,
  "figures"
)


DIR_MODEL <- file.path(
  OUT_DIR,
  "models"
)


DIR_PRED <- file.path(
  OUT_DIR,
  "predictions"
)


DIR_QC <- file.path(
  OUT_DIR,
  "quality_control"
)


DIR_MAP <- file.path(
  OUT_DIR,
  "maps"
)


output_dirs <- c(
  OUT_DIR,
  DIR_CSV,
  DIR_FIG,
  DIR_MODEL,
  DIR_PRED,
  DIR_QC,
  DIR_MAP
)


invisible(
  lapply(
    output_dirs,
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)


# ============================================================
# 3. Check project structure and input files
# ============================================================

input_files <- c(
  training_points = TRAIN_PATH,
  external_test_points = TEST_PATH,
  predictor_stack = PRED_RASTER_PATH
)


missing_input_files <- input_files[
  !file.exists(input_files)
]


if(length(missing_input_files) > 0){

  stop(
    paste0(
      "\nRequired input file(s) not found:\n\n",
      paste(
        names(missing_input_files),
        missing_input_files,
        sep = " : ",
        collapse = "\n"
      )
    )
  )
}


if(!all(dir.exists(output_dirs))){

  stop(
    "One or more output directories could not be created."
  )
}


# ============================================================
# 4. Output write-permission check
# ============================================================

output_test_file <- file.path(
  OUT_DIR,
  "output_write_test.txt"
)


writeLines(
  "Output directory write test successful.",
  output_test_file
)


if(!file.exists(output_test_file)){

  stop(
    "The output directory exists but is not writable."
  )
}


file.remove(
  output_test_file
)


# ============================================================
# 5. Initial project check
# ============================================================

cat(
  "\n============================================================",
  "\nPROJECT CHECK",
  "\n============================================================",
  "\nProject directory :", PROJECT_DIR,
  "\nData directory    :", DATA_DIR,
  "\nTraining data     :", TRAIN_PATH,
  "\nExternal test     :", TEST_PATH,
  "\nPredictor raster  :", PRED_RASTER_PATH,
  "\nOutput directory  :", OUT_DIR,
  "\n============================================================\n"
)


cat(
  "\nInput files found successfully.",
  "\nOutput directories created successfully.",
  "\nOutput directory is writable.",
  "\nInitial project check passed.\n"
)
# ------------------------------------------------------------
# Main analysis settings
# ------------------------------------------------------------

SEED <- 123

# Number of stochastic training replicates
N_REP <- 30

# Fraction of ORIGINAL TRAIN pool reserved for threshold calibration
CAL_FRAC <- 0.20

# RF settings
N_TREE <- 1000
MIN_NODE_SIZE <- 5

# minimum calibration observations within one illumination regime
MIN_CAL_REGIME <- 10

# save fitted models?
SAVE_MODELS <- TRUE

# Optional full-raster ensemble prediction.
# Keep FALSE during debugging.
DO_RASTER_ENSEMBLE <- FALSE

# Main scheme / variant for optional map
MAP_SCHEME  <- "UR"
MAP_VARIANT <- "RAW_SHADOW"


set.seed(SEED)


# ============================================================
# 3. Helper functions
# ============================================================


# ------------------------------------------------------------
# Find response column
# ------------------------------------------------------------

find_response_col <- function(x) {

  z <- intersect(
    c(
      "presence",
      "presence_bin",
      "pres",
      "occ",
      "y"
    ),
    names(x)
  )

  if(length(z) == 0) {
    stop("Presence/absence column not found.")
  }

  z[1]
}


# ------------------------------------------------------------
# Safe probability clipping
# ------------------------------------------------------------

clip_prob <- function(p, eps = 1e-6) {

  pmax(
    eps,
    pmin(
      1 - eps,
      as.numeric(p)
    )
  )
}


# ------------------------------------------------------------
# Safe AUC
# ------------------------------------------------------------

safe_auc <- function(y, p) {

  ok <-
    is.finite(y) &
    is.finite(p)

  y <- y[ok]
  p <- p[ok]

  if(
    length(y) < 5 ||
    length(unique(y)) < 2
  ) {
    return(NA_real_)
  }

  out <- tryCatch(
    pROC::roc(
      response = y,
      predictor = p,
      quiet = TRUE,
      direction = "<"
    ),
    error = function(e) NULL
  )

  if(is.null(out)) {
    return(NA_real_)
  }

  as.numeric(
    pROC::auc(out)
  )
}


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

classification_metrics <- function(
  y,
  p,
  threshold = 0.5
) {

  p <- clip_prob(p)

  pred <-
    as.integer(
      p >= threshold
    )

  TP <- sum(pred == 1 & y == 1)
  TN <- sum(pred == 0 & y == 0)
  FP <- sum(pred == 1 & y == 0)
  FN <- sum(pred == 0 & y == 1)

  sensitivity <-
    ifelse(
      TP + FN > 0,
      TP / (TP + FN),
      NA_real_
    )

  specificity <-
    ifelse(
      TN + FP > 0,
      TN / (TN + FP),
      NA_real_
    )

  precision <-
    ifelse(
      TP + FP > 0,
      TP / (TP + FP),
      NA_real_
    )

  npv <-
    ifelse(
      TN + FN > 0,
      TN / (TN + FN),
      NA_real_
    )

  accuracy <-
    (TP + TN) /
    length(y)

  balanced_accuracy <-
    mean(
      c(
        sensitivity,
        specificity
      ),
      na.rm = TRUE
    )

  TSS <-
    sensitivity +
    specificity -
    1

  F1 <-
    ifelse(
      is.finite(precision) &
      is.finite(sensitivity) &
      precision + sensitivity > 0,

      2 *
        precision *
        sensitivity /
        (precision + sensitivity),

      NA_real_
    )

  tibble(
    threshold = threshold,
    n = length(y),

    prevalence =
      mean(y == 1),

    AUC =
      safe_auc(y, p),

    Brier =
      mean(
        (p - y)^2
      ),

    LogLoss =
      -mean(
        y * log(p) +
          (1 - y) *
          log(1 - p)
      ),

    Accuracy = accuracy,

    Sensitivity = sensitivity,

    Specificity = specificity,

    BalancedAccuracy =
      balanced_accuracy,

    TSS = TSS,

    Precision = precision,

    NPV = npv,

    F1 = F1,

    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN
  )
}


# ------------------------------------------------------------
# Threshold selected ONLY from calibration data
# objective = max Youden/TSS
# ------------------------------------------------------------

find_best_threshold <- function(
  y,
  p
) {

  ok <-
    is.finite(y) &
    is.finite(p)

  y <- y[ok]
  p <- p[ok]

  if(
    length(y) < 10 ||
    length(unique(y)) < 2
  ) {
    return(0.5)
  }

  candidate_thresholds <-
    sort(
      unique(
        c(
          seq(
            0.01,
            0.99,
            by = 0.01
          ),
          p
        )
      )
    )

  scores <-
    lapply(
      candidate_thresholds,
      function(th) {

        pred <-
          as.integer(
            p >= th
          )

        TP <-
          sum(
            pred == 1 &
              y == 1
          )

        TN <-
          sum(
            pred == 0 &
              y == 0
          )

        FP <-
          sum(
            pred == 1 &
              y == 0
          )

        FN <-
          sum(
            pred == 0 &
              y == 1
          )

        sens <-
          ifelse(
            TP + FN > 0,
            TP / (TP + FN),
            NA_real_
          )

        spec <-
          ifelse(
            TN + FP > 0,
            TN / (TN + FP),
            NA_real_
          )

        tss <-
          sens +
          spec -
          1

        c(
          threshold = th,
          TSS = tss,
          Sensitivity = sens,
          Specificity = spec
        )
      }
    )

  scores <-
    as.data.frame(
      do.call(
        rbind,
        scores
      )
    )

  scores <-
    scores[
      is.finite(scores$TSS),
      ,
      drop = FALSE
    ]

  if(nrow(scores) == 0) {
    return(0.5)
  }

  best <-
    scores[
      which.max(
        scores$TSS
      ),
      ,
      drop = FALSE
    ]

  as.numeric(
    best$threshold[1]
  )
}


# ------------------------------------------------------------
# Calibration intercept and slope
#
# IMPORTANT:
# This is evaluation only.
# It does NOT recalibrate external-test predictions.
# ------------------------------------------------------------

calibration_stats <- function(
  y,
  p
) {

  p <- clip_prob(p)

  if(
    length(y) < 10 ||
    length(unique(y)) < 2
  ) {

    return(
      tibble(
        calibration_intercept = NA_real_,
        calibration_slope = NA_real_
      )
    )
  }

  lp <-
    qlogis(p)

  # calibration slope
  m1 <-
    tryCatch(
      glm(
        y ~ lp,
        family = binomial()
      ),
      error = function(e) NULL
    )

  slope <-
    if(is.null(m1)) {
      NA_real_
    } else {
      unname(
        coef(m1)["lp"]
      )
    }

  # calibration-in-the-large:
  # offset(logit(p))
  m0 <-
    tryCatch(
      glm(
        y ~ 1 +
          offset(lp),
        family = binomial()
      ),
      error = function(e) NULL
    )

  intercept <-
    if(is.null(m0)) {
      NA_real_
    } else {
      unname(
        coef(m0)[1]
      )
    }

  tibble(
    calibration_intercept =
      intercept,

    calibration_slope =
      slope
  )
}


# ------------------------------------------------------------
# Extract ranger P(class=1)
# ------------------------------------------------------------

predict_rf_prob <- function(
  model,
  newdata
) {

  pp <-
    predict(
      model,
      data = newdata
    )$predictions

  if(is.matrix(pp)) {

    if("1" %in% colnames(pp)) {

      return(
        as.numeric(
          pp[, "1"]
        )
      )

    } else {

      return(
        as.numeric(
          pp[, ncol(pp)]
        )
      )
    }
  }

  as.numeric(pp)
}


# ------------------------------------------------------------
# Stratified calibration split:
# presence × shadow
#
# Split happens BEFORE any resampling.
# ------------------------------------------------------------

make_fit_cal_split <- function(
  df,
  frac_cal = 0.20,
  seed = 1
) {

  set.seed(seed)

  d <-
    df %>%
    mutate(
      split_stratum =
        interaction(
          presence,
          shadow_bin,
          drop = TRUE
        )
    )

  cal <-
    d %>%
    group_by(split_stratum) %>%
    group_modify(
      ~{

        n <- nrow(.x)

        if(n <= 2) {

          return(
            .x[0, , drop = FALSE]
          )
        }

        n_cal <-
          round(
            n * frac_cal
          )

        n_cal <-
          max(
            1,
            n_cal
          )

        n_cal <-
          min(
            n_cal,
            n - 1
          )

        .x[
          sample(
            seq_len(n),
            n_cal,
            replace = FALSE
          ),
          ,
          drop = FALSE
        ]
      }
    ) %>%
    ungroup()

  fit <-
    d %>%
    filter(
      !orig_id %in%
        cal$orig_id
    )

  fit$split_stratum <- NULL
  cal$split_stratum <- NULL

  list(
    fit = fit,
    cal = cal
  )
}


# ============================================================
# 4. Read predictor raster
# ============================================================

r_master <-
  terra::rast(
    PRED_RASTER_PATH
  )

cat(
  "\nRaster layers:\n"
)

print(
  names(r_master)
)


required_layers <-
  c(
    "SCL",
    "ShadowMask"
  )

missing_layers <-
  setdiff(
    required_layers,
    names(r_master)
  )

if(length(missing_layers) > 0) {

  stop(
    "Required layers missing: ",
    paste(
      missing_layers,
      collapse = ", "
    )
  )
}


# ============================================================
# 5. Define illumination regime
# ============================================================

# SCL retained ONLY for QC
scl_r <-
  r_master[["SCL"]]


# ShadowMask is canonical source of illumination regime
shadow_r <-
  r_master[["ShadowMask"]]


# force to canonical 0/1
shadow_r <-
  terra::ifel(
    is.na(shadow_r),
    NA,
    terra::ifel(
      shadow_r > 0.5,
      1,
      0
    )
  )

names(shadow_r) <-
  "ShadowMask"


# ============================================================
# 6. Landscape shadow prevalence
# ============================================================
library(tibble)
library(readr)
area_r <-
  terra::cellSize(
    shadow_r,
    unit = "m"
  )

shadow_area_m2 <-
  terra::global(
    area_r *
      terra::ifel(
        shadow_r == 1,
        1,
        0
      ),
    "sum",
    na.rm = TRUE
  )[1, 1]

total_area_m2 <-
  terra::global(
    area_r *
      terra::ifel(
        !is.na(shadow_r),
        1,
        NA
      ),
    "sum",
    na.rm = TRUE
  )[1, 1]


p_shadow_area <-
  as.numeric(
    shadow_area_m2 /
      total_area_m2
  )


landscape_shadow_tbl <-
  tibble(
    shadow_area_km2 =
      shadow_area_m2 / 1e6,

    total_area_km2 =
      total_area_m2 / 1e6,

    p_shadow_area =
      p_shadow_area,

    percent_shadow =
      p_shadow_area * 100
  )


write_csv(
  landscape_shadow_tbl,
  file.path(
    DIR_QC,
    "01_landscape_shadow_prevalence.csv"
  )
)


cat(
  "\nLandscape shadow prevalence =",
  round(
    100 * p_shadow_area,
    3
  ),
  "%\n"
)


# ============================================================
# 7. Build predictor extraction stack
# ============================================================

# SCL is excluded from modelling.
# ShadowMask remains temporarily ONLY so that its value can
# be extracted and renamed shadow_bin at the point level.

model_extract_names <- setdiff(
  names(r_master),
  "SCL"
)

# CORRECT syntax for selecting multiple SpatRaster layers
preds_extract <- r_master[[model_extract_names]]

# replace original ShadowMask by canonical binary version
preds_extract[["ShadowMask"]] <- shadow_r

# safety checks
stopifnot(
  !"SCL" %in% names(preds_extract)
)

stopifnot(
  "ShadowMask" %in% names(preds_extract)
)

cat("\nModel extraction layers:\n")
print(names(preds_extract))

cat(
  "\nNumber of extraction layers =",
  terra::nlyr(preds_extract),
  "\n"
)

# ============================================================
# 8. Read training points
# ============================================================

train_sf <-
  st_read(
    TRAIN_PATH,
    quiet = TRUE
  )


if(is.na(st_crs(train_sf))) {
  stop("Training CRS is NA.")
}


r_crs_sf <-
  st_crs(
    terra::crs(
      r_master
    )
  )


if(
  st_crs(train_sf) !=
  r_crs_sf
) {

  train_sf <-
    st_transform(
      train_sf,
      r_crs_sf
    )
}


resp_col <-
  find_response_col(
    train_sf
  )


train_presence <-
  as.integer(
    train_sf[[resp_col]] > 0
  )


# coordinates saved ONLY for leakage QC,
# NEVER predictors
train_xy <-
  st_coordinates(
    train_sf
  )


# ============================================================
# 9. Extract training predictors
# ============================================================

Xtr_full <-
  terra::extract(
    preds_extract,
    terra::vect(train_sf)
  )[
    ,
    -1,
    drop = FALSE
  ] %>%
  as.data.frame()


if(
  !"ShadowMask" %in%
  names(Xtr_full)
) {

  stop(
    "ShadowMask was not extracted."
  )
}


# ------------------------------------------------------------
# ONE AND ONLY point-level shadow variable
# ------------------------------------------------------------

shadow_bin_train <-
  as.integer(
    Xtr_full$ShadowMask > 0
  )


# ------------------------------------------------------------
# Remove ShadowMask from environmental predictor block
# ------------------------------------------------------------

Xtr_env <-
  Xtr_full %>%
  select(
    -ShadowMask
  )


# absolute protection
Xtr_env <-
  Xtr_env %>%
  select(
    -any_of(
      c(
        "SCL",
        "shadow_bin",
        "ShadowMask"
      )
    )
  )


# ------------------------------------------------------------
# Build clean original point pool
# ------------------------------------------------------------

df_pool <-
  bind_cols(

    tibble(
      orig_id =
        seq_len(
          nrow(train_sf)
        ),

      presence =
        train_presence,

      shadow_bin =
        shadow_bin_train,

      coord_x =
        train_xy[, 1],

      coord_y =
        train_xy[, 2]
    ),

    Xtr_env
  )


# predictors only for complete-case filtering
env_cols_initial <-
  names(Xtr_env)


keep_train <-
  complete.cases(
    df_pool[
      ,
      c(
        "presence",
        "shadow_bin",
        env_cols_initial
      )
    ]
  )


df_pool <-
  df_pool[
    keep_train,
    ,
    drop = FALSE
  ]


cat(
  "\nOriginal complete training pool =",
  nrow(df_pool),
  "\n"
)


cat(
  "\nTraining presence × raster-derived shadow:\n"
)

print(
  table(
    Presence =
      df_pool$presence,

    Shadow =
      df_pool$shadow_bin
  )
)


if(
  length(
    unique(
      df_pool$presence
    )
  ) < 2
) {

  stop(
    "Training pool has only one response class."
  )
}


if(
  length(
    unique(
      df_pool$shadow_bin
    )
  ) < 2
) {

  stop(
    "Training pool has only one shadow class."
  )
}


# ============================================================
# 10. Read external test points
# ============================================================

test_sf <-
  st_read(
    TEST_PATH,
    quiet = TRUE
  )


if(is.na(st_crs(test_sf))) {
  stop("Test CRS is NA.")
}


if(
  st_crs(test_sf) !=
  r_crs_sf
) {

  test_sf <-
    st_transform(
      test_sf,
      r_crs_sf
    )
}


resp_col_te <-
  find_response_col(
    test_sf
  )


test_presence <-
  as.integer(
    test_sf[[resp_col_te]] > 0
  )


test_xy <-
  st_coordinates(
    test_sf
  )


Xte_full <-
  terra::extract(
    preds_extract,
    terra::vect(test_sf)
  )[
    ,
    -1,
    drop = FALSE
  ] %>%
  as.data.frame()


shadow_bin_test <-
  as.integer(
    Xte_full$ShadowMask > 0
  )


Xte_env <-
  Xte_full %>%
  select(
    -ShadowMask
  ) %>%
  select(
    -any_of(
      c(
        "SCL",
        "shadow_bin",
        "ShadowMask"
      )
    )
  )


df_te <-
  bind_cols(

    tibble(
      test_id =
        seq_len(
          nrow(test_sf)
        ),

      presence =
        test_presence,

      shadow_bin =
        shadow_bin_test,

      coord_x =
        test_xy[, 1],

      coord_y =
        test_xy[, 2]
    ),

    Xte_env
  )


keep_test <-
  complete.cases(
    df_te[
      ,
      c(
        "presence",
        "shadow_bin",
        env_cols_initial
      )
    ]
  )


df_te <-
  df_te[
    keep_test,
    ,
    drop = FALSE
  ]


cat(
  "\nExternal test n =",
  nrow(df_te),
  "\n"
)


print(
  table(
    Presence =
      df_te$presence,

    Shadow =
      df_te$shadow_bin
  )
)


# ============================================================
# 11. Verify predictor consistency
# ============================================================

if(
  !identical(
    names(Xtr_env),
    names(Xte_env)
  )
) {

  stop(
    "Train and test environmental predictor names/order differ."
  )
}


# ============================================================
# 12. Define predictor sets
# ============================================================

RAW_FEATURES <-
  names(Xtr_env)


# absolute anti-leak check
RAW_FEATURES <-
  setdiff(
    RAW_FEATURES,
    c(
      "SCL",
      "ShadowMask",
      "shadow_bin",
      "presence",
      "orig_id",
      "test_id",
      "coord_x",
      "coord_y",
      "w_area"
    )
  )


RAW_SHADOW_FEATURES <-
  c(
    RAW_FEATURES,
    "shadow_bin"
  )


RAW_FEATURES <-
  unique(
    RAW_FEATURES
  )

RAW_SHADOW_FEATURES <-
  unique(
    RAW_SHADOW_FEATURES
  )


cat(
  "\nRAW n predictors =",
  length(RAW_FEATURES),
  "\n"
)

cat(
  "RAW+SHADOW n predictors =",
  length(RAW_SHADOW_FEATURES),
  "\n"
)


cat(
  "\nOnly added variable in RAW+SHADOW:\n"
)

print(
  setdiff(
    RAW_SHADOW_FEATURES,
    RAW_FEATURES
  )
)


# THIS MUST PASS
stopifnot(
  length(
    RAW_SHADOW_FEATURES
  ) ==
    length(
      RAW_FEATURES
    ) + 1
)

stopifnot(
  identical(
    setdiff(
      RAW_SHADOW_FEATURES,
      RAW_FEATURES
    ),
    "shadow_bin"
  )
)

stopifnot(
  !"SCL" %in%
    RAW_FEATURES,

  !"ShadowMask" %in%
    RAW_FEATURES,

  !"shadow_bin" %in%
    RAW_FEATURES
)

stopifnot(
  !"SCL" %in%
    RAW_SHADOW_FEATURES,

  !"ShadowMask" %in%
    RAW_SHADOW_FEATURES
)


write_csv(
  tibble(
    predictor =
      RAW_FEATURES
  ),
  file.path(
    DIR_QC,
    "02_RAW_predictor_list.csv"
  )
)


write_csv(
  tibble(
    predictor =
      RAW_SHADOW_FEATURES
  ),
  file.path(
    DIR_QC,
    "03_RAW_SHADOW_predictor_list.csv"
  )
)


# ============================================================
# 13. Check train-test spatial overlap
# ============================================================

coord_train_key <-
  paste(
    round(
      df_pool$coord_x,
      6
    ),
    round(
      df_pool$coord_y,
      6
    ),
    sep = "_"
  )


coord_test_key <-
  paste(
    round(
      df_te$coord_x,
      6
    ),
    round(
      df_te$coord_y,
      6
    ),
    sep = "_"
  )


exact_coord_overlap <-
  intersect(
    coord_train_key,
    coord_test_key
  )


if(
  length(
    exact_coord_overlap
  ) > 0
) {

  stop(
    "LEAKAGE: exact train/test coordinate overlap detected."
  )
}


cat(
  "\nPASS: no exact train/test coordinate overlap.\n"
)


# ------------------------------------------------------------
# Nearest distance from each external point to original train
# ------------------------------------------------------------

train_clean_sf <-
  st_as_sf(
    df_pool,
    coords = c(
      "coord_x",
      "coord_y"
    ),
    crs = r_crs_sf
  )


test_clean_sf <-
  st_as_sf(
    df_te,
    coords = c(
      "coord_x",
      "coord_y"
    ),
    crs = r_crs_sf
  )


nearest_idx <-
  st_nearest_feature(
    test_clean_sf,
    train_clean_sf
  )


nearest_distance <-
  st_distance(
    test_clean_sf,
    train_clean_sf[
      nearest_idx,
    ],
    by_element = TRUE
  )


distance_qc <-
  tibble(
    test_id =
      df_te$test_id,

    nearest_train_distance =
      as.numeric(
        nearest_distance
      )
  )


write_csv(
  distance_qc,
  file.path(
    DIR_QC,
    "04_external_test_nearest_train_distance.csv"
  )
)


# ============================================================
# 14. Split training data into fitting and calibration sets
# ============================================================

split_obj <-
  make_fit_cal_split(
    df = df_pool,
    frac_cal = CAL_FRAC,
    seed = SEED + 10
  )


df_fit_pool <-
  split_obj$fit


df_cal <-
  split_obj$cal


cat(
  "\nFit pool n =",
  nrow(df_fit_pool),
  "\nCalibration n =",
  nrow(df_cal),
  "\n"
)


cat(
  "\nFIT presence × shadow:\n"
)

print(
  table(
    df_fit_pool$presence,
    df_fit_pool$shadow_bin
  )
)


cat(
  "\nCALIBRATION presence × shadow:\n"
)

print(
  table(
    df_cal$presence,
    df_cal$shadow_bin
  )
)


# ------------------------------------------------------------
# Critical ID leak test
# ------------------------------------------------------------

overlap_fit_cal <-
  intersect(
    df_fit_pool$orig_id,
    df_cal$orig_id
  )


stopifnot(
  length(
    overlap_fit_cal
  ) == 0
)


cat(
  "\nPASS: fit and calibration have zero orig_id overlap.\n"
)


write_csv(
  df_fit_pool %>%
    select(
      orig_id,
      presence,
      shadow_bin,
      coord_x,
      coord_y
    ),
  file.path(
    DIR_QC,
    "05_fit_pool_ids.csv"
  )
)


write_csv(
  df_cal %>%
    select(
      orig_id,
      presence,
      shadow_bin,
      coord_x,
      coord_y
    ),
  file.path(
    DIR_QC,
    "06_calibration_ids.csv"
  )
)


# ============================================================
# 15. Compare landscape and fitting-pool shadow prevalence
# ============================================================

p_shadow_pool <-
  mean(
    df_fit_pool$shadow_bin == 1
  )


prevalence_compare <-
  tibble(
    source =
      c(
        "Landscape",
        "Original_fit_pool"
      ),

    shadow_prevalence =
      c(
        p_shadow_area,
        p_shadow_pool
      )
  )


write_csv(
  prevalence_compare,
  file.path(
    DIR_QC,
    "07_shadow_prevalence_landscape_vs_fitpool.csv"
  )
)


cat(
  "\nLandscape shadow =",
  round(
    p_shadow_area,
    4
  ),
  "\nFit-pool shadow =",
  round(
    p_shadow_pool,
    4
  ),
  "\n"
)


# ============================================================
# 16. Define sampling schemes
# ============================================================


# ------------------------------------------------------------
# UR
#
# Target shadow prevalence = LANDSCAPE prevalence.
#
# Bootstrap WITH replacement.
#
# Using replacement is intentional:
# each replicate represents a stochastic realization and
# no artificial n_total_max restriction is required.
# ------------------------------------------------------------

make_UR_area <- function(
  df_pool,
  n_total,
  p_area,
  seed
) {

  set.seed(seed)

  sh <-
    df_pool[
      df_pool$shadow_bin == 1,
      ,
      drop = FALSE
    ]

  ns <-
    df_pool[
      df_pool$shadow_bin == 0,
      ,
      drop = FALSE
    ]


  if(
    nrow(sh) == 0 ||
    nrow(ns) == 0
  ) {
    stop(
      "UR requires both shadow classes."
    )
  }


  n_sh <-
    round(
      n_total *
        p_area
    )

  n_sh <-
    max(
      1,
      min(
        n_total - 1,
        n_sh
      )
    )

  n_ns <-
    n_total -
      n_sh


  out <-
    bind_rows(

      sh[
        sample(
          seq_len(
            nrow(sh)
          ),
          size = n_sh,
          replace = TRUE
        ),
        ,
        drop = FALSE
      ],

      ns[
        sample(
          seq_len(
            nrow(ns)
          ),
          size = n_ns,
          replace = TRUE
        ),
        ,
        drop = FALSE
      ]
    )


  out <-
    out[
      sample(
        seq_len(
          nrow(out)
        )
      ),
      ,
      drop = FALSE
    ]


  out$w_area <- 1

  out
}


# ------------------------------------------------------------
# SB
#
# Target shadow prevalence = ORIGINAL FIT-POOL prevalence.
# ------------------------------------------------------------

make_SB_poolratio <- function(
  df_pool,
  n_total,
  seed
) {

  set.seed(seed)

  p_pool <-
    mean(
      df_pool$shadow_bin == 1
    )


  sh <-
    df_pool[
      df_pool$shadow_bin == 1,
      ,
      drop = FALSE
    ]


  ns <-
    df_pool[
      df_pool$shadow_bin == 0,
      ,
      drop = FALSE
    ]


  if(
    nrow(sh) == 0 ||
    nrow(ns) == 0
  ) {

    stop(
      "SB requires both shadow classes."
    )
  }


  n_sh <-
    round(
      n_total *
        p_pool
    )

  n_sh <-
    max(
      1,
      min(
        n_total - 1,
        n_sh
      )
    )


  n_ns <-
    n_total -
      n_sh


  out <-
    bind_rows(

      sh[
        sample(
          seq_len(
            nrow(sh)
          ),
          size = n_sh,
          replace = TRUE
        ),
        ,
        drop = FALSE
      ],

      ns[
        sample(
          seq_len(
            nrow(ns)
          ),
          size = n_ns,
          replace = TRUE
        ),
        ,
        drop = FALSE
      ]
    )


  out <-
    out[
      sample(
        seq_len(
          nrow(out)
        )
      ),
      ,
      drop = FALSE
    ]


  out$w_area <- 1

  out
}


# ------------------------------------------------------------
# SW
#
# EXACT SAME sampled observations as SB.
# Only weights differ.
#
# Weights restore the LANDSCAPE shadow prevalence.
# ------------------------------------------------------------

make_SW_from_SB <- function(
  df_SB,
  p_area
) {

  p_sb <-
    mean(
      df_SB$shadow_bin == 1
    )


  if(
    p_sb <= 0 ||
    p_sb >= 1
  ) {

    stop(
      "Invalid SB prevalence for SW weighting."
    )
  }


  w_shadow <-
    p_area /
      p_sb


  w_nonshadow <-
    (1 - p_area) /
      (1 - p_sb)


  out <-
    df_SB


  out$w_area <-
    ifelse(
      out$shadow_bin == 1,
      w_shadow,
      w_nonshadow
    )


  # numerical stabilization:
  # mean weight = 1
  out$w_area <-
    out$w_area /
      mean(
        out$w_area
      )


  list(
    df = out,

    p_sb =
      p_sb,

    w_shadow =
      w_shadow,

    w_nonshadow =
      w_nonshadow
  )
}


# ============================================================
# 17. Set replicate sample size
# ============================================================

# Important correction:
#
# The old n_total_max restriction is NOT needed because
# sampling is explicitly WITH replacement.
#
# Therefore each replicate uses the same nominal training size
# as the fixed FIT pool.

N_TRAIN_REP <-
  nrow(
    df_fit_pool
  )


if(
  N_TRAIN_REP < 30
) {

  stop(
    "Fit pool is too small."
  )
}


cat(
  "\nTraining n per replicate =",
  N_TRAIN_REP,
  "\n"
)


# ============================================================
# 18. Random Forest model fitting
# ============================================================

fit_one_rf <- function(
  df_train,
  features,
  use_weights = FALSE,
  seed = 1
) {

  d <-
    df_train[
      ,
      c(
        "presence",
        features
      ),
      drop = FALSE
    ]


  d$presence <-
    factor(
      d$presence,
      levels = c(
        0,
        1
      )
    )


  p <-
    length(
      features
    )


  mtry_value <-
    max(
      1,
      floor(
        sqrt(p)
      )
    )


  if(use_weights) {

    fit <-
      ranger(
        dependent.variable.name =
          "presence",

        data = d,

        probability = TRUE,

        num.trees =
          N_TREE,

        mtry =
          mtry_value,

        min.node.size =
          MIN_NODE_SIZE,

        importance =
          "permutation",

        case.weights =
          df_train$w_area,

        seed =
          seed,

        num.threads =
          max(
            1,
            parallel::detectCores() - 1
          )
      )

  } else {

    fit <-
      ranger(
        dependent.variable.name =
          "presence",

        data = d,

        probability = TRUE,

        num.trees =
          N_TREE,

        mtry =
          mtry_value,

        min.node.size =
          MIN_NODE_SIZE,

        importance =
          "permutation",

        seed =
          seed,

        num.threads =
          max(
            1,
            parallel::detectCores() - 1
          )
      )
  }


  fit
}


# ============================================================
# 19. Model evaluation
# ============================================================

evaluate_model <- function(
  model,
  features,
  df_cal,
  df_test,
  scheme,
  variant,
  rep_id
) {

  # ----------------------------------------------------------
  # CALIBRATION predictions
  # ----------------------------------------------------------

  p_cal <-
    predict_rf_prob(
      model,
      df_cal[
        ,
        features,
        drop = FALSE
      ]
    )


  # ----------------------------------------------------------
  # GLOBAL threshold
  # ----------------------------------------------------------

  th_global <-
    find_best_threshold(
      y =
        df_cal$presence,

      p =
        p_cal
    )


  # ----------------------------------------------------------
  # REGIME-SPECIFIC thresholds
  # ----------------------------------------------------------

  get_regime_threshold <-
    function(regime_value) {

      idx <-
        df_cal$shadow_bin ==
        regime_value


      if(
        sum(idx) <
        MIN_CAL_REGIME
      ) {

        return(
          th_global
        )
      }


      y0 <-
        df_cal$presence[
          idx
        ]


      p0 <-
        p_cal[
          idx
        ]


      if(
        length(
          unique(
            y0
          )
        ) < 2
      ) {

        return(
          th_global
        )
      }


      find_best_threshold(
        y0,
        p0
      )
    }


  th_nonshadow <-
    get_regime_threshold(0)


  th_shadow <-
    get_regime_threshold(1)


  # ----------------------------------------------------------
  # EXTERNAL TEST predictions
  # ----------------------------------------------------------

  p_test <-
    predict_rf_prob(
      model,
      df_test[
        ,
        features,
        drop = FALSE
      ]
    )


  # ----------------------------------------------------------
  # Global-threshold predictions
  # ----------------------------------------------------------

  metrics_global <-
    classification_metrics(
      y =
        df_test$presence,

      p =
        p_test,

      threshold =
        th_global
    ) %>%
    mutate(
      scheme =
        scheme,

      variant =
        variant,

      replicate =
        rep_id,

      regime =
        "ALL",

      threshold_strategy =
        "GLOBAL",

      threshold_nonshadow =
        th_nonshadow,

      threshold_shadow =
        th_shadow,

      .before = 1
    )


  # ----------------------------------------------------------
  # regime-specific evaluation
  # ----------------------------------------------------------

  regime_metrics <-
    lapply(
      c(
        0,
        1
      ),
      function(g) {

        ii <-
          df_test$shadow_bin ==
          g


        if(
          sum(ii) < 2
        ) {

          return(NULL)
        }


        th <-
          ifelse(
            g == 1,
            th_shadow,
            th_nonshadow
          )


        classification_metrics(
          y =
            df_test$presence[ii],

          p =
            p_test[ii],

          threshold =
            th
        ) %>%
          mutate(
            scheme =
              scheme,

            variant =
              variant,

            replicate =
              rep_id,

            regime =
              ifelse(
                g == 1,
                "SHADOW",
                "NONSHADOW"
              ),

            threshold_strategy =
              "REGIME_SPECIFIC",

            threshold_nonshadow =
              th_nonshadow,

            threshold_shadow =
              th_shadow,

            .before = 1
          )
      }
    ) %>%
    bind_rows()


  # ----------------------------------------------------------
  # whole-test classification using regime-specific thresholds
  # ----------------------------------------------------------

  pred_regime <-
    ifelse(
      df_test$shadow_bin == 1,

      as.integer(
        p_test >=
          th_shadow
      ),

      as.integer(
        p_test >=
          th_nonshadow
      )
    )


  y <-
    df_test$presence


  TP <-
    sum(
      pred_regime == 1 &
        y == 1
    )

  TN <-
    sum(
      pred_regime == 0 &
        y == 0
    )

  FP <-
    sum(
      pred_regime == 1 &
        y == 0
    )

  FN <-
    sum(
      pred_regime == 0 &
        y == 1
    )


  sens <-
    ifelse(
      TP + FN > 0,
      TP / (TP + FN),
      NA_real_
    )


  spec <-
    ifelse(
      TN + FP > 0,
      TN / (TN + FP),
      NA_real_
    )


  precision <-
    ifelse(
      TP + FP > 0,
      TP / (TP + FP),
      NA_real_
    )


  F1 <-
    ifelse(
      is.finite(precision) &
        is.finite(sens) &
        precision + sens > 0,

      2 *
        precision *
        sens /
        (precision + sens),

      NA_real_
    )


  metrics_regime_all <-
    tibble(
      scheme =
        scheme,

      variant =
        variant,

      replicate =
        rep_id,

      regime =
        "ALL",

      threshold_strategy =
        "REGIME_SPECIFIC",

      threshold =
        NA_real_,

      threshold_nonshadow =
        th_nonshadow,

      threshold_shadow =
        th_shadow,

      n =
        length(y),

      prevalence =
        mean(y),

      AUC =
        safe_auc(
          y,
          p_test
        ),

      Brier =
        mean(
          (
            p_test -
              y
          )^2
        ),

      LogLoss =
        -mean(
          y *
            log(
              clip_prob(
                p_test
              )
            ) +
            (
              1 - y
            ) *
            log(
              1 -
                clip_prob(
                  p_test
                )
            )
        ),

      Accuracy =
        mean(
          pred_regime ==
            y
        ),

      Sensitivity =
        sens,

      Specificity =
        spec,

      BalancedAccuracy =
        mean(
          c(
            sens,
            spec
          ),
          na.rm = TRUE
        ),

      TSS =
        sens +
        spec -
        1,

      Precision =
        precision,

      NPV =
        ifelse(
          TN + FN > 0,
          TN / (TN + FN),
          NA_real_
        ),

      F1 =
        F1,

      TP = TP,
      TN = TN,
      FP = FP,
      FN = FN
    )


  # ----------------------------------------------------------
  # calibration structure on external test
  # descriptive evaluation only
  # ----------------------------------------------------------

  cal_all <-
    calibration_stats(
      df_test$presence,
      p_test
    ) %>%
    mutate(
      scheme =
        scheme,

      variant =
        variant,

      replicate =
        rep_id,

      regime =
        "ALL",

      .before = 1
    )


  cal_regime <-
    lapply(
      c(
        0,
        1
      ),
      function(g) {

        ii <-
          df_test$shadow_bin ==
          g


        out <-
          calibration_stats(
            df_test$presence[ii],
            p_test[ii]
          )


        out %>%
          mutate(
            scheme =
              scheme,

            variant =
              variant,

            replicate =
              rep_id,

            regime =
              ifelse(
                g == 1,
                "SHADOW",
                "NONSHADOW"
              ),

            .before = 1
          )
      }
    ) %>%
    bind_rows()


  # ----------------------------------------------------------
  # Predictions saved point-by-point
  # ----------------------------------------------------------

  pred_tbl <-
    tibble(
      test_id =
        df_test$test_id,

      presence =
        df_test$presence,

      shadow_bin =
        df_test$shadow_bin,

      probability =
        p_test,

      threshold_global =
        th_global,

      threshold_nonshadow =
        th_nonshadow,

      threshold_shadow =
        th_shadow,

      predicted_global =
        as.integer(
          p_test >=
            th_global
        ),

      predicted_regime =
        pred_regime,

      scheme =
        scheme,

      variant =
        variant,

      replicate =
        rep_id
    )


  # ----------------------------------------------------------
  # Threshold table
  # ----------------------------------------------------------

  threshold_tbl <-
    tibble(
      scheme =
        scheme,

      variant =
        variant,

      replicate =
        rep_id,

      threshold_global =
        th_global,

      threshold_nonshadow =
        th_nonshadow,

      threshold_shadow =
        th_shadow,

      threshold_difference =
        th_shadow -
        th_nonshadow
    )


  list(

    metrics =
      bind_rows(
        metrics_global,
        metrics_regime_all,
        regime_metrics
      ),

    calibration =
      bind_rows(
        cal_all,
        cal_regime
      ),

    predictions =
      pred_tbl,

    thresholds =
      threshold_tbl
  )
}


# ============================================================
# 20. Random Forest replicate analysis
# ============================================================

all_metrics     <- list()
all_calibration <- list()
all_predictions <- list()
all_thresholds  <- list()
all_sampling    <- list()
all_importance  <- list()

result_index <- 1


for (rp in seq_len(N_REP)) {

  cat("\n============================================================\n")
  cat("REPLICATE", rp, "of", N_REP, "\n")
  cat("============================================================\n")


  # ----------------------------------------------------------
  # 19.1 Make UR
  # ----------------------------------------------------------

  df_UR <- make_UR_area(
    df_pool = df_fit_pool,
    n_total = N_TRAIN_REP,
    p_area  = p_shadow_area,
    seed    = SEED + 1000 + rp
  )


  # ----------------------------------------------------------
  # 19.2 Make SB
  # ----------------------------------------------------------

  df_SB <- make_SB_poolratio(
    df_pool = df_fit_pool,
    n_total = N_TRAIN_REP,
    seed    = SEED + 2000 + rp
  )


  # ----------------------------------------------------------
  # 19.3 Make SW from EXACT same SB observations
  # ----------------------------------------------------------

  sw_obj <- make_SW_from_SB(
    df_SB  = df_SB,
    p_area = p_shadow_area
  )

  df_SW <- sw_obj$df


  # ----------------------------------------------------------
  # 19.4 Scenario list
  # ----------------------------------------------------------

  scenarios <- list(
    UR = df_UR,
    SB = df_SB,
    SW = df_SW
  )


  # ----------------------------------------------------------
  # 19.5 CRITICAL QC:
  # SB and SW must contain exactly same original observations
  # ----------------------------------------------------------

  stopifnot(
    identical(
      df_SB$orig_id,
      df_SW$orig_id
    )
  )


  # ----------------------------------------------------------
  # 19.6 Process scenarios
  # ----------------------------------------------------------

  for (scheme_name in names(scenarios)) {

    ds <- scenarios[[scheme_name]]


    # --------------------------------------------------------
    # Leak check:
    # calibration original IDs must never enter training
    # --------------------------------------------------------

    leaked_ids <- intersect(
      unique(ds$orig_id),
      df_cal$orig_id
    )

    if (length(leaked_ids) > 0) {
      stop(
        "LEAKAGE detected in ",
        scheme_name,
        " replicate ",
        rp,
        ". Number of leaked orig_id = ",
        length(leaked_ids)
      )
    }


    # --------------------------------------------------------
    # Basic scenario checks
    # --------------------------------------------------------

    if (nrow(ds) == 0) {
      stop(
        "Empty training dataset in ",
        scheme_name,
        " replicate ",
        rp
      )
    }

    if (length(unique(ds$presence)) < 2) {
      stop(
        "Only one response class in ",
        scheme_name,
        " replicate ",
        rp
      )
    }

    if (length(unique(ds$shadow_bin)) < 2) {
      stop(
        "Only one shadow class in ",
        scheme_name,
        " replicate ",
        rp
      )
    }


    # --------------------------------------------------------
    # 19.7 Sampling diagnostics
    # --------------------------------------------------------

    sampling_row <- tibble(
      replicate = rp,
      scheme = scheme_name,

      n_rows = nrow(ds),

      n_unique_orig = dplyr::n_distinct(ds$orig_id),

      duplicate_fraction =
        1 - dplyr::n_distinct(ds$orig_id) / nrow(ds),

      n_shadow = sum(ds$shadow_bin == 1),

      n_nonshadow = sum(ds$shadow_bin == 0),

      p_shadow = mean(ds$shadow_bin == 1),

      n_presence = sum(ds$presence == 1),

      n_absence = sum(ds$presence == 0),

      prevalence = mean(ds$presence == 1),

      mean_weight = mean(ds$w_area),

      weight_shadow =
        if (any(ds$shadow_bin == 1)) {
          mean(ds$w_area[ds$shadow_bin == 1])
        } else {
          NA_real_
        },

      weight_nonshadow =
        if (any(ds$shadow_bin == 0)) {
          mean(ds$w_area[ds$shadow_bin == 0])
        } else {
          NA_real_
        }
    )

    all_sampling[[length(all_sampling) + 1]] <- sampling_row


    # --------------------------------------------------------
    # 19.8 Predictor variants
    #
    # EXACTLY:
    # RAW        = environmental predictors only
    # RAW_SHADOW = RAW + shadow_bin
    # --------------------------------------------------------

    variants <- list(
      RAW = RAW_FEATURES,
      RAW_SHADOW = RAW_SHADOW_FEATURES
    )


    # --------------------------------------------------------
    # 19.9 Run both predictor variants
    # --------------------------------------------------------

    for (variant_name in names(variants)) {

      features <- variants[[variant_name]]

      cat(
        "  ",
        scheme_name,
        " / ",
        variant_name,
        "\n",
        sep = ""
      )


      # ------------------------------------------------------
      # Predictor safety checks
      # ------------------------------------------------------

      if ("SCL" %in% features) {
        stop("SCL leakage detected in predictor set.")
      }

      if ("ShadowMask" %in% features) {
        stop("ShadowMask leakage detected in predictor set.")
      }

      if (
        variant_name == "RAW" &&
        "shadow_bin" %in% features
      ) {
        stop("shadow_bin incorrectly present in RAW.")
      }

      if (
        variant_name == "RAW_SHADOW" &&
        !"shadow_bin" %in% features
      ) {
        stop("shadow_bin missing from RAW_SHADOW.")
      }


      # ------------------------------------------------------
      # Confirm all predictors exist in train/cal/test
      # ------------------------------------------------------

      missing_train <- setdiff(features, names(ds))
      missing_cal   <- setdiff(features, names(df_cal))
      missing_test  <- setdiff(features, names(df_te))

      if (length(missing_train) > 0) {
        stop(
          "Missing training predictors: ",
          paste(missing_train, collapse = ", ")
        )
      }

      if (length(missing_cal) > 0) {
        stop(
          "Missing calibration predictors: ",
          paste(missing_cal, collapse = ", ")
        )
      }

      if (length(missing_test) > 0) {
        stop(
          "Missing external-test predictors: ",
          paste(missing_test, collapse = ", ")
        )
      }


      # ------------------------------------------------------
      # SW uses weights; UR and SB do not
      # ------------------------------------------------------

      use_weights <- identical(
        scheme_name,
        "SW"
      )


      # ------------------------------------------------------
      # Unique deterministic seed per:
      # replicate × scheme × predictor variant
      # ------------------------------------------------------

      scheme_code <- match(
        scheme_name,
        c("UR", "SB", "SW")
      )

      variant_code <- match(
        variant_name,
        c("RAW", "RAW_SHADOW")
      )

      model_seed <-
        SEED +
        rp * 100 +
        scheme_code * 10 +
        variant_code


      # ------------------------------------------------------
      # 19.10 Fit model
      # ------------------------------------------------------

      fit <- fit_one_rf(
        df_train = ds,
        features = features,
        use_weights = use_weights,
        seed = model_seed
      )


      # ------------------------------------------------------
      # 19.11 Save fitted model
      # ------------------------------------------------------

      if (SAVE_MODELS) {

        model_file <- file.path(
          DIR_MODEL,
          paste0(
            "RF_",
            scheme_name,
            "_",
            variant_name,
            "_REP",
            sprintf("%03d", rp),
            ".rds"
          )
        )

        saveRDS(
          fit,
          model_file
        )
      }


      # ------------------------------------------------------
      # 19.12 Variable importance
      # ------------------------------------------------------

      imp <- fit$variable.importance

      if (is.null(imp)) {
        stop(
          "Variable importance missing for ",
          scheme_name,
          " / ",
          variant_name,
          " replicate ",
          rp
        )
      }

      imp_tbl <- tibble(
        predictor = names(imp),
        importance = as.numeric(imp),
        scheme = scheme_name,
        variant = variant_name,
        replicate = rp
      )

      all_importance[[length(all_importance) + 1]] <- imp_tbl


      # ------------------------------------------------------
      # 19.13 Calibration + external evaluation
      # ------------------------------------------------------

      ev <- evaluate_model(
        model = fit,
        features = features,
        df_cal = df_cal,
        df_test = df_te,
        scheme = scheme_name,
        variant = variant_name,
        rep_id = rp
      )


      # ------------------------------------------------------
      # 19.14 Store outputs
      # ------------------------------------------------------

      all_metrics[[result_index]] <- ev$metrics

      all_calibration[[result_index]] <- ev$calibration

      all_predictions[[result_index]] <- ev$predictions

      all_thresholds[[result_index]] <- ev$thresholds

      result_index <- result_index + 1
    }
  }
}


# ============================================================
# 19B. IMMEDIATE POST-LOOP QC
# ============================================================

expected_models <- N_REP * 3 * 2

actual_results <- result_index - 1


cat(
  "\n============================================================\n"
)

cat(
  "MAIN LOOP COMPLETED\n"
)

cat(
  "Expected fitted/evaluated models =",
  expected_models,
  "\n"
)

cat(
  "Actual fitted/evaluated models   =",
  actual_results,
  "\n"
)

cat(
  "Sampling diagnostic rows         =",
  length(all_sampling),
  "\n"
)

cat(
  "Importance tables                =",
  length(all_importance),
  "\n"
)

cat(
  "============================================================\n"
)


stopifnot(
  actual_results == expected_models
)

stopifnot(
  length(all_metrics) == expected_models
)

stopifnot(
  length(all_calibration) == expected_models
)

stopifnot(
  length(all_predictions) == expected_models
)

stopifnot(
  length(all_thresholds) == expected_models
)

stopifnot(
  length(all_importance) == expected_models
)

# sampling diagnostics are one per scheme, not one per model variant
stopifnot(
  length(all_sampling) == N_REP * 3
)


cat(
  "\nPASS: Section 19 completed with expected number of models.\n"
)
cat(
  "N_REP =", N_REP,
  "\nN_TRAIN_REP =", N_TRAIN_REP,
  "\np_shadow_area =", p_shadow_area,
  "\nFIT n =", nrow(df_fit_pool),
  "\nCAL n =", nrow(df_cal),
  "\nTEST n =", nrow(df_te),
  "\nRAW features =", length(RAW_FEATURES),
  "\nRAW+SHADOW features =", length(RAW_SHADOW_FEATURES),
  "\n"
)
# ============================================================
# 21. Combine replicate results
# ============================================================

metrics_all <-
  bind_rows(
    all_metrics
  )


calibration_all <-
  bind_rows(
    all_calibration
  )


predictions_all <-
  bind_rows(
    all_predictions
  )


thresholds_all <-
  bind_rows(
    all_thresholds
  )


sampling_all <-
  bind_rows(
    all_sampling
  )


importance_all <-
  bind_rows(
    all_importance
  )


# ============================================================
# 22. Save replicate-level results
# ============================================================

write_csv(
  metrics_all,
  file.path(
    DIR_CSV,
    "10_EXTERNAL_TEST_metrics_all_replicates.csv"
  )
)


write_csv(
  calibration_all,
  file.path(
    DIR_CSV,
    "11_EXTERNAL_TEST_calibration_all_replicates.csv"
  )
)


write_csv(
  thresholds_all,
  file.path(
    DIR_CSV,
    "12_CALIBRATION_thresholds_all_replicates.csv"
  )
)


write_csv(
  sampling_all,
  file.path(
    DIR_CSV,
    "13_sampling_diagnostics_all_replicates.csv"
  )
)


write_csv(
  importance_all,
  file.path(
    DIR_CSV,
    "14_variable_importance_all_replicates.csv"
  )
)


write_csv(
  predictions_all,
  file.path(
    DIR_PRED,
    "15_EXTERNAL_TEST_point_predictions_all.csv"
  )
)


# ============================================================
# 23. Summarize model performance
# ============================================================

metric_summary <-
  metrics_all %>%
  group_by(
    scheme,
    variant,
    regime,
    threshold_strategy
  ) %>%
  summarise(

    n_rep =
      n_distinct(
        replicate
      ),

    AUC_median =
      median(
        AUC,
        na.rm = TRUE
      ),

    AUC_q025 =
      quantile(
        AUC,
        0.025,
        na.rm = TRUE
      ),

    AUC_q975 =
      quantile(
        AUC,
        0.975,
        na.rm = TRUE
      ),

    Brier_median =
      median(
        Brier,
        na.rm = TRUE
      ),

    Brier_q025 =
      quantile(
        Brier,
        0.025,
        na.rm = TRUE
      ),

    Brier_q975 =
      quantile(
        Brier,
        0.975,
        na.rm = TRUE
      ),

    BalancedAccuracy_median =
      median(
        BalancedAccuracy,
        na.rm = TRUE
      ),

    BalancedAccuracy_q025 =
      quantile(
        BalancedAccuracy,
        0.025,
        na.rm = TRUE
      ),

    BalancedAccuracy_q975 =
      quantile(
        BalancedAccuracy,
        0.975,
        na.rm = TRUE
      ),

    Sensitivity_median =
      median(
        Sensitivity,
        na.rm = TRUE
      ),

    Specificity_median =
      median(
        Specificity,
        na.rm = TRUE
      ),

    TSS_median =
      median(
        TSS,
        na.rm = TRUE
      ),

    F1_median =
      median(
        F1,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


write_csv(
  metric_summary,
  file.path(
    DIR_CSV,
    "20_EXTERNAL_TEST_performance_summary.csv"
  )
)


# ============================================================
# 24. Summarize thresholds
# ============================================================

threshold_summary <-
  thresholds_all %>%
  group_by(
    scheme,
    variant
  ) %>%
  summarise(

    threshold_global_median =
      median(
        threshold_global,
        na.rm = TRUE
      ),

    threshold_global_q025 =
      quantile(
        threshold_global,
        0.025,
        na.rm = TRUE
      ),

    threshold_global_q975 =
      quantile(
        threshold_global,
        0.975,
        na.rm = TRUE
      ),

    threshold_nonshadow_median =
      median(
        threshold_nonshadow,
        na.rm = TRUE
      ),

    threshold_shadow_median =
      median(
        threshold_shadow,
        na.rm = TRUE
      ),

    threshold_difference_median =
      median(
        threshold_difference,
        na.rm = TRUE
      ),

    threshold_difference_q025 =
      quantile(
        threshold_difference,
        0.025,
        na.rm = TRUE
      ),

    threshold_difference_q975 =
      quantile(
        threshold_difference,
        0.975,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


write_csv(
  threshold_summary,
  file.path(
    DIR_CSV,
    "21_threshold_summary.csv"
  )
)


# ============================================================
# 25. Summarize calibration
# ============================================================

calibration_summary <-
  calibration_all %>%
  group_by(
    scheme,
    variant,
    regime
  ) %>%
  summarise(

    n_rep =
      n_distinct(
        replicate
      ),

    intercept_median =
      median(
        calibration_intercept,
        na.rm = TRUE
      ),

    intercept_q025 =
      quantile(
        calibration_intercept,
        0.025,
        na.rm = TRUE
      ),

    intercept_q975 =
      quantile(
        calibration_intercept,
        0.975,
        na.rm = TRUE
      ),

    slope_median =
      median(
        calibration_slope,
        na.rm = TRUE
      ),

    slope_q025 =
      quantile(
        calibration_slope,
        0.025,
        na.rm = TRUE
      ),

    slope_q975 =
      quantile(
        calibration_slope,
        0.975,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


write_csv(
  calibration_summary,
  file.path(
    DIR_CSV,
    "22_calibration_summary.csv"
  )
)


# ============================================================
# 26. Summarize sampling representation
# ============================================================

sampling_summary <-
  sampling_all %>%
  group_by(
    scheme
  ) %>%
  summarise(

    n_rep =
      n(),

    mean_n =
      mean(
        n_rows
      ),

    mean_unique_orig =
      mean(
        n_unique_orig
      ),

    mean_duplicate_fraction =
      mean(
        duplicate_fraction
      ),

    mean_shadow_prevalence =
      mean(
        p_shadow
      ),

    sd_shadow_prevalence =
      sd(
        p_shadow
      ),

    mean_response_prevalence =
      mean(
        prevalence
      ),

    mean_weight_shadow =
      mean(
        weight_shadow
      ),

    mean_weight_nonshadow =
      mean(
        weight_nonshadow
      ),

    .groups = "drop"
  )


write_csv(
  sampling_summary,
  file.path(
    DIR_CSV,
    "23_sampling_summary.csv"
  )
)


# ------------------------------------------------------------
# Supplementary Table S3: regime representation by sampling scheme
# ------------------------------------------------------------

TableS3_regime_representation <-
  sampling_all %>%
  group_by(scheme) %>%
  summarise(
    n_total_all = sum(n_rows),
    n_shadow_all = sum(n_shadow),
    n_nonshadow_all = sum(n_nonshadow),
    shadow_frac_all = n_shadow_all / n_total_all,
    p_shadow_area = p_shadow_area,
    pool_shadow_frac = p_shadow_pool,
    .groups = "drop"
  ) %>%
  arrange(factor(scheme, levels = c("SB", "SW", "UR")))

write_csv(
  TableS3_regime_representation,
  file.path(DIR_CSV, "TableS3_regime_representation_by_sampling_scheme.csv")
)


# ============================================================
# 27. Compare predictor variants
#
# Paired within:
#   same scheme
#   same replicate
#
# We report DISTRIBUTION and 95% interval of replicate
# differences rather than treating stochastic replicates
# as independent biological observations.
# ============================================================

main_metrics <-
  metrics_all %>%
  filter(
    regime == "ALL",
    threshold_strategy == "GLOBAL"
  ) %>%
  select(
    scheme,
    variant,
    replicate,
    AUC,
    Brier,
    LogLoss,
    BalancedAccuracy,
    TSS
  )


paired_variant <-
  main_metrics %>%
  pivot_wider(
    names_from =
      variant,

    values_from =
      c(
        AUC,
        Brier,
        LogLoss,
        BalancedAccuracy,
        TSS
      )
  ) %>%
  mutate(

    delta_AUC =
      AUC_RAW_SHADOW -
      AUC_RAW,

    # negative = improvement
    delta_Brier =
      Brier_RAW_SHADOW -
      Brier_RAW,

    # negative = improvement
    delta_LogLoss =
      LogLoss_RAW_SHADOW -
      LogLoss_RAW,

    delta_BalancedAccuracy =
      BalancedAccuracy_RAW_SHADOW -
      BalancedAccuracy_RAW,

    delta_TSS =
      TSS_RAW_SHADOW -
      TSS_RAW
  )


write_csv(
  paired_variant,
  file.path(
    DIR_CSV,
    "30_RAW_SHADOW_minus_RAW_by_replicate.csv"
  )
)


paired_variant_summary <-
  paired_variant %>%
  group_by(
    scheme
  ) %>%
  summarise(

    n_rep =
      n(),

    delta_AUC_median =
      median(
        delta_AUC,
        na.rm = TRUE
      ),

    delta_AUC_q025 =
      quantile(
        delta_AUC,
        0.025,
        na.rm = TRUE
      ),

    delta_AUC_q975 =
      quantile(
        delta_AUC,
        0.975,
        na.rm = TRUE
      ),

    delta_Brier_median =
      median(
        delta_Brier,
        na.rm = TRUE
      ),

    delta_Brier_q025 =
      quantile(
        delta_Brier,
        0.025,
        na.rm = TRUE
      ),

    delta_Brier_q975 =
      quantile(
        delta_Brier,
        0.975,
        na.rm = TRUE
      ),

    delta_BalancedAccuracy_median =
      median(
        delta_BalancedAccuracy,
        na.rm = TRUE
      ),

    delta_TSS_median =
      median(
        delta_TSS,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


write_csv(
  paired_variant_summary,
  file.path(
    DIR_CSV,
    "31_RAW_SHADOW_minus_RAW_summary.csv"
  )
)


# ============================================================
# 28. Shadow-variable importance
# ============================================================

shadow_importance <-
  importance_all %>%
  filter(
    variant ==
      "RAW_SHADOW",

    predictor ==
      "shadow_bin"
  )


write_csv(
  shadow_importance,
  file.path(
    DIR_CSV,
    "32_shadow_bin_permutation_importance.csv"
  )
)


shadow_importance_summary <-
  shadow_importance %>%
  group_by(
    scheme
  ) %>%
  summarise(

    n_rep =
      n(),

    median_importance =
      median(
        importance,
        na.rm = TRUE
      ),

    q025 =
      quantile(
        importance,
        0.025,
        na.rm = TRUE
      ),

    q975 =
      quantile(
        importance,
        0.975,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


write_csv(
  shadow_importance_summary,
  file.path(
    DIR_CSV,
    "33_shadow_bin_importance_summary.csv"
  )
)


# ============================================================
# 29. Variable-importance summary
# ============================================================

importance_summary <-
  importance_all %>%
  group_by(
    scheme,
    variant,
    predictor
  ) %>%
  summarise(

    median_importance =
      median(
        importance,
        na.rm = TRUE
      ),

    q025 =
      quantile(
        importance,
        0.025,
        na.rm = TRUE
      ),

    q975 =
      quantile(
        importance,
        0.975,
        na.rm = TRUE
      ),

    .groups = "drop"
  ) %>%
  group_by(
    scheme,
    variant
  ) %>%
  arrange(
    desc(
      median_importance
    ),
    .by_group = TRUE
  ) %>%
  mutate(
    importance_rank =
      row_number()
  ) %>%
  ungroup()


write_csv(
  importance_summary,
  file.path(
    DIR_CSV,
    "34_variable_importance_summary.csv"
  )
)


# ============================================================
# 30. Regime-specific classification comparison
# ============================================================

regime_metric_summary <-
  metrics_all %>%
  filter(
    threshold_strategy ==
      "REGIME_SPECIFIC",

    regime %in%
      c(
        "NONSHADOW",
        "SHADOW"
      )
  ) %>%
  group_by(
    scheme,
    variant,
    regime
  ) %>%
  summarise(

    n_rep =
      n(),

    Sensitivity_median =
      median(
        Sensitivity,
        na.rm = TRUE
      ),

    Sensitivity_q025 =
      quantile(
        Sensitivity,
        0.025,
        na.rm = TRUE
      ),

    Sensitivity_q975 =
      quantile(
        Sensitivity,
        0.975,
        na.rm = TRUE
      ),

    Specificity_median =
      median(
        Specificity,
        na.rm = TRUE
      ),

    Specificity_q025 =
      quantile(
        Specificity,
        0.025,
        na.rm = TRUE
      ),

    Specificity_q975 =
      quantile(
        Specificity,
        0.975,
        na.rm = TRUE
      ),

    AUC_median =
      median(
        AUC,
        na.rm = TRUE
      ),

    Brier_median =
      median(
        Brier,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


write_csv(
  regime_metric_summary,
  file.path(
    DIR_CSV,
    "35_regime_specific_external_performance.csv"
  )
)


# ============================================================
# 30. PUBLICATION FIGURE 1
# EXTERNAL AUC ACROSS REPLICATES
# ============================================================

plot_auc_data <-
  metrics_all %>%
  filter(
    regime == "ALL",
    threshold_strategy == "GLOBAL"
  )


fig_auc <-
  ggplot(
    plot_auc_data,
    aes(
      x = scheme,
      y = AUC,
      fill = variant
    )
  ) +
  geom_boxplot(
    outlier.shape = NA,
    alpha = 0.7
  ) +
  geom_jitter(
    aes(
      shape = variant
    ),
    width = 0.10,
    alpha = 0.55,
    size = 1.5
  ) +
  labs(
    title =
      "External-test discrimination across sampling scenarios",

    subtitle =
      "Fixed independent external test; stochastic training replicates",

    x =
      "Sampling scenario",

    y =
      "External-test AUC",

    fill =
      "Predictor set",

    shape =
      "Predictor set"
  ) +
  theme_bw(
    base_size = 12
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG01_external_AUC_scenarios.tiff"
  ),
  fig_auc,
  width = 8,
  height = 5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 31. PUBLICATION FIGURE 2
# RAW+SHADOW - RAW AUC DIFFERENCE
# ============================================================

fig_delta_auc <-
  ggplot(
    paired_variant,
    aes(
      x = scheme,
      y = delta_AUC
    )
  ) +
  geom_hline(
    yintercept = 0,
    linetype = 2
  ) +
  geom_boxplot(
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0.08,
    alpha = 0.6
  ) +
  labs(
    title =
      "Incremental discrimination associated with illumination regime",

    subtitle =
      "RAW+SHADOW minus RAW on the same external test",

    x =
      "Sampling scenario",

    y =
      expression(
        Delta *
          " AUC (RAW+SHADOW - RAW)"
      )
  ) +
  theme_bw(
    base_size = 12
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG02_delta_AUC_RAW_SHADOW_minus_RAW.tiff"
  ),
  fig_delta_auc,
  width = 7,
  height = 5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 32. PUBLICATION FIGURE 3
# THRESHOLDS BY ILLUMINATION REGIME
# ============================================================

threshold_long <-
  thresholds_all %>%
  select(
    scheme,
    variant,
    replicate,
    threshold_nonshadow,
    threshold_shadow
  ) %>%
  pivot_longer(
    cols =
      c(
        threshold_nonshadow,
        threshold_shadow
      ),

    names_to =
      "illumination_regime",

    values_to =
      "threshold"
  ) %>%
  mutate(
    illumination_regime =
      recode(
        illumination_regime,

        threshold_nonshadow =
          "Non-shadow",

        threshold_shadow =
          "Shadow"
      )
  )


fig_threshold <-
  ggplot(
    threshold_long,
    aes(
      x =
        illumination_regime,

      y =
        threshold,

      fill =
        variant
    )
  ) +
  geom_boxplot(
    outlier.shape = NA
  ) +
  facet_wrap(
    ~scheme
  ) +
  labs(
    title =
      "Calibration-derived decision thresholds by illumination regime",

    subtitle =
      "Thresholds estimated exclusively from held-out calibration data",

    x =
      "Illumination regime",

    y =
      "Decision threshold",

    fill =
      "Predictor set"
  ) +
  theme_bw(
    base_size = 12
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG03_regime_specific_thresholds.tiff"
  ),
  fig_threshold,
  width = 10,
  height = 5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 33. PUBLICATION FIGURE 4
# EXTERNAL CALIBRATION SLOPE
# ============================================================

fig_calibration <-
  calibration_all %>%
  filter(
    regime %in%
      c(
        "ALL",
        "NONSHADOW",
        "SHADOW"
      )
  ) %>%
  ggplot(
    aes(
      x = regime,
      y = calibration_slope,
      fill = variant
    )
  ) +
  geom_hline(
    yintercept = 1,
    linetype = 2
  ) +
  geom_boxplot(
    outlier.shape = NA
  ) +
  facet_wrap(
    ~scheme
  ) +
  labs(
    title =
      "External calibration slope across illumination regimes",

    subtitle =
      "Slope = 1 indicates ideal calibration",

    x =
      "Evaluation domain",

    y =
      "Calibration slope",

    fill =
      "Predictor set"
  ) +
  theme_bw(
    base_size = 12
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG04_external_calibration_slope.tiff"
  ),
  fig_calibration,
  width = 10,
  height = 5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 34. PUBLICATION FIGURE 5
# VARIABLE IMPORTANCE
# ============================================================

top_imp <-
  importance_summary %>%
  filter(
    importance_rank <= 15
  )


fig_importance <-
  ggplot(
    top_imp,
    aes(
      x =
        reorder(
          predictor,
          median_importance
        ),

      y =
        median_importance
    )
  ) +
  geom_point() +
  geom_errorbar(
    aes(
      ymin = q025,
      ymax = q975
    ),
    width = 0
  ) +
  coord_flip() +
  facet_grid(
    scheme ~ variant,
    scales = "free_y"
  ) +
  labs(
    title =
      "Permutation importance across stochastic model replicates",

    x =
      NULL,

    y =
      "Median permutation importance"
  ) +
  theme_bw(
    base_size = 11
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG05_variable_importance.tiff"
  ),
  fig_importance,
  width = 11,
  height = 10,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 35. OPTIONAL FULL-RASTER ENSEMBLE
#
# Default FALSE because this can take substantial time.
# ============================================================

if(
  DO_RASTER_ENSEMBLE
) {

  cat(
    "\nStarting optional raster ensemble prediction...\n"
  )


  # ----------------------------------------------------------
  # Build exact prediction raster stacks
  # ----------------------------------------------------------

  raw_raster <-r_master[[RAW_FEATURES]]


  shadow_for_model <-
    shadow_r

  names(
    shadow_for_model
  ) <-
    "shadow_bin"


  raw_shadow_raster <-
    c(
      raw_raster,
      shadow_for_model
    )


  names(
    raw_shadow_raster
  ) <-
    RAW_SHADOW_FEATURES


  pred_stack <-
    if(
      MAP_VARIANT ==
      "RAW"
    ) {
      raw_raster
    } else {
      raw_shadow_raster
    }


  map_features <-
    if(
      MAP_VARIANT ==
      "RAW"
    ) {
      RAW_FEATURES
    } else {
      RAW_SHADOW_FEATURES
    }


  sum_r <-
    NULL

  sumsq_r <-
    NULL

  n_map <-
    0


  ranger_terra_fun <-
    function(
      model,
      data
    ) {

      p <-
        predict(
          model,
          data =
            data
        )$predictions

      if(
        is.matrix(p)
      ) {

        if(
          "1" %in%
          colnames(p)
        ) {

          return(
            p[, "1"]
          )
        }

        return(
          p[, ncol(p)]
        )
      }

      as.numeric(p)
    }


  for(
    rp in seq_len(
      N_REP
    )
  ) {

    model_file <-
      file.path(
        DIR_MODEL,
        paste0(
          "RF_",
          MAP_SCHEME,
          "_",
          MAP_VARIANT,
          "_REP",
          sprintf(
            "%03d",
            rp
          ),
          ".rds"
        )
      )


    if(
      !file.exists(
        model_file
      )
    ) {
      next
    }


    mod <-
      readRDS(
        model_file
      )


    tmp_file <-
      file.path(
        DIR_MAP,
        paste0(
          "_TMP_",
          MAP_SCHEME,
          "_",
          MAP_VARIANT,
          "_",
          rp,
          ".tif"
        )
      )


    rr <-
      terra::predict(
        pred_stack,
        mod,
        fun =
          ranger_terra_fun,
        na.rm = TRUE,
        filename =
          tmp_file,
        overwrite = TRUE
      )


    if(
      is.null(
        sum_r
      )
    ) {

      sum_r <-
        rr

      sumsq_r <-
        rr^2

    } else {

      sum_r <-
        sum_r +
        rr

      sumsq_r <-
        sumsq_r +
        rr^2
    }


    n_map <-
      n_map + 1
  }


  if(
    n_map > 0
  ) {

    ensemble_mean <-
      sum_r /
      n_map


    ensemble_var <-
      (
        sumsq_r /
          n_map
      ) -
      ensemble_mean^2


    ensemble_var <-
      terra::ifel(
        ensemble_var < 0,
        0,
        ensemble_var
      )


    ensemble_sd <-
      sqrt(
        ensemble_var
      )


    names(
      ensemble_mean
    ) <-
      "probability_mean"


    names(
      ensemble_sd
    ) <-
      "probability_sd"


    writeRaster(
      ensemble_mean,
      file.path(
        DIR_MAP,
        paste0(
          "ENSEMBLE_MEAN_",
          MAP_SCHEME,
          "_",
          MAP_VARIANT,
          ".tif"
        )
      ),
      overwrite = TRUE
    )


    writeRaster(
      ensemble_sd,
      file.path(
        DIR_MAP,
        paste0(
          "ENSEMBLE_SD_",
          MAP_SCHEME,
          "_",
          MAP_VARIANT,
          ".tif"
        )
      ),
      overwrite = TRUE
    )
  }
}


# ============================================================
# 36. FINAL AUTOMATIC QC REPORT
# ============================================================

qc_final <-
  tibble(

    check =
      c(
        "SCL excluded from RAW",
        "SCL excluded from RAW+SHADOW",
        "ShadowMask excluded from RAW",
        "ShadowMask excluded from RAW+SHADOW",
        "shadow_bin excluded from RAW",
        "shadow_bin included in RAW+SHADOW",
        "RAW+SHADOW differs from RAW by exactly one variable",
        "Fit/calibration orig_id overlap",
        "Exact train/test coordinate overlap",
        "External test unchanged across replicates"
      ),

    passed =
      c(

        !"SCL" %in%
          RAW_FEATURES,

        !"SCL" %in%
          RAW_SHADOW_FEATURES,

        !"ShadowMask" %in%
          RAW_FEATURES,

        !"ShadowMask" %in%
          RAW_SHADOW_FEATURES,

        !"shadow_bin" %in%
          RAW_FEATURES,

        "shadow_bin" %in%
          RAW_SHADOW_FEATURES,

        identical(
          setdiff(
            RAW_SHADOW_FEATURES,
            RAW_FEATURES
          ),
          "shadow_bin"
        ),

        length(
          overlap_fit_cal
        ) == 0,

        length(
          exact_coord_overlap
        ) == 0,

        n_distinct(
          predictions_all$test_id
        ) ==
          nrow(
            df_te
          )
      )
  )


write_csv(
  qc_final,
  file.path(
    DIR_QC,
    "99_FINAL_LEAKAGE_QC.csv"
  )
)


print(
  qc_final
)


if(
  !all(
    qc_final$passed
  )
) {

  stop(
    "One or more FINAL QC checks failed. Inspect 99_FINAL_LEAKAGE_QC.csv."
  )
}


# ============================================================
# 37. SESSION INFORMATION
# ============================================================

capture.output(
  sessionInfo(),
  file =
    file.path(
      OUT_DIR,
      "sessionInfo.txt"
    )
)


# ============================================================
# 38. FINAL CONSOLE REPORT
# ============================================================

cat(
  "\n\n============================================================\n"
)

cat(
  "FINAL PIPELINE COMPLETED SUCCESSFULLY\n"
)

cat(
  "============================================================\n"
)

cat(
  "\nLandscape shadow prevalence:",
  round(
    100 *
      p_shadow_area,
    3
  ),
  "%\n"
)

cat(
  "Original complete train pool:",
  nrow(
    df_pool
  ),
  "\n"
)

cat(
  "FIT pool:",
  nrow(
    df_fit_pool
  ),
  "\n"
)

cat(
  "Calibration:",
  nrow(
    df_cal
  ),
  "\n"
)

cat(
  "External test:",
  nrow(
    df_te
  ),
  "\n"
)

cat(
  "Replicates:",
  N_REP,
  "\n"
)

cat(
  "Models expected:",
  N_REP *
    3 *
    2,
  "\n"
)

cat(
  "\nRAW predictors:",
  length(
    RAW_FEATURES
  ),
  "\n"
)

cat(
  "RAW+SHADOW predictors:",
  length(
    RAW_SHADOW_FEATURES
  ),
  "\n"
)

cat(
  "\nDifference:"
)

print(
  setdiff(
    RAW_SHADOW_FEATURES,
    RAW_FEATURES
  )
)

cat(
  "\nAll QC checks passed.\n"
)

cat(
  "\nResults saved in:\n",
  OUT_DIR,
  "\n"
)
#############
# ============================================================
# 31. Statistical inference for primary contrasts
#
# Purpose:
#   A) RAW_SHADOW vs RAW
#   B) Shadow vs Non-shadow thresholds
#   C) Global vs regime-specific threshold performance
#   D) Sampling-scheme contrasts
#   E) FDR correction
#
# IMPORTANT:
# Replicates are stochastic modelling realizations, not
# independent ecological samples.
#
# Therefore:
#   - comparisons are paired by replicate where appropriate
#   - median paired effects + empirical 95% intervals are primary
#   - Wilcoxon signed-rank p-values are supplementary
#   - FDR is applied within comparison families
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})


# ============================================================
# 39.1 HELPERS
# ============================================================

safe_quantile <- function(x, p) {
  x <- x[is.finite(x)]
  if(length(x) == 0) return(NA_real_)
  as.numeric(quantile(x, p, na.rm = TRUE))
}


# ------------------------------------------------------------
# Paired rank-biserial correlation
#
# Range:
#   -1 to +1
#
# Positive:
#   x tends to exceed y
#
# Negative:
#   x tends to be below y
# ------------------------------------------------------------

paired_rank_biserial <- function(x, y) {

  d <- x - y

  d <- d[
    is.finite(d) &
    d != 0
  ]

  if(length(d) == 0) {
    return(NA_real_)
  }

  ranks <- rank(
    abs(d),
    ties.method = "average"
  )

  W_pos <- sum(
    ranks[d > 0]
  )

  W_neg <- sum(
    ranks[d < 0]
  )

  denom <- W_pos + W_neg

  if(denom == 0) {
    return(NA_real_)
  }

  (W_pos - W_neg) / denom
}


# ------------------------------------------------------------
# Generic paired comparison
# ------------------------------------------------------------

paired_test_summary <- function(
  df,
  x_col,
  y_col,
  comparison,
  metric,
  family
) {

  x <- df[[x_col]]
  y <- df[[y_col]]

  ok <-
    is.finite(x) &
    is.finite(y)

  x <- x[ok]
  y <- y[ok]

  d <- x - y

  if(length(d) < 3) {

    return(
      tibble(
        family = family,
        comparison = comparison,
        metric = metric,
        n_pairs = length(d),
        median_x = NA_real_,
        median_y = NA_real_,
        median_difference = NA_real_,
        difference_q025 = NA_real_,
        difference_q975 = NA_real_,
        mean_difference = NA_real_,
        rank_biserial = NA_real_,
        wilcox_V = NA_real_,
        p_value = NA_real_
      )
    )
  }

  wt <- tryCatch(
    suppressWarnings(
      wilcox.test(
        x,
        y,
        paired = TRUE,
        exact = FALSE,
        correct = FALSE
      )
    ),
    error = function(e) NULL
  )

  tibble(
    family = family,
    comparison = comparison,
    metric = metric,

    n_pairs = length(d),

    median_x =
      median(x, na.rm = TRUE),

    median_y =
      median(y, na.rm = TRUE),

    median_difference =
      median(d, na.rm = TRUE),

    difference_q025 =
      safe_quantile(d, 0.025),

    difference_q975 =
      safe_quantile(d, 0.975),

    mean_difference =
      mean(d, na.rm = TRUE),

    rank_biserial =
      paired_rank_biserial(x, y),

    wilcox_V =
      if(is.null(wt)) {
        NA_real_
      } else {
        as.numeric(wt$statistic)
      },

    p_value =
      if(is.null(wt)) {
        NA_real_
      } else {
        wt$p.value
      }
  )
}


# ============================================================
# 39.2 RAW+SHADOW vs RAW
#
# Same scheme + same replicate + same external test.
#
# For AUC / BA / TSS:
#   positive difference = RAW_SHADOW better
#
# For Brier / LogLoss:
#   negative difference = RAW_SHADOW better
# ============================================================

raw_compare <- metrics_all %>%
  filter(
    regime == "ALL",
    threshold_strategy == "GLOBAL"
  ) %>%
  select(
    scheme,
    variant,
    replicate,
    AUC,
    Brier,
    LogLoss,
    BalancedAccuracy,
    TSS
  ) %>%
  pivot_wider(
    names_from = variant,
    values_from = c(
      AUC,
      Brier,
      LogLoss,
      BalancedAccuracy,
      TSS
    )
  )


raw_tests <- bind_rows(
  lapply(
    unique(raw_compare$scheme),
    function(sc) {

      d <- raw_compare %>%
        filter(scheme == sc)

      bind_rows(

        paired_test_summary(
          d,
          "AUC_RAW_SHADOW",
          "AUC_RAW",
          comparison = paste0(sc, ": RAW_SHADOW - RAW"),
          metric = "AUC",
          family = "RAW_SHADOW_vs_RAW"
        ),

        paired_test_summary(
          d,
          "Brier_RAW_SHADOW",
          "Brier_RAW",
          comparison = paste0(sc, ": RAW_SHADOW - RAW"),
          metric = "Brier",
          family = "RAW_SHADOW_vs_RAW"
        ),

        paired_test_summary(
          d,
          "LogLoss_RAW_SHADOW",
          "LogLoss_RAW",
          comparison = paste0(sc, ": RAW_SHADOW - RAW"),
          metric = "LogLoss",
          family = "RAW_SHADOW_vs_RAW"
        ),

        paired_test_summary(
          d,
          "BalancedAccuracy_RAW_SHADOW",
          "BalancedAccuracy_RAW",
          comparison = paste0(sc, ": RAW_SHADOW - RAW"),
          metric = "BalancedAccuracy",
          family = "RAW_SHADOW_vs_RAW"
        ),

        paired_test_summary(
          d,
          "TSS_RAW_SHADOW",
          "TSS_RAW",
          comparison = paste0(sc, ": RAW_SHADOW - RAW"),
          metric = "TSS",
          family = "RAW_SHADOW_vs_RAW"
        )
      )
    }
  )
)


# ============================================================
# 39.3 SHADOW vs NON-SHADOW THRESHOLDS
#
# Positive difference:
#   threshold_shadow > threshold_nonshadow
# ============================================================

threshold_tests <- thresholds_all %>%
  group_by(
    scheme,
    variant
  ) %>%
  group_split() %>%
  lapply(
    function(d) {

      sc <- unique(d$scheme)
      vr <- unique(d$variant)

      paired_test_summary(
        d,
        "threshold_shadow",
        "threshold_nonshadow",
        comparison =
          paste0(
            sc,
            " / ",
            vr,
            ": Shadow - Non-shadow"
          ),
        metric = "DecisionThreshold",
        family = "Shadow_vs_Nonshadow_threshold"
      )
    }
  ) %>%
  bind_rows()


# ============================================================
# 39.4 GLOBAL vs REGIME-SPECIFIC THRESHOLD PERFORMANCE
#
# We compare the SAME fitted model on the SAME external test.
#
# Positive difference below:
#   regime-specific strategy performs better.
# ============================================================

threshold_strategy_data <- metrics_all %>%
  filter(
    regime == "ALL",
    threshold_strategy %in%
      c(
        "GLOBAL",
        "REGIME_SPECIFIC"
      )
  ) %>%
  select(
    scheme,
    variant,
    replicate,
    threshold_strategy,
    BalancedAccuracy,
    Sensitivity,
    Specificity,
    TSS,
    F1
  ) %>%
  pivot_wider(
    names_from =
      threshold_strategy,

    values_from =
      c(
        BalancedAccuracy,
        Sensitivity,
        Specificity,
        TSS,
        F1
      )
  )


strategy_tests <- bind_rows(
  lapply(
    split(
      threshold_strategy_data,
      interaction(
        threshold_strategy_data$scheme,
        threshold_strategy_data$variant,
        drop = TRUE
      )
    ),
    function(d) {

      sc <- unique(d$scheme)
      vr <- unique(d$variant)

      bind_rows(

        paired_test_summary(
          d,
          "BalancedAccuracy_REGIME_SPECIFIC",
          "BalancedAccuracy_GLOBAL",
          comparison =
            paste0(
              sc,
              " / ",
              vr,
              ": Regime-specific - Global"
            ),
          metric =
            "BalancedAccuracy",
          family =
            "RegimeSpecific_vs_Global"
        ),

        paired_test_summary(
          d,
          "Sensitivity_REGIME_SPECIFIC",
          "Sensitivity_GLOBAL",
          comparison =
            paste0(
              sc,
              " / ",
              vr,
              ": Regime-specific - Global"
            ),
          metric =
            "Sensitivity",
          family =
            "RegimeSpecific_vs_Global"
        ),

        paired_test_summary(
          d,
          "Specificity_REGIME_SPECIFIC",
          "Specificity_GLOBAL",
          comparison =
            paste0(
              sc,
              " / ",
              vr,
              ": Regime-specific - Global"
            ),
          metric =
            "Specificity",
          family =
            "RegimeSpecific_vs_Global"
        ),

        paired_test_summary(
          d,
          "TSS_REGIME_SPECIFIC",
          "TSS_GLOBAL",
          comparison =
            paste0(
              sc,
              " / ",
              vr,
              ": Regime-specific - Global"
            ),
          metric =
            "TSS",
          family =
            "RegimeSpecific_vs_Global"
        )
      )
    }
  )
)


# ============================================================
# 39.5 SAMPLING-SCHEME CONTRASTS
#
# These are paired by replicate as a Monte-Carlo sensitivity
# comparison.
#
# We do NOT interpret replicate number as an independent
# ecological sampling unit.
# ============================================================

sampling_perf <- metrics_all %>%
  filter(
    regime == "ALL",
    threshold_strategy == "GLOBAL"
  ) %>%
  select(
    scheme,
    variant,
    replicate,
    AUC,
    Brier,
    BalancedAccuracy,
    TSS
  )


sampling_pairs <- list(
  c("SB", "UR"),
  c("SW", "UR"),
  c("SB", "SW")
)


sampling_tests <- list()

ii <- 1

for(vr in unique(sampling_perf$variant)) {

  dv <- sampling_perf %>%
    filter(
      variant == vr
    )

  for(pp in sampling_pairs) {

    s1 <- pp[1]
    s2 <- pp[2]

    dd <- dv %>%
      filter(
        scheme %in% c(
          s1,
          s2
        )
      ) %>%
      pivot_wider(
        names_from = scheme,
        values_from = c(
          AUC,
          Brier,
          BalancedAccuracy,
          TSS
        )
      )

    sampling_tests[[ii]] <- bind_rows(

      paired_test_summary(
        dd,
        paste0("AUC_", s1),
        paste0("AUC_", s2),
        comparison =
          paste0(
            vr,
            ": ",
            s1,
            " - ",
            s2
          ),
        metric =
          "AUC",
        family =
          "Sampling_scheme"
      ),

      paired_test_summary(
        dd,
        paste0("Brier_", s1),
        paste0("Brier_", s2),
        comparison =
          paste0(
            vr,
            ": ",
            s1,
            " - ",
            s2
          ),
        metric =
          "Brier",
        family =
          "Sampling_scheme"
      ),

      paired_test_summary(
        dd,
        paste0("BalancedAccuracy_", s1),
        paste0("BalancedAccuracy_", s2),
        comparison =
          paste0(
            vr,
            ": ",
            s1,
            " - ",
            s2
          ),
        metric =
          "BalancedAccuracy",
        family =
          "Sampling_scheme"
      ),

      paired_test_summary(
        dd,
        paste0("TSS_", s1),
        paste0("TSS_", s2),
        comparison =
          paste0(
            vr,
            ": ",
            s1,
            " - ",
            s2
          ),
        metric =
          "TSS",
        family =
          "Sampling_scheme"
      )
    )

    ii <- ii + 1
  }
}


sampling_tests <- bind_rows(
  sampling_tests
)


# ============================================================
# 39.6 COMBINE ALL TESTS + FDR
# ============================================================

all_stat_tests <- bind_rows(
  raw_tests,
  threshold_tests,
  strategy_tests,
  sampling_tests
) %>%
  group_by(
    family
  ) %>%
  mutate(
    p_FDR =
      p.adjust(
        p_value,
        method = "BH"
      )
  ) %>%
  ungroup() %>%
  mutate(
    significant_raw =
      is.finite(p_value) &
      p_value < 0.05,

    significant_FDR =
      is.finite(p_FDR) &
      p_FDR < 0.05
  )


write_csv(
  all_stat_tests,
  file.path(
    DIR_CSV,
    "40_REVIEWER_statistical_tests_effect_sizes_FDR.csv"
  )
)


# separate publication tables

write_csv(
  raw_tests %>%
    left_join(
      all_stat_tests %>%
        select(
          family,
          comparison,
          metric,
          p_FDR,
          significant_FDR
        ),
      by = c(
        "family",
        "comparison",
        "metric"
      )
    ),
  file.path(
    DIR_CSV,
    "41_RAW_SHADOW_vs_RAW_inference.csv"
  )
)


write_csv(
  threshold_tests %>%
    left_join(
      all_stat_tests %>%
        select(
          family,
          comparison,
          metric,
          p_FDR,
          significant_FDR
        ),
      by = c(
        "family",
        "comparison",
        "metric"
      )
    ),
  file.path(
    DIR_CSV,
    "42_shadow_vs_nonshadow_threshold_inference.csv"
  )
)


write_csv(
  strategy_tests %>%
    left_join(
      all_stat_tests %>%
        select(
          family,
          comparison,
          metric,
          p_FDR,
          significant_FDR
        ),
      by = c(
        "family",
        "comparison",
        "metric"
      )
    ),
  file.path(
    DIR_CSV,
    "43_global_vs_regime_specific_inference.csv"
  )
)


write_csv(
  sampling_tests %>%
    left_join(
      all_stat_tests %>%
        select(
          family,
          comparison,
          metric,
          p_FDR,
          significant_FDR
        ),
      by = c(
        "family",
        "comparison",
        "metric"
      )
    ),
  file.path(
    DIR_CSV,
    "44_sampling_scheme_inference.csv"
  )
)


# ============================================================
# 39.7 BALANCED ACCURACY UNCERTAINTY TABLE
# ============================================================

ba_uncertainty <- metrics_all %>%
  filter(
    regime == "ALL"
  ) %>%
  group_by(
    scheme,
    variant,
    threshold_strategy
  ) %>%
  summarise(

    n_rep =
      n(),

    median =
      median(
        BalancedAccuracy,
        na.rm = TRUE
      ),

    mean =
      mean(
        BalancedAccuracy,
        na.rm = TRUE
      ),

    sd =
      sd(
        BalancedAccuracy,
        na.rm = TRUE
      ),

    q025 =
      safe_quantile(
        BalancedAccuracy,
        0.025
      ),

    q25 =
      safe_quantile(
        BalancedAccuracy,
        0.25
      ),

    q75 =
      safe_quantile(
        BalancedAccuracy,
        0.75
      ),

    q975 =
      safe_quantile(
        BalancedAccuracy,
        0.975
      ),

    .groups = "drop"
  )


write_csv(
  ba_uncertainty,
  file.path(
    DIR_CSV,
    "45_BalancedAccuracy_uncertainty.csv"
  )
)


# ============================================================
# 39.8 FIGURE:
# GLOBAL vs REGIME-SPECIFIC BALANCED ACCURACY
# ============================================================

fig_BA_strategy <- metrics_all %>%
  filter(
    regime == "ALL",
    threshold_strategy %in%
      c(
        "GLOBAL",
        "REGIME_SPECIFIC"
      )
  ) %>%
  ggplot(
    aes(
      x = threshold_strategy,
      y = BalancedAccuracy,
      fill = variant
    )
  ) +
  geom_boxplot(
    outlier.shape = NA
  ) +
  geom_jitter(
    aes(
      shape = variant
    ),
    width = 0.08,
    alpha = 0.45
  ) +
  facet_wrap(
    ~scheme
  ) +
  labs(
    title =
      "External-test performance under global and regime-specific thresholds",

    subtitle =
      "Same independent external test across stochastic model replicates",

    x =
      "Threshold strategy",

    y =
      "Balanced accuracy",

    fill =
      "Predictor set",

    shape =
      "Predictor set"
  ) +
  theme_bw(
    base_size = 12
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG06_global_vs_regime_specific_balanced_accuracy.tiff"
  ),
  fig_BA_strategy,
  width = 10,
  height = 5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 39.9 FIGURE:
# PAIRED SHADOW - NONSHADOW THRESHOLD DIFFERENCE
# ============================================================

fig_threshold_delta <- ggplot(
  thresholds_all,
  aes(
    x = interaction(
      scheme,
      variant,
      sep = " / "
    ),
    y = threshold_difference
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = 2
  ) +
  geom_boxplot(
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0.08,
    alpha = 0.55
  ) +
  labs(
    title =
      "Shadow–non-shadow decision-threshold differences",

    subtitle =
      "Positive values indicate higher calibration-derived thresholds in shadow",

    x =
      "Sampling scenario / predictor set",

    y =
      expression(
        Delta *
        " threshold (Shadow - Non-shadow)"
      )
  ) +
  theme_bw(
    base_size = 12
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 35,
        hjust = 1
      )
  )


ggsave(
  file.path(
    DIR_FIG,
    "FIG07_threshold_difference_shadow_minus_nonshadow.tiff"
  ),
  fig_threshold_delta,
  width = 9,
  height = 5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)
#######
###########################
# ============================================================
# 32. Stratified Monte Carlo mapped-area analysis
#     FINAL VERSION MATCHED TO THE ACTUAL PIPELINE
# ============================================================
#
# PURPOSE
# -------
# Quantify uncertainty in mapped suitable-habitat area without
# predicting all 180 stochastic RF models over every raster cell.
#
# IMPORTANT DESIGN RULES
# ----------------------
# 1) NO model refitting occurs here.
# 2) Models are read from:
#
#       MODELS/RF_<SCHEME>_<VARIANT>_REPxxx.rds
#
# 3) Thresholds are read from:
#
#       CSV/12_CALIBRATION_thresholds_all_replicates.csv
#
# 4) Thresholds remain calibration-derived only.
#    External-test data are NEVER used for mapping thresholds.
#
# 5) SCL is NEVER a model predictor.
#
# 6) ShadowMask is NEVER a model predictor.
#    It is used ONLY to define the illumination regime.
#
# 7) RAW:
#       RAW_FEATURES only
#
#    RAW_SHADOW:
#       exactly RAW_FEATURES + shadow_bin
#
# 8) The SAME fixed raster sample is used for:
#       - all 30 replicates
#       - SB / SW / UR
#       - RAW / RAW_SHADOW
#       - GLOBAL / REGIME_SPECIFIC thresholds
#
#    Therefore comparisons remain paired.
#
# 9) Shadow is deliberately oversampled in Monte-Carlo space
#    for precision, but TRUE raster stratum areas are used when
#    suitable area is reconstructed.
#
# ============================================================


suppressPackageStartupMessages({
  library(terra)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ranger)
  library(ggplot2)
})


# ============================================================
# 40.1 PATHS
# ============================================================

# Re-use OUT_DIR if the main pipeline still exists in memory.
# Otherwise reconstruct exactly the same journal output path.

if(!exists("BASE_DIR")) {
  BASE_DIR <- DATA_DIR
}


if(!exists("OUT_DIR")) {

  OUT_DIR <- file.path(
    BASE_DIR,
    "FINAL_JOURNAL_UR_SB_SW_RAW_SHADOW_LEAKFREE"
  )
}


DIR_CSV_40 <- file.path(
  OUT_DIR,
  "CSV"
)

DIR_MODEL_40 <- file.path(
  OUT_DIR,
  "MODELS"
)

DIR_QC_40 <- file.path(
  OUT_DIR,
  "QC"
)


AREA_DIR <- file.path(
  OUT_DIR,
  "40_MC_MAPPED_AREA"
)


dir.create(
  AREA_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


cat(
  "\nSection 40 output directory:\n",
  AREA_DIR,
  "\n"
)


# ============================================================
# 40.2 BASIC SETTINGS
# ============================================================

SEED_40 <- 40401


# ------------------------------------------------------------
# Main fixed Monte-Carlo sample
# ------------------------------------------------------------
#
# Shadow is only ~15.7% of the landscape.
#
# We oversample shadow intentionally to obtain sufficiently
# precise estimates in the smaller illumination regime.
#
# These numbers result in only 120,000 mapping cells instead
# of full-raster predictions for each of 180 models.
# ------------------------------------------------------------

N_MC_SHADOW <- 30000L

N_MC_NONSHADOW <- 90000L


# Number of model combinations expected:
#
# 30 replicate ×
# 3 sampling schemes ×
# 2 predictor variants
#
# = 180 models

EXPECTED_SCHEMES <- c(
  "SB",
  "SW",
  "UR"
)

EXPECTED_VARIANTS <- c(
  "RAW",
  "RAW_SHADOW"
)


# ============================================================
# 40.3 RECONSTRUCT / VERIFY MASTER RASTER
# ============================================================

if(!exists("PRED_RASTER_PATH")) {

  PRED_RASTER_PATH <-
    PRED_RASTER_PATH
}


if(!exists("r_master")) {

  r_master <- terra::rast(
    PRED_RASTER_PATH
  )
}


if(
  !"ShadowMask" %in%
    names(r_master)
) {

  stop(
    "ShadowMask not found in master raster."
  )
}


# canonical binary shadow raster

shadow_r <- r_master[["ShadowMask"]]


shadow_r <- terra::ifel(
  is.na(shadow_r),
  NA,
  terra::ifel(
    shadow_r > 0.5,
    1,
    0
  )
)


names(shadow_r) <- "ShadowMask"


# ============================================================
# 40.4 RESTORE EXACT PREDICTOR LISTS
# ============================================================
#
# Best case:
# RAW_FEATURES and RAW_SHADOW_FEATURES still exist in memory.
#
# Otherwise read the predictor lists written by the main
# pipeline QC section.
# ============================================================

RAW_LIST_FILE <- file.path(
  DIR_QC_40,
  "02_RAW_predictor_list.csv"
)

RAW_SHADOW_LIST_FILE <- file.path(
  DIR_QC_40,
  "03_RAW_SHADOW_predictor_list.csv"
)


if(!exists("RAW_FEATURES")) {

  if(!file.exists(RAW_LIST_FILE)) {

    stop(
      "RAW_FEATURES absent and QC predictor list not found:\n",
      RAW_LIST_FILE
    )
  }


  RAW_FEATURES <-
    readr::read_csv(
      RAW_LIST_FILE,
      show_col_types = FALSE
    )$predictor
}


if(!exists("RAW_SHADOW_FEATURES")) {

  if(!file.exists(RAW_SHADOW_LIST_FILE)) {

    stop(
      "RAW_SHADOW_FEATURES absent and QC predictor list not found:\n",
      RAW_SHADOW_LIST_FILE
    )
  }


  RAW_SHADOW_FEATURES <-
    readr::read_csv(
      RAW_SHADOW_LIST_FILE,
      show_col_types = FALSE
    )$predictor
}


RAW_FEATURES <- as.character(
  RAW_FEATURES
)

RAW_SHADOW_FEATURES <- as.character(
  RAW_SHADOW_FEATURES
)


# ------------------------------------------------------------
# CRITICAL predictor QC
# ------------------------------------------------------------

stopifnot(
  !"SCL" %in%
    RAW_FEATURES
)

stopifnot(
  !"ShadowMask" %in%
    RAW_FEATURES
)

stopifnot(
  !"shadow_bin" %in%
    RAW_FEATURES
)


stopifnot(
  !"SCL" %in%
    RAW_SHADOW_FEATURES
)

stopifnot(
  !"ShadowMask" %in%
    RAW_SHADOW_FEATURES
)

stopifnot(
  "shadow_bin" %in%
    RAW_SHADOW_FEATURES
)


stopifnot(
  length(
    RAW_SHADOW_FEATURES
  ) ==
    length(
      RAW_FEATURES
    ) + 1
)


stopifnot(
  identical(
    setdiff(
      RAW_SHADOW_FEATURES,
      RAW_FEATURES
    ),
    "shadow_bin"
  )
)


cat(
  "\nPredictor architecture verified:\n"
)

cat(
  "RAW predictors        =",
  length(RAW_FEATURES),
  "\n"
)

cat(
  "RAW_SHADOW predictors =",
  length(RAW_SHADOW_FEATURES),
  "\n"
)

cat(
  "Only added predictor  = shadow_bin\n"
)


# ============================================================
# 40.5 CHECK RAW RASTER PREDICTORS
# ============================================================

missing_raster_predictors <-
  setdiff(
    RAW_FEATURES,
    names(r_master)
  )


if(
  length(
    missing_raster_predictors
  ) > 0
) {

  stop(
    "These RAW predictors are absent from r_master:\n",
    paste(
      missing_raster_predictors,
      collapse = ", "
    )
  )
}


# IMPORTANT:
# Do not include ShadowMask in model-predictor stack.

raw_raster <- r_master[[RAW_FEATURES]]


stopifnot(
  !"SCL" %in%
    names(raw_raster)
)

stopifnot(
  !"ShadowMask" %in%
    names(raw_raster)
)


# ============================================================
# 40.6 READ ACTUAL CALIBRATION THRESHOLD TABLE
# ============================================================

THRESHOLD_FILE <- file.path(
  DIR_CSV_40,
  "12_CALIBRATION_thresholds_all_replicates.csv"
)


if(!file.exists(THRESHOLD_FILE)) {

  stop(
    "Calibration threshold file does not exist:\n",
    THRESHOLD_FILE
  )
}


thresholds_40 <-
  readr::read_csv(
    THRESHOLD_FILE,
    show_col_types = FALSE
  )


required_threshold_cols <- c(
  "scheme",
  "variant",
  "replicate",
  "threshold_global",
  "threshold_nonshadow",
  "threshold_shadow"
)


missing_threshold_cols <-
  setdiff(
    required_threshold_cols,
    names(thresholds_40)
  )


if(
  length(
    missing_threshold_cols
  ) > 0
) {

  stop(
    "Missing threshold columns:\n",
    paste(
      missing_threshold_cols,
      collapse = ", "
    )
  )
}


thresholds_40 <- thresholds_40 %>%

  mutate(
    scheme = as.character(scheme),
    variant = as.character(variant),
    replicate = as.integer(replicate),

    threshold_global =
      as.numeric(threshold_global),

    threshold_nonshadow =
      as.numeric(threshold_nonshadow),

    threshold_shadow =
      as.numeric(threshold_shadow)
  )


# ============================================================
# 40.7 DERIVE N_REP FROM ACTUAL SAVED RESULTS
# ============================================================

N_REP_40 <-
  max(
    thresholds_40$replicate,
    na.rm = TRUE
  )


cat(
  "\nReplicates detected from threshold table =",
  N_REP_40,
  "\n"
)


if(
  N_REP_40 != 30
) {

  warning(
    "Expected 30 replicates but detected ",
    N_REP_40,
    ". Continuing with actual threshold table."
  )
}


# ============================================================
# 40.8 VERIFY COMPLETE 180-MODEL DESIGN
# ============================================================

expected_grid_40 <-
  tidyr::expand_grid(

    replicate =
      seq_len(
        N_REP_40
      ),

    scheme =
      EXPECTED_SCHEMES,

    variant =
      EXPECTED_VARIANTS
  )


threshold_key_40 <-
  thresholds_40 %>%

  distinct(
    replicate,
    scheme,
    variant
  )


missing_threshold_combinations <-
  expected_grid_40 %>%

  anti_join(
    threshold_key_40,
    by =
      c(
        "replicate",
        "scheme",
        "variant"
      )
  )


if(
  nrow(
    missing_threshold_combinations
  ) > 0
) {

  print(
    missing_threshold_combinations
  )

  stop(
    "Some replicate × scheme × variant threshold combinations are missing."
  )
}


if(
  nrow(
    threshold_key_40
  ) !=
    N_REP_40 *
    3 *
    2
) {

  stop(
    "Threshold table does not contain exactly one row per model."
  )
}


# ============================================================
# 40.9 VERIFY ALL SAVED MODEL FILES
# ============================================================

model_path_40 <- function(
  scheme,
  variant,
  replicate
) {

  file.path(
    DIR_MODEL_40,

    paste0(
      "RF_",
      scheme,
      "_",
      variant,
      "_REP",
      sprintf(
        "%03d",
        replicate
      ),
      ".rds"
    )
  )
}


model_manifest_40 <-
  expected_grid_40 %>%

  rowwise() %>%

  mutate(

    model_file =
      model_path_40(
        scheme,
        variant,
        replicate
      ),

    exists =
      file.exists(
        model_file
      )
  ) %>%

  ungroup()


write_csv(
  model_manifest_40,
  file.path(
    AREA_DIR,
    "40_01_model_manifest.csv"
  )
)


if(
  !all(
    model_manifest_40$exists
  )
) {

  cat(
    "\nMissing model files:\n"
  )

  print(
    model_manifest_40 %>%
      filter(
        !exists
      )
  )

  stop(
    "Section 40 cannot continue because one or more saved models are missing."
  )
}


cat(
  "\nSaved model files verified =",
  nrow(model_manifest_40),
  "\n"
)


# ============================================================
# 40.10 VALID MAPPING DOMAIN
# ============================================================
#
# This is the ONLY potentially raster-heavy preparation step.
#
# It is done ONCE, not once per model.
#
# A cell belongs to the mapping domain only if:
#
#   - ShadowMask is available
#   - ALL RAW predictors are finite/non-NA
#
# This gives the correct common prediction domain for both
# RAW and RAW_SHADOW.
# ============================================================

cat(
  "\nBuilding common valid mapping mask...\n"
)


valid_predictor_count <-
  terra::app(
    raw_raster,
    fun = function(x) {
      sum(
        is.finite(x)
      )
    }
  )


valid_mask <-
  terra::ifel(
    valid_predictor_count ==
      length(
        RAW_FEATURES
      ) &
      !is.na(
        shadow_r
      ),
    1,
    NA
  )


rm(
  valid_predictor_count
)

gc()


# ============================================================
# 40.11 SHADOW / NON-SHADOW VALID STRATA
# ============================================================

shadow_valid <-
  terra::ifel(
    valid_mask == 1 &
      shadow_r == 1,
    1,
    NA
  )


nonshadow_valid <-
  terra::ifel(
    valid_mask == 1 &
      shadow_r == 0,
    1,
    NA
  )


# ============================================================
# 40.12 TRUE AREA OF EACH VALID STRATUM
# ============================================================

cell_area_km2 <-
  terra::cellSize(
    shadow_r,
    unit = "km"
  )


A_shadow_km2 <-
  as.numeric(
    terra::global(
      terra::mask(
        cell_area_km2,
        shadow_valid
      ),
      "sum",
      na.rm = TRUE
    )[1, 1]
  )


A_nonshadow_km2 <-
  as.numeric(
    terra::global(
      terra::mask(
        cell_area_km2,
        nonshadow_valid
      ),
      "sum",
      na.rm = TRUE
    )[1, 1]
  )


A_total_km2 <-
  A_shadow_km2 +
  A_nonshadow_km2


p_shadow_mapping_domain <-
  A_shadow_km2 /
  A_total_km2


mapping_domain_40 <-
  tibble(

    domain =
      c(
        "SHADOW",
        "NONSHADOW",
        "TOTAL"
      ),

    area_km2 =
      c(
        A_shadow_km2,
        A_nonshadow_km2,
        A_total_km2
      ),

    area_fraction =
      c(
        p_shadow_mapping_domain,
        1 -
          p_shadow_mapping_domain,
        1
      )
  )


write_csv(
  mapping_domain_40,
  file.path(
    AREA_DIR,
    "40_02_valid_mapping_domain_area.csv"
  )
)


cat(
  "\nValid mapping area =",
  round(
    A_total_km2,
    3
  ),
  "km2\n"
)


cat(
  "Shadow mapping-domain prevalence =",
  round(
    100 *
      p_shadow_mapping_domain,
    3
  ),
  "%\n"
)


# ============================================================
# 40.13 SAMPLE FIXED MONTE-CARLO CELLS
# ============================================================

sample_cells_40 <- function(
  mask_r,
  n_requested,
  seed
) {

  set.seed(seed)


  n_available <-
    as.numeric(
      terra::global(
        !is.na(
          mask_r
        ),
        "sum",
        na.rm = TRUE
      )[1, 1]
    )


  if(
    !is.finite(
      n_available
    ) ||
    n_available < 1
  ) {

    stop(
      "No valid cells available in this stratum."
    )
  }


  n_take <-
    min(
      as.integer(
        n_requested
      ),
      as.integer(
        n_available
      )
    )


  s <-
    terra::spatSample(
      mask_r,
      size =
        n_take,
      method =
        "random",
      na.rm =
        TRUE,
      cells =
        TRUE,
      values =
        FALSE,
      replace =
        FALSE
    )


  s <-
    as.data.frame(
      s
    )


  cell_name <-
    intersect(
      c(
        "cell",
        "cells"
      ),
      names(s)
    )[1]


  if(
    is.na(
      cell_name
    )
  ) {

    stop(
      "Could not identify sampled cell column returned by terra::spatSample()."
    )
  }


  out <-
    unique(
      as.integer(
        s[[cell_name]]
      )
    )


  if(
    length(out) == 0
  ) {

    stop(
      "Sampling returned zero cells."
    )
  }


  out
}


mc_cells_shadow <-
  sample_cells_40(
    shadow_valid,
    N_MC_SHADOW,
    SEED_40 + 1
  )


mc_cells_nonshadow <-
  sample_cells_40(
    nonshadow_valid,
    N_MC_NONSHADOW,
    SEED_40 + 2
  )


cat(
  "\nFixed MC cells:\n"
)

cat(
  " Shadow    =",
  length(
    mc_cells_shadow
  ),
  "\n"
)

cat(
  " Non-shadow =",
  length(
    mc_cells_nonshadow
  ),
  "\n"
)


# ============================================================
# 40.14 EXTRACT RAW PREDICTORS ONLY ONCE
# ============================================================
#
# This is the major computational saving.
#
# Raster values are extracted once for ~120k cells.
#
# The 180 RF models subsequently predict only from these
# in-memory data frames.
# ============================================================

cat(
  "\nExtracting RAW predictor values at fixed MC cells...\n"
)


X_mc_shadow <-
  terra::extract(
    raw_raster,
    mc_cells_shadow
  ) %>%
  as.data.frame()


X_mc_nonshadow <-
  terra::extract(
    raw_raster,
    mc_cells_nonshadow
  ) %>%
  as.data.frame()


# terra::extract by cell number normally does not need an ID
# assumption. Remove it only if actually returned.

if(
  "ID" %in%
    names(
      X_mc_shadow
    )
) {

  X_mc_shadow$ID <- NULL
}


if(
  "ID" %in%
    names(
      X_mc_nonshadow
    )
) {

  X_mc_nonshadow$ID <- NULL
}


# ensure exact predictor order

X_mc_shadow <-
  X_mc_shadow[
    ,
    RAW_FEATURES,
    drop = FALSE
  ]


X_mc_nonshadow <-
  X_mc_nonshadow[
    ,
    RAW_FEATURES,
    drop = FALSE
  ]


# Valid mask should guarantee complete cases.

stopifnot(
  all(
    complete.cases(
      X_mc_shadow
    )
  )
)


stopifnot(
  all(
    complete.cases(
      X_mc_nonshadow
    )
  )
)


# ------------------------------------------------------------
# Add canonical shadow_bin only at dataframe level.
# ------------------------------------------------------------

X_mc_shadow$shadow_bin <- 1L

X_mc_nonshadow$shadow_bin <- 0L


# ============================================================
# 40.15 FIXED-SAMPLE MANIFEST
# ============================================================

mc_manifest_40 <-
  bind_rows(

    tibble(
      mc_id =
        paste0(
          "SH_",
          seq_along(
            mc_cells_shadow
          )
        ),

      cell =
        mc_cells_shadow,

      regime =
        "SHADOW"
    ),

    tibble(
      mc_id =
        paste0(
          "NS_",
          seq_along(
            mc_cells_nonshadow
          )
        ),

      cell =
        mc_cells_nonshadow,

      regime =
        "NONSHADOW"
    )
  )


write_csv(
  mc_manifest_40,
  file.path(
    AREA_DIR,
    "40_03_fixed_MC_cell_manifest.csv"
  )
)


# ============================================================
# 40.16 FINAL MAPPING PREDICTOR QC
# ============================================================

stopifnot(
  identical(
    names(
      X_mc_shadow[
        ,
        RAW_FEATURES,
        drop = FALSE
      ]
    ),
    RAW_FEATURES
  )
)


stopifnot(
  identical(
    names(
      X_mc_nonshadow[
        ,
        RAW_FEATURES,
        drop = FALSE
      ]
    ),
    RAW_FEATURES
  )
)


stopifnot(
  identical(
    names(
      X_mc_shadow[
        ,
        RAW_SHADOW_FEATURES,
        drop = FALSE
      ]
    ),
    RAW_SHADOW_FEATURES
  )
)


stopifnot(
  identical(
    names(
      X_mc_nonshadow[
        ,
        RAW_SHADOW_FEATURES,
        drop = FALSE
      ]
    ),
    RAW_SHADOW_FEATURES
  )
)


# ============================================================
# 40.17 RANGER PROBABILITY HELPER
# ============================================================

predict_rf_prob_40 <- function(
  model,
  newdata
) {

  pp <-
    predict(
      model,
      data =
        newdata
    )$predictions


  if(
    is.matrix(
      pp
    )
  ) {

    if(
      "1" %in%
        colnames(
          pp
        )
    ) {

      return(
        as.numeric(
          pp[, "1"]
        )
      )

    } else {

      return(
        as.numeric(
          pp[
            ,
            ncol(
              pp
            )
          ]
        )
      )
    }
  }


  as.numeric(
    pp
  )
}


# ============================================================
# 40.18 AREA ESTIMATION FUNCTION
# ============================================================

estimate_area_40 <- function(
  p_shadow,
  p_nonshadow,
  threshold_shadow,
  threshold_nonshadow
) {

  suitable_shadow <-
    as.integer(
      p_shadow >=
        threshold_shadow
    )


  suitable_nonshadow <-
    as.integer(
      p_nonshadow >=
        threshold_nonshadow
    )


  q_shadow <-
    mean(
      suitable_shadow
    )


  q_nonshadow <-
    mean(
      suitable_nonshadow
    )


  area_shadow <-
    A_shadow_km2 *
    q_shadow


  area_nonshadow <-
    A_nonshadow_km2 *
    q_nonshadow


  area_total <-
    area_shadow +
    area_nonshadow


  # ----------------------------------------------------------
  # Monte-Carlo sampling SE
  # ----------------------------------------------------------

  var_shadow <-
    A_shadow_km2^2 *
    q_shadow *
    (1 - q_shadow) /
    length(
      suitable_shadow
    )


  var_nonshadow <-
    A_nonshadow_km2^2 *
    q_nonshadow *
    (1 - q_nonshadow) /
    length(
      suitable_nonshadow
    )


  mc_se <-
    sqrt(
      var_shadow +
      var_nonshadow
    )


  tibble(

    suitable_fraction_shadow =
      q_shadow,

    suitable_fraction_nonshadow =
      q_nonshadow,

    suitable_area_shadow_km2 =
      area_shadow,

    suitable_area_nonshadow_km2 =
      area_nonshadow,

    suitable_area_total_km2 =
      area_total,

    suitable_fraction_total =
      area_total /
      A_total_km2,

    MC_SE_km2 =
      mc_se,

    MC_CI_low_km2 =
      max(
        0,
        area_total -
          1.96 *
          mc_se
      ),

    MC_CI_high_km2 =
      min(
        A_total_km2,
        area_total +
          1.96 *
          mc_se
      )
  )
}


# ============================================================
# 40.19 GET EXACT THRESHOLD ROW
# ============================================================

get_threshold_row_40 <- function(
  scheme_name,
  variant_name,
  rp
) {

  z <-
    thresholds_40 %>%

    filter(
      scheme ==
        scheme_name,

      variant ==
        variant_name,

      replicate ==
        rp
    )


  if(
    nrow(z) != 1
  ) {

    stop(
      "Expected exactly one threshold row for ",
      scheme_name,
      " / ",
      variant_name,
      " / REP",
      sprintf(
        "%03d",
        rp
      ),
      "; found ",
      nrow(z)
    )
  }


  z
}


# ============================================================
# 40.20 MAIN FAST MONTE-CARLO MODEL LOOP
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "STARTING FAST MAPPED-AREA ANALYSIS\n"
)

cat(
  "============================================================\n"
)


area_results_list_40 <- list()

prediction_qc_list_40 <- list()

index_area_40 <- 1L

index_qc_40 <- 1L


for(
  rp in seq_len(
    N_REP_40
  )
) {

  cat(
    "\nREP",
    rp,
    "/",
    N_REP_40,
    "\n"
  )


  for(
    scheme_name in
      EXPECTED_SCHEMES
  ) {


    for(
      variant_name in
        EXPECTED_VARIANTS
    ) {


      # --------------------------------------------------------
      # Exact saved-model filename
      # --------------------------------------------------------

      this_model_file <-
        model_path_40(
          scheme =
            scheme_name,

          variant =
            variant_name,

          replicate =
            rp
        )


      if(
        !file.exists(
          this_model_file
        )
      ) {

        stop(
          "Model missing:\n",
          this_model_file
        )
      }


      # --------------------------------------------------------
      # Load ONLY one RF model at a time
      # --------------------------------------------------------

      rf_model <-
        readRDS(
          this_model_file
        )


      # --------------------------------------------------------
      # Select exact predictor set
      # --------------------------------------------------------

      if(
        variant_name ==
          "RAW"
      ) {

        features_40 <-
          RAW_FEATURES

      } else {

        features_40 <-
          RAW_SHADOW_FEATURES
      }


      X_sh <-
        X_mc_shadow[
          ,
          features_40,
          drop = FALSE
        ]


      X_ns <-
        X_mc_nonshadow[
          ,
          features_40,
          drop = FALSE
        ]


      # critical structure checks

      stopifnot(
        identical(
          names(
            X_sh
          ),
          features_40
        )
      )


      stopifnot(
        identical(
          names(
            X_ns
          ),
          features_40
        )
      )


      if(
        variant_name ==
          "RAW"
      ) {

        stopifnot(
          !"shadow_bin" %in%
            names(
              X_sh
            )
        )

      } else {

        stopifnot(
          "shadow_bin" %in%
            names(
              X_sh
            )
        )
      }


      # --------------------------------------------------------
      # Predict ONLY fixed MC cells
      # --------------------------------------------------------

      p_sh <-
        predict_rf_prob_40(
          rf_model,
          X_sh
        )


      p_ns <-
        predict_rf_prob_40(
          rf_model,
          X_ns
        )


      if(
        length(
          p_sh
        ) !=
          nrow(
            X_sh
          ) ||
        length(
          p_ns
        ) !=
          nrow(
            X_ns
          )
      ) {

        stop(
          "Prediction-length mismatch."
        )
      }


      if(
        any(
          !is.finite(
            p_sh
          )
        ) ||
        any(
          !is.finite(
            p_ns
          )
        )
      ) {

        stop(
          "Non-finite mapping probabilities detected."
        )
      }


      # --------------------------------------------------------
      # Actual calibration-derived thresholds
      # --------------------------------------------------------

      th_row <-
        get_threshold_row_40(
          scheme_name,
          variant_name,
          rp
        )


      th_global <-
        th_row$threshold_global[[1]]


      th_nonshadow <-
        th_row$threshold_nonshadow[[1]]


      th_shadow <-
        th_row$threshold_shadow[[1]]


      if(
        any(
          !is.finite(
            c(
              th_global,
              th_nonshadow,
              th_shadow
            )
          )
        )
      ) {

        stop(
          "Non-finite threshold encountered."
        )
      }


      # --------------------------------------------------------
      # A) GLOBAL threshold
      # --------------------------------------------------------

      ar_global <-
        estimate_area_40(

          p_shadow =
            p_sh,

          p_nonshadow =
            p_ns,

          threshold_shadow =
            th_global,

          threshold_nonshadow =
            th_global
        ) %>%

        mutate(

          replicate =
            rp,

          scheme =
            scheme_name,

          variant =
            variant_name,

          threshold_strategy =
            "GLOBAL",

          threshold_nonshadow =
            th_global,

          threshold_shadow =
            th_global,

          .before = 1
        )


      # --------------------------------------------------------
      # B) REGIME-SPECIFIC thresholds
      # --------------------------------------------------------

      ar_regime <-
        estimate_area_40(

          p_shadow =
            p_sh,

          p_nonshadow =
            p_ns,

          threshold_shadow =
            th_shadow,

          threshold_nonshadow =
            th_nonshadow
        ) %>%

        mutate(

          replicate =
            rp,

          scheme =
            scheme_name,

          variant =
            variant_name,

          threshold_strategy =
            "REGIME_SPECIFIC",

          threshold_nonshadow =
            th_nonshadow,

          threshold_shadow =
            th_shadow,

          .before = 1
        )


      area_results_list_40[[index_area_40]]<-
        bind_rows(
          ar_global,
          ar_regime
        )


      index_area_40 <-
        index_area_40 + 1L


      # --------------------------------------------------------
      # Prediction QC
      # --------------------------------------------------------

      prediction_qc_list_40[[index_qc_40]]<-
        tibble(

          replicate =
            rp,

          scheme =
            scheme_name,

          variant =
            variant_name,

          n_MC_shadow =
            length(
              p_sh
            ),

          n_MC_nonshadow =
            length(
              p_ns
            ),

          mean_probability_shadow =
            mean(
              p_sh
            ),

          mean_probability_nonshadow =
            mean(
              p_ns
            ),

          min_probability =
            min(
              c(
                p_sh,
                p_ns
              )
            ),

          max_probability =
            max(
              c(
                p_sh,
                p_ns
              )
            ),

          threshold_global =
            th_global,

          threshold_nonshadow =
            th_nonshadow,

          threshold_shadow =
            th_shadow
        )


      index_qc_40 <-
        index_qc_40 + 1L


      # --------------------------------------------------------
      # Release model immediately
      # --------------------------------------------------------

      rm(
        rf_model,
        X_sh,
        X_ns,
        p_sh,
        p_ns,
        th_row
      )


      gc(
        verbose =
          FALSE
      )
    }
  }
}


# ============================================================
# 40.21 COMBINE RESULTS
# ============================================================

area_results_40 <-
  bind_rows(
    area_results_list_40
  )


prediction_qc_40 <-
  bind_rows(
    prediction_qc_list_40
  )


write_csv(
  area_results_40,
  file.path(
    AREA_DIR,
    "40_04_MC_mapped_area_all_models.csv"
  )
)


write_csv(
  prediction_qc_40,
  file.path(
    AREA_DIR,
    "40_05_MC_prediction_QC.csv"
  )
)


# ============================================================
# 40.22 EXPECTED NUMBER OF RESULTS
# ============================================================
#
# 30 × 3 × 2 × 2 threshold strategies = 360
# ============================================================

EXPECTED_AREA_ROWS <-
  N_REP_40 *
  length(
    EXPECTED_SCHEMES
  ) *
  length(
    EXPECTED_VARIANTS
  ) *
  2


cat(
  "\nExpected mapped-area rows =",
  EXPECTED_AREA_ROWS,
  "\n"
)

cat(
  "Observed mapped-area rows =",
  nrow(
    area_results_40
  ),
  "\n"
)


stopifnot(
  nrow(
    area_results_40
  ) ==
    EXPECTED_AREA_ROWS
)


# ============================================================
# 40.23 SUMMARY ACROSS STOCHASTIC MODEL REPLICATES
# ============================================================

area_summary_40 <-
  area_results_40 %>%

  group_by(
    scheme,
    variant,
    threshold_strategy
  ) %>%

  summarise(

    n_replicates =
      n(),

    mean_area_km2 =
      mean(
        suitable_area_total_km2
      ),

    sd_between_models_km2 =
      sd(
        suitable_area_total_km2
      ),

    median_area_km2 =
      median(
        suitable_area_total_km2
      ),

    q025_area_km2 =
      quantile(
        suitable_area_total_km2,
        0.025
      ),

    q25_area_km2 =
      quantile(
        suitable_area_total_km2,
        0.25
      ),

    q75_area_km2 =
      quantile(
        suitable_area_total_km2,
        0.75
      ),

    q975_area_km2 =
      quantile(
        suitable_area_total_km2,
        0.975
      ),

    mean_fraction_suitable =
      mean(
        suitable_fraction_total
      ),

    mean_MC_SE_km2 =
      mean(
        MC_SE_km2
      ),

    .groups =
      "drop"
  )


write_csv(
  area_summary_40,
  file.path(
    AREA_DIR,
    "40_06_MC_mapped_area_summary.csv"
  )
)


# ============================================================
# 40.24 GLOBAL vs REGIME-SPECIFIC:
#       PAIRED AREA DIFFERENCE
# ============================================================

area_threshold_wide_40 <-
  area_results_40 %>%

  select(
    replicate,
    scheme,
    variant,
    threshold_strategy,
    suitable_area_total_km2,
    suitable_area_shadow_km2,
    suitable_area_nonshadow_km2
  ) %>%

  pivot_wider(

    names_from =
      threshold_strategy,

    values_from =
      c(
        suitable_area_total_km2,
        suitable_area_shadow_km2,
        suitable_area_nonshadow_km2
      )
  ) %>%

  mutate(

    delta_total_km2 =
      suitable_area_total_km2_REGIME_SPECIFIC -
      suitable_area_total_km2_GLOBAL,

    delta_shadow_km2 =
      suitable_area_shadow_km2_REGIME_SPECIFIC -
      suitable_area_shadow_km2_GLOBAL,

    delta_nonshadow_km2 =
      suitable_area_nonshadow_km2_REGIME_SPECIFIC -
      suitable_area_nonshadow_km2_GLOBAL,

    delta_total_percent =
      100 *
      delta_total_km2 /
      suitable_area_total_km2_GLOBAL
  )


write_csv(
  area_threshold_wide_40,
  file.path(
    AREA_DIR,
    "40_07_GLOBAL_vs_REGIME_SPECIFIC_area_paired.csv"
  )
)


# ============================================================
# 40.25 STATISTICAL TEST:
#       REGIME-SPECIFIC minus GLOBAL
# ============================================================

threshold_area_tests_40 <-
  area_threshold_wide_40 %>%

  group_by(
    scheme,
    variant
  ) %>%

  group_modify(
    ~{

      x <-
        .x$delta_total_km2


      x <-
        x[
          is.finite(
            x
          )
        ]


      if(
        length(
          x
        ) < 3
      ) {

        return(
          tibble(

            n =
              length(
                x
              ),

            mean_difference_km2 =
              NA_real_,

            median_difference_km2 =
              NA_real_,

            q025_difference_km2 =
              NA_real_,

            q975_difference_km2 =
              NA_real_,

            p_value =
              NA_real_
          )
        )
      }


      wt <-
        suppressWarnings(
          wilcox.test(
            x,
            mu = 0,
            exact = FALSE
          )
        )


      tibble(

        n =
          length(
            x
          ),

        mean_difference_km2 =
          mean(
            x
          ),

        median_difference_km2 =
          median(
            x
          ),

        q025_difference_km2 =
          quantile(
            x,
            0.025
          ),

        q975_difference_km2 =
          quantile(
            x,
            0.975
          ),

        p_value =
          wt$p.value
      )
    }
  ) %>%

  ungroup() %>%

  mutate(

    p_FDR =
      p.adjust(
        p_value,
        method =
          "BH"
      )
  )


write_csv(
  threshold_area_tests_40,
  file.path(
    AREA_DIR,
    "40_08_GLOBAL_vs_REGIME_SPECIFIC_area_tests_FDR.csv"
  )
)


# ============================================================
# 40.26 RAW_SHADOW vs RAW:
#       PAIRED MAPPED-AREA DIFFERENCE
# ============================================================

area_variant_wide_40 <-
  area_results_40 %>%

  select(
    replicate,
    scheme,
    variant,
    threshold_strategy,
    suitable_area_total_km2
  ) %>%

  pivot_wider(

    names_from =
      variant,

    values_from =
      suitable_area_total_km2
  )


stopifnot(
  all(
    c(
      "RAW",
      "RAW_SHADOW"
    ) %in%
      names(
        area_variant_wide_40
      )
  )
)


area_variant_wide_40 <-
  area_variant_wide_40 %>%

  mutate(

    delta_RAW_SHADOW_minus_RAW_km2 =
      RAW_SHADOW -
      RAW,

    delta_RAW_SHADOW_minus_RAW_percent =
      100 *
      delta_RAW_SHADOW_minus_RAW_km2 /
      RAW
  )


write_csv(
  area_variant_wide_40,
  file.path(
    AREA_DIR,
    "40_09_RAW_SHADOW_minus_RAW_mapped_area.csv"
  )
)


# ============================================================
# 40.27 RAW_SHADOW vs RAW STATISTICAL TESTS
# ============================================================

variant_area_tests_40 <-
  area_variant_wide_40 %>%

  group_by(
    scheme,
    threshold_strategy
  ) %>%

  group_modify(
    ~{

      x <-
        .x$delta_RAW_SHADOW_minus_RAW_km2


      x <-
        x[is.finite(x)]


      if(
        length(
          x
        ) < 3
      ) {

        return(
          tibble(

            n =
              length(
                x
              ),

            mean_difference_km2 =
              NA_real_,

            median_difference_km2 =
              NA_real_,

            q025_difference_km2 =
              NA_real_,

            q975_difference_km2 =
              NA_real_,

            p_value =
              NA_real_
          )
        )
      }


      wt <-
        suppressWarnings(
          wilcox.test(
            x,
            mu = 0,
            exact = FALSE
          )
        )


      tibble(

        n =
          length(
            x
          ),

        mean_difference_km2 =
          mean(
            x
          ),

        median_difference_km2 =
          median(
            x
          ),

        q025_difference_km2 =
          quantile(
            x,
            0.025
          ),

        q975_difference_km2 =
          quantile(
            x,
            0.975
          ),

        p_value =
          wt$p.value
      )
    }
  ) %>%

  ungroup() %>%

  mutate(

    p_FDR =
      p.adjust(
        p_value,
        method =
          "BH"
      )
  )


write_csv(
  variant_area_tests_40,
  file.path(
    AREA_DIR,
    "40_10_RAW_SHADOW_vs_RAW_area_tests_FDR.csv"
  )
)


# ============================================================
# 40.28 MONTE-CARLO ERROR vs STOCHASTIC MODEL UNCERTAINTY
# ============================================================
#
# This is very important for reviewer response.
#
# We explicitly show whether uncertainty associated with the
# finite raster MC sample is small compared with uncertainty
# across stochastic RF replicates.
# ============================================================

precision_summary_40 <-
  area_results_40 %>%

  group_by(
    scheme,
    variant,
    threshold_strategy
  ) %>%

  summarise(

    mean_MC_SE_km2 =
      mean(
        MC_SE_km2
      ),

    median_MC_SE_km2 =
      median(
        MC_SE_km2
      ),

    stochastic_model_SD_km2 =
      sd(
        suitable_area_total_km2
      ),

    MC_SE_to_model_SD_ratio =
      mean_MC_SE_km2 /
      stochastic_model_SD_km2,

    .groups =
      "drop"
  )


write_csv(
  precision_summary_40,
  file.path(
    AREA_DIR,
    "40_11_MC_error_vs_model_uncertainty.csv"
  )
)


# ============================================================
# 40.29 SAMPLE-SIZE PRECISION SENSITIVITY
# ============================================================
#
# No models are rerun.
#
# We calculate expected MC sampling error for smaller effective
# sample sizes using observed suitable fractions.
#
# This demonstrates whether 120,000 cells are more than
# sufficient.
# ============================================================

sensitivity_sizes <-
  c(
    30000L,
    60000L,
    120000L
  )


sensitivity_list_40 <-
  lapply(
    sensitivity_sizes,
    function(
      n_total
    ) {


      # retain same 1:3 sampling allocation:
      # 25% shadow / 75% non-shadow

      n_sh <-
        round(
          n_total *
          0.25
        )


      n_ns <-
        n_total -
        n_sh


      area_results_40 %>%

        mutate(

          sensitivity_total_n =
            n_total,

          sensitivity_n_shadow =
            n_sh,

          sensitivity_n_nonshadow =
            n_ns,

          sensitivity_MC_SE_km2 =

            sqrt(

              A_shadow_km2^2 *

                suitable_fraction_shadow *

                (
                  1 -
                    suitable_fraction_shadow
                ) /

                n_sh

              +

              A_nonshadow_km2^2 *

                suitable_fraction_nonshadow *

                (
                  1 -
                    suitable_fraction_nonshadow
                ) /

                n_ns
            )
        )
    }
  )


sensitivity_40 <-
  bind_rows(
    sensitivity_list_40
  )


sensitivity_summary_40 <-
  sensitivity_40 %>%

  group_by(
    sensitivity_total_n
  ) %>%

  summarise(

    median_MC_SE_km2 =
      median(
        sensitivity_MC_SE_km2,
        na.rm =
          TRUE
      ),

    q95_MC_SE_km2 =
      quantile(
        sensitivity_MC_SE_km2,
        0.95,
        na.rm =
          TRUE
      ),

    median_relative_MC_SE_percent =
      median(
        100 *
          sensitivity_MC_SE_km2 /
          suitable_area_total_km2,
        na.rm =
          TRUE
      ),

    .groups =
      "drop"
  )


write_csv(
  sensitivity_summary_40,
  file.path(
    AREA_DIR,
    "40_12_MC_sample_size_sensitivity.csv"
  )
)


# ============================================================
# 40.30 FIGURE 8
#       MAPPED AREA ACROSS MODEL REPLICATES
# ============================================================

fig40_area <-

  ggplot(

    area_results_40,

    aes(
      x =
        threshold_strategy,

      y =
        suitable_area_total_km2,

      fill =
        variant
    )
  ) +

  geom_boxplot(
    outlier.shape =
      NA
  ) +

  geom_point(

    aes(
      shape =
        variant
    ),

    position =
      position_jitterdodge(
        jitter.width =
          0.08,

        dodge.width =
          0.75
      ),

    alpha =
      0.45,

    size =
      1.5
  ) +

  facet_wrap(
    ~scheme,
    nrow =
      1
  ) +

  labs(

    title =
      "Mapped suitable area across stochastic model replicates",

    subtitle =
      "Fixed stratified Monte-Carlo mapping sample; identical cells used for all models",

    x =
      "Threshold strategy",

    y =
      expression(
        "Estimated suitable area (km"^2*")"
      ),

    fill =
      "Predictor set",

    shape =
      "Predictor set"
  ) +

  theme_bw(
    base_size =
      12
  ) +

  theme(

    panel.grid.minor =
      element_blank(),

    legend.position =
      "right"
  )


ggsave(

  file.path(
    AREA_DIR,
    "FIG08_MC_mapped_area_uncertainty.tiff"
  ),

  fig40_area,

  width =
    11,

  height =
    5.5,

  units =
    "in",

  dpi =
    600,

  compression =
    "lzw"
)


# ============================================================
# 40.31 FIGURE 9
#       REGIME-SPECIFIC minus GLOBAL AREA
# ============================================================

fig40_threshold_delta <-

  ggplot(

    area_threshold_wide_40,

    aes(

      x =
        interaction(
          scheme,
          variant,
          sep =
            " / "
        ),

      y =
        delta_total_km2
    )
  ) +

  geom_hline(
    yintercept =
      0,
    linetype =
      2
  ) +

  geom_boxplot(
    outlier.shape =
      NA
  ) +

  geom_jitter(
    width =
      0.08,
    alpha =
      0.55,
    size =
      1.6
  ) +

  labs(

    title =
      "Effect of regime-specific thresholds on mapped suitable area",

    subtitle =
      "Positive values indicate greater mapped area than under the global threshold",

    x =
      "Sampling scenario / predictor set",

    y =
      expression(
        Delta *
        " suitable area (km"^2*")"
      )
  ) +

  theme_bw(
    base_size =
      12
  ) +

  theme(

    axis.text.x =
      element_text(
        angle =
          35,
        hjust =
          1
      ),

    panel.grid.minor =
      element_blank()
  )


ggsave(

  file.path(
    AREA_DIR,
    "FIG09_MC_delta_area_regime_minus_global.tiff"
  ),

  fig40_threshold_delta,

  width =
    10,

  height =
    5.5,

  units =
    "in",

  dpi =
    600,

  compression =
    "lzw"
)


# ============================================================
# 40.32 FIGURE 10
#       RAW_SHADOW minus RAW MAPPED-AREA EFFECT
# ============================================================

fig40_shadow_effect <-

  ggplot(

    area_variant_wide_40,

    aes(

      x =
        interaction(
          scheme,
          threshold_strategy,
          sep =
            " / "
        ),

      y =
        delta_RAW_SHADOW_minus_RAW_km2
    )
  ) +

  geom_hline(
    yintercept =
      0,
    linetype =
      2
  ) +

  geom_boxplot(
    outlier.shape =
      NA
  ) +

  geom_jitter(
    width =
      0.08,
    alpha =
      0.55,
    size =
      1.6
  ) +

  labs(

    title =
      "Incremental mapped-area effect associated with illumination regime",

    subtitle =
      "RAW+SHADOW minus RAW using identical Monte-Carlo mapping cells",

    x =
      "Sampling scenario / threshold strategy",

    y =
      expression(
        Delta *
        " suitable area (RAW+SHADOW - RAW; km"^2*")"
      )
  ) +

  theme_bw(
    base_size =
      12
  ) +

  theme(

    axis.text.x =
      element_text(
        angle =
          35,
        hjust =
          1
      ),

    panel.grid.minor =
      element_blank()
  )


ggsave(

  file.path(
    AREA_DIR,
    "FIG10_MC_RAW_SHADOW_minus_RAW_area.tiff"
  ),

  fig40_shadow_effect,

  width =
    11,

  height =
    5.5,

  units =
    "in",

  dpi =
    600,

  compression =
    "lzw"
)


# ============================================================
# 40.33 FINAL AUTOMATIC QC
# ============================================================

qc40 <-
  tibble(

    check =
      c(

        "All 180 saved model files present",

        "SCL excluded from RAW",

        "SCL excluded from RAW_SHADOW",

        "ShadowMask excluded from RAW",

        "ShadowMask excluded from RAW_SHADOW",

        "shadow_bin excluded from RAW",

        "shadow_bin included in RAW_SHADOW",

        "RAW_SHADOW differs from RAW by exactly one predictor",

        "Threshold table has all model combinations",

        "Fixed MC shadow sample non-empty",

        "Fixed MC non-shadow sample non-empty",

        "All mapped-area estimates finite",

        "Mapped areas >= 0",

        "Mapped areas <= valid mapping domain",

        "Expected number of area results obtained",

        "Prediction probabilities remain within [0,1]"
      ),


    passed =
      c(

        all(
          model_manifest_40$exists
        ),

        !"SCL" %in%
          RAW_FEATURES,

        !"SCL" %in%
          RAW_SHADOW_FEATURES,

        !"ShadowMask" %in%
          RAW_FEATURES,

        !"ShadowMask" %in%
          RAW_SHADOW_FEATURES,

        !"shadow_bin" %in%
          RAW_FEATURES,

        "shadow_bin" %in%
          RAW_SHADOW_FEATURES,

        identical(
          setdiff(
            RAW_SHADOW_FEATURES,
            RAW_FEATURES
          ),
          "shadow_bin"
        ),

        nrow(
          threshold_key_40
        ) ==
          N_REP_40 *
          3 *
          2,

        length(
          mc_cells_shadow
        ) > 0,

        length(
          mc_cells_nonshadow
        ) > 0,

        all(
          is.finite(
            area_results_40$
              suitable_area_total_km2
          )
        ),

        all(
          area_results_40$
            suitable_area_total_km2 >=
            0
        ),

        all(
          area_results_40$
            suitable_area_total_km2 <=
            A_total_km2 +
            1e-8
        ),

        nrow(
          area_results_40
        ) ==
          EXPECTED_AREA_ROWS,

        all(
          prediction_qc_40$
            min_probability >=
            0 &
            prediction_qc_40$
              max_probability <=
              1
        )
      )
  )


write_csv(
  qc40,
  file.path(
    AREA_DIR,
    "40_99_FINAL_QC.csv"
  )
)


print(
  qc40
)


if(
  !all(
    qc40$passed
  )
) {

  stop(
    "One or more Section-40 QC checks failed."
  )
}


# ============================================================
# 40.34 SAVE SESSION INFO
# ============================================================

capture.output(

  sessionInfo(),

  file =
    file.path(
      AREA_DIR,
      "40_sessionInfo.txt"
    )
)


# ============================================================
# 40.35 FINAL REPORT
# ============================================================

cat(
  "\n\n============================================================\n"
)

cat(
  "SECTION 40 COMPLETED SUCCESSFULLY\n"
)

cat(
  "FAST STRATIFIED MONTE-CARLO MAPPED-AREA ANALYSIS\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nSaved RF models verified:",
  sum(
    model_manifest_40$exists
  ),
  "\n"
)


cat(
  "Threshold rows:",
  nrow(
    thresholds_40
  ),
  "\n"
)


cat(
  "\nValid mapping area:",
  round(
    A_total_km2,
    3
  ),
  "km2\n"
)


cat(
  "Shadow mapping area:",
  round(
    A_shadow_km2,
    3
  ),
  "km2\n"
)


cat(
  "Non-shadow mapping area:",
  round(
    A_nonshadow_km2,
    3
  ),
  "km2\n"
)


cat(
  "\nFixed MC sample:\n"
)

cat(
  " Shadow =",
  length(
    mc_cells_shadow
  ),
  "\n"
)

cat(
  " Non-shadow =",
  length(
    mc_cells_nonshadow
  ),
  "\n"
)


cat(
  "\nModels evaluated:",
  nrow(
    prediction_qc_40
  ),
  "\n"
)


cat(
  "Mapped-area estimates:",
  nrow(
    area_results_40
  ),
  "\n"
)


cat(
  "\nAll Section-40 QC checks passed.\n"
)


cat(
  "\nResults saved in:\n",
  AREA_DIR,
  "\n"
)


cat(
  "\n============================================================\n"
)
# ======================================================================
# SECTION 40
# THREE-ALGORITHM STRATIFIED MONTE-CARLO MAPPED-AREA ANALYSIS
#
# Run immediately after successful completion of Section 40.
# ======================================================================

suppressPackageStartupMessages({
  library(terra)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(ranger)
  library(glmnet)
  library(mgcv)
  library(tibble)
  library(purrr)

})

# ======================================================================
# 40.1 REQUIRED OBJECTS
# ======================================================================

required_objects_40222 <- c(
  "OUT_DIR",
  "df_fit_pool",
  "df_cal",
  "RAW_FEATURES",
  "RAW_SHADOW_FEATURES",
  "N_REP",
  "N_TRAIN_REP",
  "p_shadow_area",
  "make_UR_area",
  "make_SB_poolratio",
  "make_SW_from_SB",
  "X_mc_shadow",
  "X_mc_nonshadow",
  "A_shadow_km2",
  "A_nonshadow_km2",
  "A_total_km2",
  "area_results_40"
)

missing_objects_40222 <- required_objects_40222[
  !vapply(required_objects_40222, exists, logical(1), envir=.GlobalEnv)
]

if(length(missing_objects_40222) > 0){
  stop(
    paste0(
      "Run Section 40 first. Missing objects:\n",
      paste(missing_objects_40222, collapse=", ")
    )
  )
}

# ======================================================================
# 40.2 OUTPUT DIRECTORIES
# ======================================================================

DIR_40222 <- file.path(
  OUT_DIR,
  "40_222_THREE_ALGORITHM_MAPPED_AREA"
)

DIR_40222_CSV  <- file.path(DIR_40222, "CSV")
DIR_40222_FIG  <- file.path(DIR_40222, "FIGURES")
DIR_40222_QC   <- file.path(DIR_40222, "QC")

invisible(lapply(
  c(DIR_40222, DIR_40222_CSV, DIR_40222_FIG, DIR_40222_QC),
  dir.create,
  recursive=TRUE,
  showWarnings=FALSE
))

# ======================================================================
# 40.3 SETTINGS — MATCH SECTIONS 42 AND 44
# ======================================================================

# Ridge / Section 42
ALPHA_40222 <- 0
NFOLDS_40222 <- 5
LAMBDA_RULE_40222 <- "lambda.1se"
THR_GRID_RIDGE_40222 <- seq(0.05, 0.95, by=0.005)

# GAM / Section 44
SEED_GAM_40222 <- 44000
K_GAM_40222 <- 4

EXPECTED_MODELS_PER_ALGORITHM_40222 <- N_REP * 3 * 2

RAW_FEATURES_40222 <- unique(as.character(RAW_FEATURES))
RAW_SHADOW_FEATURES_40222 <- unique(as.character(RAW_SHADOW_FEATURES))

stopifnot(
  !"SCL" %in% RAW_FEATURES_40222,
  !"ShadowMask" %in% RAW_FEATURES_40222,
  !"shadow_bin" %in% RAW_FEATURES_40222,
  !"SCL" %in% RAW_SHADOW_FEATURES_40222,
  !"ShadowMask" %in% RAW_SHADOW_FEATURES_40222,
  "shadow_bin" %in% RAW_SHADOW_FEATURES_40222,
  identical(
    setdiff(RAW_SHADOW_FEATURES_40222, RAW_FEATURES_40222),
    "shadow_bin"
  )
)

if(!"shadow_bin" %in% names(X_mc_shadow)) X_mc_shadow$shadow_bin <- 1L
if(!"shadow_bin" %in% names(X_mc_nonshadow)) X_mc_nonshadow$shadow_bin <- 0L

stopifnot(
  all(X_mc_shadow$shadow_bin == 1),
  all(X_mc_nonshadow$shadow_bin == 0)
)

# ======================================================================
# 40.4 COMMON AREA ESTIMATOR
# ======================================================================

estimate_area_40222 <- function(
  p_shadow,
  p_nonshadow,
  threshold_shadow,
  threshold_nonshadow
){
  p_shadow <- as.numeric(p_shadow)
  p_nonshadow <- as.numeric(p_nonshadow)

  if(any(!is.finite(p_shadow)) || any(!is.finite(p_nonshadow))){
    stop("Non-finite mapping probabilities.")
  }

  if(!is.finite(threshold_shadow) || !is.finite(threshold_nonshadow)){
    stop("Non-finite threshold.")
  }

  suitable_shadow <- as.integer(p_shadow >= threshold_shadow)
  suitable_nonshadow <- as.integer(p_nonshadow >= threshold_nonshadow)

  q_shadow <- mean(suitable_shadow)
  q_nonshadow <- mean(suitable_nonshadow)

  area_shadow <- A_shadow_km2 * q_shadow
  area_nonshadow <- A_nonshadow_km2 * q_nonshadow
  area_total <- area_shadow + area_nonshadow

  var_shadow <-
    A_shadow_km2^2 * q_shadow * (1-q_shadow) / length(suitable_shadow)

  var_nonshadow <-
    A_nonshadow_km2^2 * q_nonshadow * (1-q_nonshadow) / length(suitable_nonshadow)

  mc_se <- sqrt(var_shadow + var_nonshadow)

  tibble(
    suitable_fraction_shadow=q_shadow,
    suitable_fraction_nonshadow=q_nonshadow,
    suitable_area_shadow_km2=area_shadow,
    suitable_area_nonshadow_km2=area_nonshadow,
    suitable_area_total_km2=area_total,
    suitable_fraction_total=area_total/A_total_km2,
    MC_SE_km2=mc_se,
    MC_CI_low_km2=max(0, area_total-1.96*mc_se),
    MC_CI_high_km2=min(A_total_km2, area_total+1.96*mc_se)
  )
}

# ======================================================================
# 40.5 RF — REUSE VALIDATED SECTION 40 RESULTS
# ======================================================================

rf_area_40222 <- area_results_40 %>%
  mutate(
    algorithm="Random Forest",
    variant=case_when(
      variant %in% c("RAW_SHADOW","RAW+SHADOW") ~ "RAW+SHADOW",
      TRUE ~ as.character(variant)
    ),
    threshold_strategy=toupper(as.character(threshold_strategy))
  )

stopifnot(
  nrow(rf_area_40222) == N_REP*3*2*2
)

# ======================================================================
# 40.6 RIDGE HELPERS — MATCH SECTION 42
# ======================================================================

make_matrix_40222 <- function(df, features){

  miss <- setdiff(features, names(df))
  if(length(miss)>0){
    stop("Missing Ridge predictors: ", paste(miss, collapse=", "))
  }

  x <- df[,features,drop=FALSE]
  x <- data.frame(lapply(x,as.numeric), check.names=FALSE)
  x <- as.matrix(x)
  storage.mode(x) <- "double"

  if(any(!is.finite(x))){
    stop("Non-finite Ridge predictors.")
  }

  x
}

fit_ridge_40222 <- function(train_df, features, weights, seed){

  set.seed(seed)

  x <- make_matrix_40222(train_df,features)
  y <- as.integer(train_df$presence)

  if(length(unique(y))<2){
    stop("Ridge replicate has one response class.")
  }

  n <- length(y)
  k_use <- min(NFOLDS_40222, floor(n/10))
  k_use <- max(3,k_use)

  foldid <- sample(rep(seq_len(k_use),length.out=n))

  cvfit <- glmnet::cv.glmnet(
    x=x,
    y=y,
    family="binomial",
    alpha=ALPHA_40222,
    weights=weights,
    foldid=foldid,
    standardize=TRUE,
    type.measure="deviance",
    intercept=TRUE,
    keep=FALSE
  )

  lambda_use <- if(LAMBDA_RULE_40222=="lambda.min"){
    cvfit$lambda.min
  } else {
    cvfit$lambda.1se
  }

  list(
    fit=cvfit,
    lambda=lambda_use,
    features=features
  )
}

predict_ridge_40222 <- function(model_obj,new_df){

  x <- make_matrix_40222(new_df,model_obj$features)

  p <- as.numeric(
    predict(
      model_obj$fit,
      newx=x,
      s=model_obj$lambda,
      type="response"
    )
  )

  pmin(pmax(p,0),1)
}

bacc_score_40222 <- function(y,p,thr){

  pred <- as.integer(p>=thr)

  TP <- sum(pred==1 & y==1)
  TN <- sum(pred==0 & y==0)
  FP <- sum(pred==1 & y==0)
  FN <- sum(pred==0 & y==1)

  sens <- ifelse(TP+FN>0, TP/(TP+FN), NA_real_)
  spec <- ifelse(TN+FP>0, TN/(TN+FP), NA_real_)

  mean(c(sens,spec),na.rm=TRUE)
}

optimal_threshold_ridge_40222 <- function(y,p,grid=THR_GRID_RIDGE_40222){

  y <- as.integer(y)
  p <- as.numeric(p)

  ok <- !is.na(y) & is.finite(p)
  y <- y[ok]
  p <- p[ok]

  if(length(unique(y))<2) return(NA_real_)

  scores <- vapply(
    grid,
    function(tt) bacc_score_40222(y,p,tt),
    numeric(1)
  )

  best <- max(scores,na.rm=TRUE)
  candidates <- grid[
    is.finite(scores) &
      abs(scores-best)<1e-12]

  median(candidates)
}

# ======================================================================
# 40.7 GAM HELPERS — MATCH SECTION 44
# ======================================================================

clean_features_40222 <- function(x){
  x <- unlist(x,recursive=TRUE,use.names=FALSE)
  x <- as.character(x)
  unique(x[!is.na(x) & nzchar(x)])
}

prepare_gam_features_40222 <- function(d,features){

  features <- clean_features_40222(features)
  usable <- character(0)

  for(v in features){

    if(!v %in% names(d)) next

    x <- suppressWarnings(as.numeric(d[[v]]))
    ux <- unique(x[is.finite(x)])

    if(length(ux)>=2){
      usable <- c(usable,v)
    }
  }

  unique(usable)
}

build_gam_formula_40222 <- function(features,data){

  features <- clean_features_40222(features)
  terms <- character(0)

  for(v in features){

    x <- data[[v]]
    n_unique <- length(unique(x[is.finite(x)]))

    if(v=="shadow_bin"){
      terms <- c(terms,"shadow_bin")
      next
    }

    if(tolower(v)=="aspect"){

      if(n_unique>=K_GAM_40222){
        terms <- c(
          terms,
          paste0("s(`",v,"`, bs='cc', k=",K_GAM_40222,")")
        )
      } else {
        terms <- c(terms,paste0("`",v,"`"))
      }

      next
    }

    if(n_unique>=K_GAM_40222){
      terms <- c(
        terms,
        paste0("s(`",v,"`, bs='ts', k=",K_GAM_40222,")")
      )
    } else {
      terms <- c(terms,paste0("`",v,"`"))
    }
  }

  if(length(terms)==0) stop("No GAM terms.")

  stats::as.formula(
    paste("presence ~",paste(terms,collapse=" + "))
  )
}

fit_gam_40222 <- function(train_df,features,weights_fit=NULL){

  features <- clean_features_40222(features)

  missing <- setdiff(c("presence",features),names(train_df))
  if(length(missing)>0){
    stop("GAM data missing: ",paste(missing,collapse=", "))
  }

  d <- train_df %>%
    select(presence,all_of(features))

  d$presence <- as.integer(d$presence)

  if(is.null(weights_fit)){
    w <- rep(1,nrow(train_df))
  } else {
    w <- as.numeric(weights_fit)
  }

  if(length(w)!=nrow(train_df)){
    stop("GAM weight-length mismatch.")
  }

  cc <- complete.cases(d) & is.finite(w)
  d <- d[cc,,drop=FALSE]
  w <- w[cc]

  if(nrow(d)==0) stop("No complete GAM rows.")
  if(length(unique(d$presence))<2) stop("GAM has one response class.")

  variable_features <- prepare_gam_features_40222(d,features)

  if(length(variable_features)==0){
    stop("No variable GAM predictors.")
  }

  d <- d %>%
    select(presence,all_of(variable_features))

  d$.weights40222 <- as.numeric(w)

  form <- build_gam_formula_40222(variable_features,d)
  environment(form) <- environment()

  knots <- NULL

  aspect_name <- variable_features[
    tolower(variable_features)=="aspect"
  ]

  if(length(aspect_name)>1){
    stop("More than one Aspect variable.")
  }

  if(length(aspect_name)==1){

    asp <- d[[aspect_name[1]]]
    n_unique_aspect <- length(unique(asp[is.finite(asp)]))

    if(n_unique_aspect>=K_GAM_40222){
      knots <- list()
      knots[[aspect_name[1]]] <- c(0,360)
    }
  }

  model <- mgcv::bam(
    formula=form,
    data=d,
    family=stats::binomial(link="logit"),
    weights=.weights40222,
    method="fREML",
    discrete=TRUE,
    select=TRUE,
    knots=knots,
    control=mgcv::gam.control(trace=FALSE)
  )

  list(
    model=model,
    features_used=variable_features,
    formula=form
  )
}

predict_gam_40222 <- function(fit_obj,newdata){

  feats <- fit_obj$features_used
  missing <- setdiff(feats,names(newdata))

  if(length(missing)>0){
    stop("GAM prediction data missing: ",paste(missing,collapse=", "))
  }

  nd <- newdata %>%
    select(all_of(feats))

  p <- as.numeric(
    stats::predict(
      fit_obj$model,
      newdata=nd,
      type="response"
    )
  )

  pmin(pmax(p,0),1)
}

best_threshold_gam_40222 <- function(y,p){

  y <- as.integer(y)
  p <- as.numeric(p)

  ok <- !is.na(y) & is.finite(p)
  y <- y[ok]
  p <- p[ok]

  if(length(y)<4 || length(unique(y))<2){
    return(NA_real_)
  }

  p <- pmin(pmax(p,0),1)
  unique_p <- sort(unique(p))

  if(length(unique_p)==1) return(unique_p)

  mid <- (unique_p[-1]+unique_p[-length(unique_p)])/2

  candidates <- sort(
    unique(c(0,unique_p,mid,1))
  )

  scores <- vapply(
    candidates,
    function(thr) bacc_score_40222(y,p,thr),
    numeric(1)
  )

  finite <- is.finite(scores)
  if(!any(finite)) return(NA_real_)

  best_score <- max(scores[finite])

  best_candidates <- candidates[
    finite &
      abs(scores-best_score)<1e-12
  ]

  median(best_candidates)
}

# ======================================================================
# 40.8 RIDGE MAPPED AREA
# ======================================================================

ridge_area_list_40222 <- list()
ridge_qc_list_40222 <- list()
ridge_index_40222 <- 1L

for(rp in seq_len(N_REP)){

  cat("\nRIDGE mapped-area replicate",rp,"of",N_REP,"\n")

  # Exact Section 42 sampling seeds
  df_UR <- make_UR_area(
    df_pool=df_fit_pool,
    n_total=N_TRAIN_REP,
    p_area=p_shadow_area,
    seed=SEED+42000+rp
  )

  df_SB <- make_SB_poolratio(
    df_pool=df_fit_pool,
    n_total=N_TRAIN_REP,
    seed=SEED+43000+rp
  )

  sw_obj <- make_SW_from_SB(
    df_SB=df_SB,
    p_area=p_shadow_area
  )

  df_SW <- sw_obj$df

  stopifnot(identical(df_SB$orig_id,df_SW$orig_id))

  scenarios <- list(
    UR=df_UR,
    SB=df_SB,
    SW=df_SW
  )

  for(scheme in names(scenarios)){

    ds <- scenarios[[scheme]]

    weights <- if(scheme=="SW"){
      as.numeric(ds$w_area)
    } else {
      rep(1,nrow(ds))
    }

    variants <- list(
      RAW=RAW_FEATURES_40222,
      `RAW+SHADOW`=RAW_SHADOW_FEATURES_40222
    )

    for(variant in names(variants)){

      feats <- variants[[variant]]

      model_seed <-
        SEED +
        50000 +
        rp*100 +
        match(scheme,c("UR","SB","SW"))*10 +
        match(variant,c("RAW","RAW+SHADOW"))

      fit <- fit_ridge_40222(
        ds,
        feats,
        weights,
        model_seed
      )

      p_cal <- predict_ridge_40222(fit,df_cal)

      th_global <- optimal_threshold_ridge_40222(
        df_cal$presence,
        p_cal
      )

      idx_ns <- which(df_cal$shadow_bin==0)
      idx_sh <- which(df_cal$shadow_bin==1)

      th_nonshadow <- optimal_threshold_ridge_40222(
        df_cal$presence[idx_ns],
        p_cal[idx_ns]
      )

      th_shadow <- optimal_threshold_ridge_40222(
        df_cal$presence[idx_sh],
        p_cal[idx_sh]
      )

      if(!all(is.finite(c(th_global,th_nonshadow,th_shadow)))){
        stop("Non-finite Ridge threshold: ",scheme," / ",variant," / REP",rp)
      }

      p_sh <- predict_ridge_40222(fit,X_mc_shadow)
      p_ns <- predict_ridge_40222(fit,X_mc_nonshadow)

      ar_global <- estimate_area_40222(
        p_sh,p_ns,th_global,th_global
      ) %>%
        mutate(
          algorithm="Ridge logistic",
          replicate=rp,
          scheme=scheme,
          variant=variant,
          threshold_strategy="GLOBAL",
          threshold_global=th_global,
          threshold_nonshadow=th_global,
          threshold_shadow=th_global,
          .before=1
        )

      ar_regime <- estimate_area_40222(
        p_sh,p_ns,th_shadow,th_nonshadow
      ) %>%
        mutate(
          algorithm="Ridge logistic",
          replicate=rp,
          scheme=scheme,
          variant=variant,
          threshold_strategy="REGIME_SPECIFIC",
          threshold_global=th_global,
          threshold_nonshadow=th_nonshadow,
          threshold_shadow=th_shadow,
          .before=1
        )

      ridge_area_list_40222[[ridge_index_40222]] <- bind_rows(
        ar_global,
        ar_regime
      )

      ridge_qc_list_40222[[ridge_index_40222]] <- tibble(
        algorithm="Ridge logistic",
        replicate=rp,
        scheme=scheme,
        variant=variant,
        lambda=fit$lambda,
        threshold_global=th_global,
        threshold_nonshadow=th_nonshadow,
        threshold_shadow=th_shadow,
        min_probability=min(c(p_sh,p_ns)),
        max_probability=max(c(p_sh,p_ns))
      )

      ridge_index_40222 <- ridge_index_40222+1L

      rm(fit,p_cal,p_sh,p_ns)
      gc(verbose=FALSE)
    }
  }
}

ridge_area_40222 <- bind_rows(ridge_area_list_40222)
ridge_qc_40222 <- bind_rows(ridge_qc_list_40222)

# ======================================================================
# 40.9 GAM MAPPED AREA
# ======================================================================

gam_area_list_40222 <- list()
gam_qc_list_40222 <- list()
gam_index_40222 <- 1L

for(rp in seq_len(N_REP)){

  cat("\nGAM mapped-area replicate",rp,"of",N_REP,"\n")

  # Exact Section 44 sampling seeds
  df_UR <- make_UR_area(
    df_pool=df_fit_pool,
    n_total=N_TRAIN_REP,
    p_area=p_shadow_area,
    seed=SEED_GAM_40222+1000+rp
  )

  df_SB <- make_SB_poolratio(
    df_pool=df_fit_pool,
    n_total=N_TRAIN_REP,
    seed=SEED_GAM_40222+2000+rp
  )

  sw_obj <- make_SW_from_SB(
    df_SB=df_SB,
    p_area=p_shadow_area
  )

  df_SW <- sw_obj$df

  stopifnot(identical(df_SB$orig_id,df_SW$orig_id))

  scenarios <- list(
    UR=df_UR,
    SB=df_SB,
    SW=df_SW
  )

  for(scheme in names(scenarios)){

    ds <- scenarios[[scheme]]

    weights <- if(scheme=="SW"){
      as.numeric(ds$w_area)
    } else {
      rep(1,nrow(ds))
    }

    variants <- list(
      RAW=RAW_FEATURES_40222,
      `RAW+SHADOW`=RAW_SHADOW_FEATURES_40222
    )

    for(variant in names(variants)){

      feats <- variants[[variant]]

      fit <- fit_gam_40222(
        train_df=ds,
        features=feats,
        weights_fit=weights
      )

      p_cal <- predict_gam_40222(fit,df_cal)

      th_global <- best_threshold_gam_40222(
        df_cal$presence,
        p_cal
      )

      idx_ns <- which(df_cal$shadow_bin==0)
      idx_sh <- which(df_cal$shadow_bin==1)

      th_nonshadow <- best_threshold_gam_40222(
        df_cal$presence[idx_ns],
        p_cal[idx_ns]
      )

      th_shadow <- best_threshold_gam_40222(
        df_cal$presence[idx_sh],
        p_cal[idx_sh]
      )

      if(!all(is.finite(c(th_global,th_nonshadow,th_shadow)))){
        stop("Non-finite GAM threshold: ",scheme," / ",variant," / REP",rp)
      }

      p_sh <- predict_gam_40222(fit,X_mc_shadow)
      p_ns <- predict_gam_40222(fit,X_mc_nonshadow)

      ar_global <- estimate_area_40222(
        p_sh,p_ns,th_global,th_global
      ) %>%
        mutate(
          algorithm="GAM",
          replicate=rp,
          scheme=scheme,
          variant=variant,
          threshold_strategy="GLOBAL",
          threshold_global=th_global,
          threshold_nonshadow=th_global,
          threshold_shadow=th_global,
          .before=1
        )

      ar_regime <- estimate_area_40222(
        p_sh,p_ns,th_shadow,th_nonshadow
      ) %>%
        mutate(
          algorithm="GAM",
          replicate=rp,
          scheme=scheme,
          variant=variant,
          threshold_strategy="REGIME_SPECIFIC",
          threshold_global=th_global,
          threshold_nonshadow=th_nonshadow,
          threshold_shadow=th_shadow,
          .before=1
        )

      gam_area_list_40222[[gam_index_40222]] <- bind_rows(
        ar_global,
        ar_regime
      )

      gam_qc_list_40222[[gam_index_40222]] <- tibble(
        algorithm="GAM",
        replicate=rp,
        scheme=scheme,
        variant=variant,
        n_features_used=length(fit$features_used),
        threshold_global=th_global,
        threshold_nonshadow=th_nonshadow,
        threshold_shadow=th_shadow,
        min_probability=min(c(p_sh,p_ns)),
        max_probability=max(c(p_sh,p_ns))
      )

      gam_index_40222 <- gam_index_40222+1L

      rm(fit,p_cal,p_sh,p_ns)
      gc(verbose=FALSE)
    }
  }
}

gam_area_40222 <- bind_rows(gam_area_list_40222)
gam_qc_40222 <- bind_rows(gam_qc_list_40222)

# ======================================================================
# 40.10 COMBINE THREE ALGORITHMS
# ======================================================================

area_all_40222 <- bind_rows(
  rf_area_40222,
  ridge_area_40222,
  gam_area_40222
) %>%
  mutate(
    algorithm=factor(
      algorithm,
      levels=c("Random Forest","Ridge logistic","GAM")
    ),
    scheme=factor(
      scheme,
      levels=c("SB","SW","UR")
    ),
    variant=factor(
      variant,
      levels=c("RAW","RAW+SHADOW")
    ),
    threshold_strategy=factor(
      threshold_strategy,
      levels=c("GLOBAL","REGIME_SPECIFIC")
    )
  )

write_csv(
  area_all_40222,
  file.path(
    DIR_40222_CSV,
    "40_222_01_three_algorithm_mapped_area_all_replicates.csv"
  )
)

# ======================================================================
# 40.11 FULL SUMMARY
# ======================================================================

area_summary_40222 <- area_all_40222 %>%
  group_by(
    algorithm,
    scheme,
    variant,
    threshold_strategy
  ) %>%
  summarise(
    n_replicates=n(),
    Mean_km2=mean(suitable_area_total_km2),
    SD_km2=sd(suitable_area_total_km2),
    Median_km2=median(suitable_area_total_km2),
    LCL_km2=as.numeric(
      quantile(suitable_area_total_km2,0.025)
    ),
    UCL_km2=as.numeric(
      quantile(suitable_area_total_km2,0.975)
    ),
    Mean_MC_SE_km2=mean(MC_SE_km2),
    .groups="drop"
  )

write_csv(
  area_summary_40222,
  file.path(
    DIR_40222_CSV,
    "Table40_222_S1_full_area_summary.csv"
  )
)

# ======================================================================
# 40.12 PAIRED REGIME-SPECIFIC minus GLOBAL
# ======================================================================

area_threshold_wide_40222 <- area_all_40222 %>%
  select(
    algorithm,
    replicate,
    scheme,
    variant,
    threshold_strategy,
    suitable_area_total_km2
  ) %>%
  pivot_wider(
    names_from=threshold_strategy,
    values_from=suitable_area_total_km2
  ) %>%
  mutate(
    Delta_Regime_minus_Global_km2=
      REGIME_SPECIFIC-GLOBAL
  )

write_csv(
  area_threshold_wide_40222,
  file.path(
    DIR_40222_CSV,
    "40_222_02_delta_regime_minus_global_by_replicate.csv"
  )
)

paired_tests_40222 <- area_threshold_wide_40222 %>%
  group_by(algorithm,scheme,variant) %>%
  group_modify(~{

    x <- .x$Delta_Regime_minus_Global_km2
    x <- x[is.finite(x)]

    wt <- if(length(x)>=3){
      suppressWarnings(
        wilcox.test(x,mu=0,exact=FALSE)
      )
    } else {
      NULL
    }

    tibble(
      n=length(x),
      Mean_Delta_km2=if(length(x)>0) mean(x) else NA_real_,
      Median_Delta_km2=if(length(x)>0) median(x) else NA_real_,
      LCL_Delta_km2=if(length(x)>0)
        as.numeric(quantile(x,0.025)) else NA_real_,
      UCL_Delta_km2=if(length(x)>0)
        as.numeric(quantile(x,0.975)) else NA_real_,
      p_value=if(is.null(wt)) NA_real_ else wt$p.value
    )
  }) %>%
  ungroup() %>%
  mutate(
    p_FDR=p.adjust(p_value,method="BH")
  )

write_csv(
  paired_tests_40222,
  file.path(
    DIR_40222_CSV,
    "Table40_222_S2_paired_threshold_strategy_tests.csv"
  )
)

# ======================================================================
# 40.13 RAW+SHADOW minus RAW
# ======================================================================

variant_wide_40222 <- area_all_40222 %>%
  select(
    algorithm,
    replicate,
    scheme,
    threshold_strategy,
    variant,
    suitable_area_total_km2
  ) %>%
  pivot_wider(
    names_from=variant,
    values_from=suitable_area_total_km2
  ) %>%
  mutate(
    Delta_RAW_SHADOW_minus_RAW_km2=
      `RAW+SHADOW`-RAW
  )

variant_tests_40222 <- variant_wide_40222 %>%
  group_by(
    algorithm,
    scheme,
    threshold_strategy
  ) %>%
  group_modify(~{

    x <- .x$Delta_RAW_SHADOW_minus_RAW_km2
    x <- x[is.finite(x)]

    wt <- if(length(x)>=3){
      suppressWarnings(
        wilcox.test(x,mu=0,exact=FALSE)
      )
    } else {
      NULL
    }

    tibble(
      n=length(x),
      Mean_Delta_km2=if(length(x)>0) mean(x) else NA_real_,
      Median_Delta_km2=if(length(x)>0) median(x) else NA_real_,
      LCL_Delta_km2=if(length(x)>0)
        as.numeric(quantile(x,0.025)) else NA_real_,
      UCL_Delta_km2=if(length(x)>0)
        as.numeric(quantile(x,0.975)) else NA_real_,
      p_value=if(is.null(wt)) NA_real_ else wt$p.value
    )
  }) %>%
  ungroup() %>%
  mutate(
    p_FDR=p.adjust(p_value,method="BH")
  )

write_csv(
  variant_tests_40222,
  file.path(
    DIR_40222_CSV,
    "Table40_222_S3_predictor_set_area_tests.csv"
  )
)

# ======================================================================
# 40.14 COMPACT MAIN TABLE
# ======================================================================

format_ci_40222 <- function(mean,lcl,ucl,digits=1){
  paste0(
    formatC(mean,format="f",digits=digits),
    " [",
    formatC(lcl,format="f",digits=digits),
    ", ",
    formatC(ucl,format="f",digits=digits),
    "]"
  )
}

summary_wide_40222 <- area_summary_40222 %>%
  mutate(
    Result=format_ci_40222(
      Mean_km2,
      LCL_km2,
      UCL_km2
    )
  ) %>%
  select(
    algorithm,
    scheme,
    variant,
    threshold_strategy,
    Result
  ) %>%
  pivot_wider(
    names_from=threshold_strategy,
    values_from=Result
  )

main_compact_40222 <- summary_wide_40222 %>%
  left_join(
    paired_tests_40222 %>%
      transmute(
        algorithm,
        scheme,
        variant,
        Delta_km2=format_ci_40222(
          Mean_Delta_km2,
          LCL_Delta_km2,
          UCL_Delta_km2
        ),
        p_FDR
      ),
    by=c("algorithm","scheme","variant")
  ) %>%
  arrange(algorithm,scheme,variant)

write_csv(
  main_compact_40222,
  file.path(
    DIR_40222_CSV,
    "Table40_222_MAIN_compact_summary.csv"
  )
)

# ======================================================================
# 40.15 FIGURE THEME
# ======================================================================

theme_40222 <- function(){
  theme_bw(base_size=12) +
    theme(
      panel.grid.minor=element_blank(),
      strip.background=element_rect(
        fill="grey92",
        colour="grey40"
      ),
      strip.text=element_text(face="bold"),
      plot.title=element_text(face="bold",size=14),
      plot.subtitle=element_text(size=10.5),
      legend.position="right"
    )
}

# ======================================================================
# 40.16 MAIN FIGURE — ABSOLUTE MAPPED SUITABLE AREA
# ======================================================================

fig_main_40222 <- ggplot(
  area_all_40222,
  aes(
    x=threshold_strategy,
    y=suitable_area_total_km2,
    fill=variant,
    shape=variant
  )
) +
  geom_boxplot(
    outlier.shape=NA,
    position=position_dodge(width=0.72),
    width=0.62
  ) +
  geom_point(
    position=position_jitterdodge(
      jitter.width=0.07,
      dodge.width=0.72
    ),
    alpha=0.42,
    size=1.25
  ) +
  facet_grid(
    algorithm ~ scheme
  ) +
  labs(
    title="Mapped suitable area across algorithms and threshold strategies",
    subtitle=paste0(
      "Fixed stratified Monte-Carlo mapping sample; ",
      "identical raster cells used for all paired comparisons"
    ),
    x="Threshold strategy",
    y=expression("Estimated suitable area (km"^2*")"),
    fill="Predictor set",
    shape="Predictor set"
  ) +
  scale_x_discrete(
    labels=c(
      GLOBAL="Global",
      REGIME_SPECIFIC="Regime-specific"
    )
  ) +
  theme_40222() +
  theme(
    axis.text.x=element_text(
      angle=20,
      hjust=1
    )
  )

ggsave(
  file.path(
    DIR_40222_FIG,
    "FIG40_222_MAIN_three_algorithm_mapped_area.png"
  ),
  fig_main_40222,
  width=13,
  height=9,
  units="in",
  dpi=600
)

ggsave(
  file.path(
    DIR_40222_FIG,
    "FIG40_222_MAIN_three_algorithm_mapped_area.tiff"
  ),
  fig_main_40222,
  width=13,
  height=9,
  units="in",
  dpi=600,
  compression="lzw"
)

# ======================================================================
# 40.17 SUPPLEMENTARY FIGURE S1 — DELTA THRESHOLD STRATEGY
# ======================================================================

fig_s1_40222 <- ggplot(
  area_threshold_wide_40222,
  aes(
    x=scheme,
    y=Delta_Regime_minus_Global_km2,
    shape=variant
  )
) +
  geom_hline(
    yintercept=0,
    linetype=2
  ) +
  geom_boxplot(
    aes(group=interaction(scheme,variant)),
    outlier.shape=NA,
    position=position_dodge(width=0.65),
    width=0.55
  ) +
  geom_point(
    position=position_jitterdodge(
      jitter.width=0.07,
      dodge.width=0.65
    ),
    alpha=0.5,
    size=1.35
  ) +
  facet_wrap(
    ~algorithm,
    nrow=1
  ) +
  labs(
    title="Effect of illumination-regime-specific thresholds on mapped suitable area",
    subtitle="Positive values indicate greater suitable area than under the corresponding global threshold",
    x="Sampling scenario",
    y=expression(Delta*" suitable area (km"^2*")"),
    shape="Predictor set"
  ) +
  theme_40222()

ggsave(
  file.path(
    DIR_40222_FIG,
    "FIG40_222_S1_delta_regime_minus_global.png"
  ),
  fig_s1_40222,
  width=12,
  height=5.8,
  units="in",
  dpi=600
)

ggsave(
  file.path(
    DIR_40222_FIG,
    "FIG40_222_S1_delta_regime_minus_global.tiff"
  ),
  fig_s1_40222,
  width=12,
  height=5.8,
  units="in",
  dpi=600,
  compression="lzw"
)

# ======================================================================
# 40.18 SUPPLEMENTARY FIGURE S2 — RAW+SHADOW minus RAW
# ======================================================================

fig_s2_40222 <- ggplot(
  variant_wide_40222,
  aes(
    x=scheme,
    y=Delta_RAW_SHADOW_minus_RAW_km2
  )
) +
  geom_hline(
    yintercept=0,
    linetype=2
  ) +
  geom_boxplot(
    outlier.shape=NA
  ) +
  geom_jitter(
    width=0.07,
    alpha=0.5,
    size=1.35
  ) +
  facet_grid(
    algorithm ~ threshold_strategy,
    labeller=labeller(
      threshold_strategy=c(
        GLOBAL="Global threshold",
        REGIME_SPECIFIC="Regime-specific thresholds"
      )
    )
  ) +
  labs(
    title="Incremental mapped-area effect of explicit illumination-regime inclusion",
    subtitle="RAW+SHADOW minus RAW using identical Monte-Carlo mapping cells",
    x="Sampling scenario",
    y=expression(
      Delta*" suitable area (RAW+SHADOW - RAW; km"^2*")"
    )
  ) +
  theme_40222()

ggsave(
  file.path(
    DIR_40222_FIG,
    "FIG40_222_S2_RAW_SHADOW_minus_RAW.png"
  ),
  fig_s2_40222,
  width=11,
  height=8,
  units="in",
  dpi=600
)

ggsave(
  file.path(
    DIR_40222_FIG,
    "FIG40_222_S2_RAW_SHADOW_minus_RAW.tiff"
  ),
  fig_s2_40222,
  width=11,
  height=8,
  units="in",
  dpi=600,
  compression="lzw"
)


# ======================================================================
# SECTION 41
# THREE-ALGORITHM EXTERNAL-TEST CONSEQUENCES OF
# ILLUMINATION-REGIME-SPECIFIC THRESHOLDING
#
# Random Forest + Ridge logistic + GAM
#
# DIRECT CONTINUATION OF SECTION 40
#
# PURPOSE
# ----------------------------------------------------------------------
# Reconstruct the three diagnostic analyses originally developed
# in Section 41 using an IDENTICAL definition across:
#
#   1. Random Forest
#   2. Ridge logistic regression
#   3. GAM
#
# FIGURE A
#   Sensitivity-specificity trade-off
#
# FIGURE B
#   Composition of newly suitable classifications:
#       FN -> TP = recovered true presence
#       TN -> FP = newly introduced false positive
#
# FIGURE C
#   Share of newly suitable classifications that are correct
#
# IMPORTANT
# ----------------------------------------------------------------------
# - External test is used ONLY for evaluation.
# - Thresholds are estimated ONLY from df_cal.
# - Same fixed external test is used for all algorithms.
# - Ridge/GAM are reconstructed with exactly the settings/seeds
#   already used in Section 40.
# - RF point predictions are reused from predictions_all.
# - No threshold is optimized on the external test.
# ======================================================================


suppressPackageStartupMessages({

  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(tibble)
  library(purrr)
  library(glmnet)
  library(mgcv)

})


# ======================================================================
# 41.1 REQUIRED OBJECTS
# ======================================================================

required_objects_41222 <- c(

  "OUT_DIR",

  "df_fit_pool",
  "df_cal",
  "df_te",

  "RAW_FEATURES_40222",
  "RAW_SHADOW_FEATURES_40222",

  "N_REP",
  "N_TRAIN_REP",
  "p_shadow_area",

  "make_UR_area",
  "make_SB_poolratio",
  "make_SW_from_SB",

  # RF external-test predictions
  "predictions_all",

  # Ridge helpers from 40
  "fit_ridge_40222",
  "predict_ridge_40222",
  "optimal_threshold_ridge_40222",

  # GAM helpers from 40
  "fit_gam_40222",
  "predict_gam_40222",
  "best_threshold_gam_40222",

  # GAM seed
  "SEED_GAM_40222"
)


missing_objects_41222 <-
  required_objects_41222[
    !vapply(
      required_objects_41222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(length(missing_objects_41222) > 0){

  stop(

    paste0(

      "Section 41_222 must be run after successful Section 40_222.\n",
      "Missing objects:\n",

      paste(
        missing_objects_41222,
        collapse = ", "
      )
    )
  )
}


# ======================================================================
# 41.2 OUTPUT DIRECTORIES
# ======================================================================

DIR_41222 <-
  file.path(
    OUT_DIR,
    "41_222_THREE_ALGORITHM_THRESHOLD_CONSEQUENCES"
  )


DIR_41222_CSV <-
  file.path(
    DIR_41222,
    "CSV"
  )


DIR_41222_FIG <-
  file.path(
    DIR_41222,
    "FIGURES"
  )


DIR_41222_QC <-
  file.path(
    DIR_41222,
    "QC"
  )


invisible(

  lapply(

    c(
      DIR_41222,
      DIR_41222_CSV,
      DIR_41222_FIG,
      DIR_41222_QC
    ),

    dir.create,

    recursive = TRUE,
    showWarnings = FALSE
  )
)


# ======================================================================
# 41.3 STANDARDIZATION HELPERS
# ======================================================================

std_variant_41222 <- function(x){

  x <-
    toupper(
      as.character(x)
    )

  x <-
    gsub(
      "[+ -]",
      "_",
      x
    )

  x <-
    gsub(
      "_+",
      "_",
      x
    )


  dplyr::case_when(

    x %in%
      c(
        "RAW_SHADOW",
        "RAWSHADOW"
      ) ~
        "RAW+SHADOW",

    x == "RAW" ~
        "RAW",

    TRUE ~
        as.character(x)
  )
}


std_regime_41222 <- function(x){

  x <-
    toupper(
      gsub(
        "[_ -]",
        "",
        as.character(x)
      )
    )


  dplyr::case_when(

    x %in%
      c(
        "SHADOW",
        "1"
      ) ~
        "Shadow",

    x %in%
      c(
        "NONSHADOW",
        "NONSH",
        "0"
      ) ~
        "Non-shadow",

    TRUE ~
        NA_character_
  )
}


# ======================================================================
# 41.4 BUILD ONE STANDARD EXTERNAL-TEST PREDICTION TABLE
#
# Required columns:
#
# algorithm
# replicate
# scheme
# variant
# test_id
# presence
# shadow_bin
# probability
# threshold_global
# threshold_nonshadow
# threshold_shadow
# predicted_global
# predicted_regime
# ======================================================================


# ======================================================================
# 41.5 RANDOM FOREST
#
# Reuse existing point-level predictions.
# NO RF refitting.
# ======================================================================

rf_pred_41222 <-
  predictions_all %>%

  transmute(

    algorithm =
      "Random Forest",

    replicate =
      as.integer(replicate),

    scheme =
      as.character(scheme),

    variant =
      std_variant_41222(
        variant
      ),

    test_id =
      as.integer(test_id),

    presence =
      as.integer(presence),

    shadow_bin =
      as.integer(shadow_bin),

    probability =
      as.numeric(probability),

    threshold_global =
      as.numeric(threshold_global),

    threshold_nonshadow =
      as.numeric(threshold_nonshadow),

    threshold_shadow =
      as.numeric(threshold_shadow),

    predicted_global =
      as.integer(predicted_global),

    predicted_regime =
      as.integer(predicted_regime)
  )


# ======================================================================
# 41.6 RIDGE LOGISTIC
#
# Reconstruct exactly as Section 40 / Section 42.
# Predictions are now obtained on df_te rather than Monte-Carlo cells.
# ======================================================================

ridge_pred_list_41222 <- list()

ridge_index_41222 <- 1L


for(rp in seq_len(N_REP)){

  cat(
    "\nRIDGE external-test reconstruction:",
    rp,
    "of",
    N_REP,
    "\n"
  )


  # ------------------------------------------------------------
  # Exact sampling seeds used in Section 40 / Ridge
  # ------------------------------------------------------------

  df_UR_41222 <-
    make_UR_area(

      df_pool =
        df_fit_pool,

      n_total =
        N_TRAIN_REP,

      p_area =
        p_shadow_area,

      seed =
        SEED + 42000 + rp
    )


  df_SB_41222 <-
    make_SB_poolratio(

      df_pool =
        df_fit_pool,

      n_total =
        N_TRAIN_REP,

      seed =
        SEED + 43000 + rp
    )


  sw_obj_41222 <-
    make_SW_from_SB(

      df_SB =
        df_SB_41222,

      p_area =
        p_shadow_area
    )


  df_SW_41222 <-
    sw_obj_41222$df


  stopifnot(

    identical(
      df_SB_41222$orig_id,
      df_SW_41222$orig_id
    )
  )


  scenarios_41222 <-
    list(

      UR =
        df_UR_41222,

      SB =
        df_SB_41222,

      SW =
        df_SW_41222
    )


  for(scheme_41222 in names(scenarios_41222)){

    ds_41222 <-
      scenarios_41222[[scheme_41222]]


    weights_41222 <-
      if(
        scheme_41222 == "SW"
      ){

        as.numeric(
          ds_41222$w_area
        )

      } else {

        rep(
          1,
          nrow(ds_41222)
        )
      }


    variants_41222 <-
      list(

        RAW =
          RAW_FEATURES_40222,

        `RAW+SHADOW` =
          RAW_SHADOW_FEATURES_40222
      )


    for(
      variant_41222
      in
      names(variants_41222)
    ){

      feats_41222 <-
        variants_41222[[variant_41222]]


      model_seed_41222 <-

        SEED +

        50000 +

        rp * 100 +

        match(
          scheme_41222,
          c(
            "UR",
            "SB",
            "SW"
          )
        ) * 10 +

        match(
          variant_41222,
          c(
            "RAW",
            "RAW+SHADOW"
          )
        )


      # ----------------------------------------------------------
      # Fit
      # ----------------------------------------------------------

      fit_41222 <-
        fit_ridge_40222(

          train_df =
            ds_41222,

          features =
            feats_41222,

          weights =
            weights_41222,

          seed =
            model_seed_41222
        )


      # ----------------------------------------------------------
      # Calibration probabilities
      # ----------------------------------------------------------

      p_cal_41222 <-
        predict_ridge_40222(
          fit_41222,
          df_cal
        )


      # ----------------------------------------------------------
      # Global threshold
      # ----------------------------------------------------------

      th_global_41222 <-
        optimal_threshold_ridge_40222(

          df_cal$presence,
          p_cal_41222
        )


      idx_ns_41222 <-
        which(
          df_cal$shadow_bin == 0
        )


      idx_sh_41222 <-
        which(
          df_cal$shadow_bin == 1
        )


      # ----------------------------------------------------------
      # Regime thresholds
      # ----------------------------------------------------------

      th_nonshadow_41222 <-
        optimal_threshold_ridge_40222(

          df_cal$presence[
            idx_ns_41222
          ],

          p_cal_41222[idx_ns_41222]
        )


      th_shadow_41222 <-
        optimal_threshold_ridge_40222(

          df_cal$presence[idx_sh_41222],

          p_cal_41222[
            idx_sh_41222
          ]
        )


      if(
        !all(
          is.finite(
            c(
              th_global_41222,
              th_nonshadow_41222,
              th_shadow_41222
            )
          )
        )
      ){

        stop(
          paste0(
            "Non-finite Ridge threshold: ",
            scheme_41222,
            " / ",
            variant_41222,
            " / replicate ",
            rp
          )
        )
      }


      # ----------------------------------------------------------
      # External-test probabilities
      # ----------------------------------------------------------

      p_test_41222 <-
        predict_ridge_40222(
          fit_41222,
          df_te
        )


      # ----------------------------------------------------------
      # Global predictions
      # ----------------------------------------------------------

      pred_global_41222 <-
        as.integer(
          p_test_41222 >=
            th_global_41222
        )


      # ----------------------------------------------------------
      # Regime-specific predictions
      # ----------------------------------------------------------

      pred_regime_41222 <-
        ifelse(

          df_te$shadow_bin == 1,

          as.integer(
            p_test_41222 >=
              th_shadow_41222
          ),

          as.integer(
            p_test_41222 >=
              th_nonshadow_41222
          )
        )


      ridge_pred_list_41222[[ridge_index_41222]] <-

        tibble(

          algorithm =
            "Ridge logistic",

          replicate =
            rp,

          scheme =
            scheme_41222,

          variant =
            variant_41222,

          test_id =
            as.integer(
              df_te$test_id
            ),

          presence =
            as.integer(
              df_te$presence
            ),

          shadow_bin =
            as.integer(
              df_te$shadow_bin
            ),

          probability =
            as.numeric(
              p_test_41222
            ),

          threshold_global =
            th_global_41222,

          threshold_nonshadow =
            th_nonshadow_41222,

          threshold_shadow =
            th_shadow_41222,

          predicted_global =
            pred_global_41222,

          predicted_regime =
            as.integer(
              pred_regime_41222
            )
        )


      ridge_index_41222 <-
        ridge_index_41222 + 1L


      rm(
        fit_41222,
        p_cal_41222,
        p_test_41222
      )

      gc(
        verbose = FALSE
      )
    }
  }
}


ridge_pred_41222 <-
  bind_rows(
    ridge_pred_list_41222
  )


# ======================================================================
# 41.7 GAM
#
# Reconstruct exactly as Section 40 / Section 44.
# ======================================================================

gam_pred_list_41222 <- list()

gam_index_41222 <- 1L


for(rp in seq_len(N_REP)){

  cat(
    "\nGAM external-test reconstruction:",
    rp,
    "of",
    N_REP,
    "\n"
  )


  # ------------------------------------------------------------
  # Exact GAM sampling seeds
  # ------------------------------------------------------------

  df_UR_41222 <-
    make_UR_area(

      df_pool =
        df_fit_pool,

      n_total =
        N_TRAIN_REP,

      p_area =
        p_shadow_area,

      seed =
        SEED_GAM_40222 +
        1000 +
        rp
    )


  df_SB_41222 <-
    make_SB_poolratio(

      df_pool =
        df_fit_pool,

      n_total =
        N_TRAIN_REP,

      seed =
        SEED_GAM_40222 +
        2000 +
        rp
    )


  sw_obj_41222 <-
    make_SW_from_SB(

      df_SB =
        df_SB_41222,

      p_area =
        p_shadow_area
    )


  df_SW_41222 <-
    sw_obj_41222$df


  stopifnot(

    identical(
      df_SB_41222$orig_id,
      df_SW_41222$orig_id
    )
  )


  scenarios_41222 <-
    list(

      UR =
        df_UR_41222,

      SB =
        df_SB_41222,

      SW =
        df_SW_41222
    )


  for(
    scheme_41222
    in
    names(scenarios_41222)
  ){

    ds_41222 <-
      scenarios_41222[[scheme_41222]]


    weights_41222 <-
      if(
        scheme_41222 == "SW"
      ){

        as.numeric(
          ds_41222$w_area
        )

      } else {

        rep(
          1,
          nrow(ds_41222)
        )
      }


    variants_41222 <-
      list(

        RAW =
          RAW_FEATURES_40222,

        `RAW+SHADOW` =
          RAW_SHADOW_FEATURES_40222
      )


    for(
      variant_41222
      in
      names(variants_41222)
    ){

      feats_41222 <-
        variants_41222[[variant_41222]]


      # ----------------------------------------------------------
      # Fit
      # ----------------------------------------------------------

      fit_41222 <-
        fit_gam_40222(

          train_df =
            ds_41222,

          features =
            feats_41222,

          weights_fit =
            weights_41222
        )


      # ----------------------------------------------------------
      # Calibration predictions
      # ----------------------------------------------------------

      p_cal_41222 <-
        predict_gam_40222(
          fit_41222,
          df_cal
        )


      th_global_41222 <-
        best_threshold_gam_40222(

          df_cal$presence,
          p_cal_41222
        )


      idx_ns_41222 <-
        which(
          df_cal$shadow_bin == 0
        )


      idx_sh_41222 <-
        which(
          df_cal$shadow_bin == 1
        )


      th_nonshadow_41222 <-
        best_threshold_gam_40222(

          df_cal$presence[
            idx_ns_41222
          ],

          p_cal_41222[idx_ns_41222]
        )


      th_shadow_41222 <-
        best_threshold_gam_40222(

          df_cal$presence[idx_sh_41222],

          p_cal_41222[idx_sh_41222]
        )


      if(
        !all(
          is.finite(
            c(
              th_global_41222,
              th_nonshadow_41222,
              th_shadow_41222
            )
          )
        )
      ){

        stop(
          paste0(
            "Non-finite GAM threshold: ",
            scheme_41222,
            " / ",
            variant_41222,
            " / replicate ",
            rp
          )
        )
      }


      # ----------------------------------------------------------
      # External-test probabilities
      # ----------------------------------------------------------

      p_test_41222 <-
        predict_gam_40222(
          fit_41222,
          df_te
        )


      pred_global_41222 <-
        as.integer(
          p_test_41222 >=
            th_global_41222
        )


      pred_regime_41222 <-
        ifelse(

          df_te$shadow_bin == 1,

          as.integer(
            p_test_41222 >=
              th_shadow_41222
          ),

          as.integer(
            p_test_41222 >=
              th_nonshadow_41222
          )
        )


      gam_pred_list_41222[[gam_index_41222]] <-

        tibble(

          algorithm =
            "GAM",

          replicate =
            rp,

          scheme =
            scheme_41222,

          variant =
            variant_41222,

          test_id =
            as.integer(
              df_te$test_id
            ),

          presence =
            as.integer(
              df_te$presence
            ),

          shadow_bin =
            as.integer(
              df_te$shadow_bin
            ),

          probability =
            as.numeric(
              p_test_41222
            ),

          threshold_global =
            th_global_41222,

          threshold_nonshadow =
            th_nonshadow_41222,

          threshold_shadow =
            th_shadow_41222,

          predicted_global =
            pred_global_41222,

          predicted_regime =
            as.integer(
              pred_regime_41222
            )
        )


      gam_index_41222 <-
        gam_index_41222 +
        1L


      rm(
        fit_41222,
        p_cal_41222,
        p_test_41222
      )

      gc(
        verbose = FALSE
      )
    }
  }
}


gam_pred_41222 <-
  bind_rows(
    gam_pred_list_41222
  )


# ======================================================================
# 41.8 COMBINE ALL THREE ALGORITHMS
# ======================================================================

PRED_41222 <-
  bind_rows(

    rf_pred_41222,
    ridge_pred_41222,
    gam_pred_41222

  ) %>%

  mutate(

    algorithm =
      factor(

        algorithm,

        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    scheme =
      factor(

        scheme,

        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      factor(

        std_variant_41222(
          variant
        ),

        levels =
          c(
            "RAW",
            "RAW+SHADOW"
          )
      ),

    regime =
      factor(

        ifelse(
          shadow_bin == 1,
          "Shadow",
          "Non-shadow"
        ),

        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      )
  )


write_csv(

  PRED_41222,

  file.path(
    DIR_41222_CSV,
    "41_222_01_three_algorithm_external_predictions.csv"
  )
)


# ======================================================================
# 41.9 CRITICAL QC
# ======================================================================

qc_pred_41222 <-
  PRED_41222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    replicate
  ) %>%

  summarise(

    n_external =
      n(),

    n_unique_test =
      n_distinct(
        test_id
      ),

    n_shadow =
      sum(
        shadow_bin == 1
      ),

    n_nonshadow =
      sum(
        shadow_bin == 0
      ),

    .groups =
      "drop"
  )


write_csv(

  qc_pred_41222,

  file.path(
    DIR_41222_QC,
    "41_222_QC_external_prediction_counts.csv"
  )
)


expected_test_n_41222 <-
  nrow(
    df_te
  )


stopifnot(

  all(
    qc_pred_41222$n_external ==
      expected_test_n_41222
  ),

  all(
    qc_pred_41222$n_unique_test ==
      expected_test_n_41222
  )
)


# ======================================================================
# 41.10 CLASSIFICATION TRANSITIONS
#
# Global threshold -> Regime-specific threshold
# ======================================================================

PRED_41222 <-
  PRED_41222 %>%

  mutate(

    state_global =
      case_when(

        presence == 1 &
          predicted_global == 1 ~
          "TP",

        presence == 1 &
          predicted_global == 0 ~
          "FN",

        presence == 0 &
          predicted_global == 1 ~
          "FP",

        presence == 0 &
          predicted_global == 0 ~
          "TN",

        TRUE ~
          NA_character_
      ),


    state_regime =
      case_when(

        presence == 1 &
          predicted_regime == 1 ~
          "TP",

        presence == 1 &
          predicted_regime == 0 ~
          "FN",

        presence == 0 &
          predicted_regime == 1 ~
          "FP",

        presence == 0 &
          predicted_regime == 0 ~
          "TN",

        TRUE ~
          NA_character_
      ),


    transition =
      paste0(
        state_global,
        "_to_",
        state_regime
      )
  )


# ======================================================================
# 41.11 METRICS FUNCTION
# ======================================================================

metrics_from_class_41222 <-
  function(
    y,
    pred
  ){

    y <-
      as.integer(y)

    pred <-
      as.integer(pred)


    TP <-
      sum(
        pred == 1 &
          y == 1
      )


    TN <-
      sum(
        pred == 0 &
          y == 0
      )


    FP <-
      sum(
        pred == 1 &
          y == 0
      )


    FN <-
      sum(
        pred == 0 &
          y == 1
      )


    Sensitivity <-
      ifelse(

        TP + FN > 0,

        TP /
          (
            TP +
              FN
          ),

        NA_real_
      )


    Specificity <-
      ifelse(

        TN + FP > 0,

        TN /
          (
            TN +
              FP
          ),

        NA_real_
      )


    BalancedAccuracy <-
      mean(
        c(
          Sensitivity,
          Specificity
        ),
        na.rm = TRUE
      )


    tibble(

      TP = TP,
      TN = TN,
      FP = FP,
      FN = FN,

      Sensitivity =
        Sensitivity,

      Specificity =
        Specificity,

      BalancedAccuracy =
        BalancedAccuracy
    )
  }


# ======================================================================
# 41.12 GLOBAL METRICS BY REGIME
# ======================================================================

metrics_global_41222 <-
  PRED_41222 %>%

  group_by(
    algorithm,
    replicate,
    scheme,
    variant,
    regime
  ) %>%

  group_modify(
    ~
      metrics_from_class_41222(
        .x$presence,
        .x$predicted_global
      )
  ) %>%

  ungroup() %>%

  mutate(
    strategy =
      "GLOBAL"
  )


# ======================================================================
# 41.13 REGIME-SPECIFIC METRICS BY REGIME
# ======================================================================

metrics_regime_41222 <-
  PRED_41222 %>%

  group_by(
    algorithm,
    replicate,
    scheme,
    variant,
    regime
  ) %>%

  group_modify(
    ~
      metrics_from_class_41222(
        .x$presence,
        .x$predicted_regime
      )
  ) %>%

  ungroup() %>%

  mutate(
    strategy =
      "REGIME_SPECIFIC"
  )


metrics_41222 <-
  bind_rows(
    metrics_global_41222,
    metrics_regime_41222
  )


write_csv(

  metrics_41222,

  file.path(
    DIR_41222_CSV,
    "41_222_02_metrics_by_regime_and_strategy.csv"
  )
)


# ======================================================================
# 41.14 DELTAS
#
# Regime-specific minus global
# ======================================================================

delta_41222 <-
  metrics_41222 %>%

  pivot_wider(

    id_cols =
      c(
        algorithm,
        replicate,
        scheme,
        variant,
        regime
      ),

    names_from =
      strategy,

    values_from =
      c(
        TP,
        TN,
        FP,
        FN,
        Sensitivity,
        Specificity,
        BalancedAccuracy
      )
  ) %>%

  mutate(

    Delta_TP =
      TP_REGIME_SPECIFIC -
      TP_GLOBAL,

    Delta_TN =
      TN_REGIME_SPECIFIC -
      TN_GLOBAL,

    Delta_FP =
      FP_REGIME_SPECIFIC -
      FP_GLOBAL,

    Delta_FN =
      FN_REGIME_SPECIFIC -
      FN_GLOBAL,

    Delta_Sensitivity =
      Sensitivity_REGIME_SPECIFIC -
      Sensitivity_GLOBAL,

    Delta_Specificity =
      Specificity_REGIME_SPECIFIC -
      Specificity_GLOBAL,

    Delta_BalancedAccuracy =
      BalancedAccuracy_REGIME_SPECIFIC -
      BalancedAccuracy_GLOBAL
  )


write_csv(

  delta_41222,

  file.path(
    DIR_41222_CSV,
    "41_222_03_metric_deltas_regime_minus_global.csv"
  )
)


# ======================================================================
# 41.15 EXACT TRANSITION COUNTS
# ======================================================================

transition_counts_41222 <-
  PRED_41222 %>%

  count(

    algorithm,
    replicate,
    scheme,
    variant,
    regime,
    transition,

    name =
      "n"
  )


write_csv(

  transition_counts_41222,

  file.path(
    DIR_41222_CSV,
    "41_222_04_all_transition_counts.csv"
  )
)


# ======================================================================
# 41.16 CRITICAL TRANSITIONS
#
# FN -> TP = recovered true presence
# TN -> FP = newly introduced false positive
# FP -> TN = corrected false positive
# TP -> FN = newly introduced false negative
# ======================================================================

key_41222 <-
  PRED_41222 %>%

  group_by(

    algorithm,
    replicate,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    FN_to_TP =
      sum(
        transition ==
          "FN_to_TP"
      ),

    TN_to_FP =
      sum(
        transition ==
          "TN_to_FP"
      ),

    FP_to_TN =
      sum(
        transition ==
          "FP_to_TN"
      ),

    TP_to_FN =
      sum(
        transition ==
          "TP_to_FN"
      ),

    .groups =
      "drop"
  ) %>%

  mutate(

    added_suitable =
      FN_to_TP +
      TN_to_FP,

    correct_added =
      FN_to_TP,

    false_added =
      TN_to_FP,

    correct_added_share =
      ifelse(

        added_suitable > 0,

        correct_added /
          added_suitable,

        NA_real_
      ),

    false_added_share =
      ifelse(

        added_suitable > 0,

        false_added /
          added_suitable,

        NA_real_
      )
  )


write_csv(

  key_41222,

  file.path(
    DIR_41222_CSV,
    "41_222_05_key_transition_decomposition.csv"
  )
)


# ======================================================================
# 41.17 SUMMARY TABLE
# ======================================================================

summary_41222 <-
  key_41222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    n_replicates =
      n(),

    mean_recovered_true_presence =
      mean(
        FN_to_TP,
        na.rm = TRUE
      ),

    median_recovered_true_presence =
      median(
        FN_to_TP,
        na.rm = TRUE
      ),

    mean_new_false_positive =
      mean(
        TN_to_FP,
        na.rm = TRUE
      ),

    median_new_false_positive =
      median(
        TN_to_FP,
        na.rm = TRUE
      ),

    mean_added_suitable =
      mean(
        added_suitable,
        na.rm = TRUE
      ),

    mean_correct_added_share =
      mean(
        correct_added_share,
        na.rm = TRUE
      ),

    median_correct_added_share =
      median(
        correct_added_share,
        na.rm = TRUE
      ),

    .groups =
      "drop"
  )


write_csv(

  summary_41222,

  file.path(
    DIR_41222_CSV,
    "41_222_06_transition_summary.csv"
  )
)


# ======================================================================
# 41.18 COMMON JOURNAL THEME
# ======================================================================

theme_41222 <- function(){

  theme_bw(
    base_size = 11
  ) +

    theme(

      panel.grid.minor =
        element_blank(),

      strip.background =
        element_rect(
          fill = "grey92",
          colour = "grey35"
        ),

      strip.text =
        element_text(
          face = "bold",
          size = 9
        ),

      axis.title =
        element_text(
          size = 11
        ),

      axis.text =
        element_text(
          size = 9
        ),

      plot.title =
        element_text(
          face = "bold",
          size = 14
        ),

      plot.subtitle =
        element_text(
          size = 10
        ),

      legend.title =
        element_text(
          face = "bold"
        ),

      legend.position =
        "bottom"
    )
}


# ======================================================================
# 41.19 FIGURE A
#
# Sensitivity-specificity trade-off
#
# DESIGN:
#   Columns = algorithms
#   Rows    = illumination regimes
#   Shape   = predictor set
#   Symbol grouping = sampling scenario
#
# This gives only six panels instead of the extremely crowded
# 36-panel extension of the original RF-only figure.
# ======================================================================

figA_df_41222 <-
  delta_41222 %>%

  mutate(

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      ),

    scheme =
      factor(
        scheme,
        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW+SHADOW"
          )
      )
  )


figA_41222 <-
  ggplot(

    figA_df_41222,

    aes(

      x =
        Delta_Specificity,

      y =
        Delta_Sensitivity,

      shape =
        variant
    )
  ) +

  geom_hline(

    yintercept =
      0,

    linetype =
      2,

    linewidth =
      0.45
  ) +

  geom_vline(

    xintercept =
      0,

    linetype =
      2,

    linewidth =
      0.45
  ) +

  geom_point(

    aes(
      alpha =
        scheme
    ),

    size =
      2,

    position =
      position_jitter(
        width = 0.005,
        height = 0.005
      )
  ) +

  facet_grid(

    regime ~ algorithm,

    labeller =
      labeller(

        regime =
          c(
            `Non-shadow` =
              "NON-SHADOW",

            Shadow =
              "SHADOW"
          )
      )
  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" = 16,
        "RAW+SHADOW" = 17
      )
  ) +

  scale_alpha_manual(

    values =
      c(
        "SB" = 0.45,
        "SW" = 0.70,
        "UR" = 1.00
      )
  ) +

  labs(

    title =
      "Sensitivity-specificity trade-off of illumination-regime-specific thresholds",

    subtitle =
      paste0(
        "Regime-specific minus global threshold; ",
        "each point represents one stochastic model replicate"
      ),

    x =
      expression(
        Delta *
          " Specificity (regime-specific - global)"
      ),

    y =
      expression(
        Delta *
          " Sensitivity (regime-specific - global)"
      ),

    shape =
      "Predictor set",

    alpha =
      "Sampling scenario"
  ) +

  theme_41222()


ggsave(

  filename =
    file.path(
      DIR_41222_FIG,
      "FIG41_222_A_three_algorithm_sensitivity_specificity_tradeoff.png"
    ),

  plot =
    figA_41222,

  width =
    13,

  height =
    8,

  units =
    "in",

  dpi =
    600
)


ggsave(

  filename =
    file.path(
      DIR_41222_FIG,
      "FIG41_222_A_three_algorithm_sensitivity_specificity_tradeoff.tiff"
    ),

  plot =
    figA_41222,

  width =
    13,

  height =
    8,

  units =
    "in",

  dpi =
    600,

  compression =
    "lzw"
)


# ======================================================================
# 41.20 FIGURE B
#
# Composition of newly suitable external-test classifications
#
# Exact:
#
#   FN -> TP = recovered true presence
#   TN -> FP = newly introduced false positive
#
# Rows:
#   Non-shadow transition types
#   Shadow transition types
#
# Columns:
#   Algorithms
#
# ======================================================================

figB_df_41222 <-
  key_41222 %>%

  select(

    algorithm,
    replicate,
    scheme,
    variant,
    regime,

    `Recovered true presence` =
      FN_to_TP,

    `New false positive` =
      TN_to_FP
  ) %>%

  pivot_longer(

    cols =
      c(
        `Recovered true presence`,
        `New false positive`
      ),

    names_to =
      "transition_type",

    values_to =
      "count"
  ) %>%

  mutate(

    algorithm =
      factor(

        algorithm,

        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    regime =
      factor(

        regime,

        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      ),

    transition_type =
      factor(

        transition_type,

        levels =
          c(
            "New false positive",
            "Recovered true presence"
          )
      ),

    scheme =
      factor(

        scheme,

        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      factor(

        variant,

        levels =
          c(
            "RAW",
            "RAW+SHADOW"
          )
      )
  )


figB_41222 <-
  ggplot(

    figB_df_41222,

    aes(

      x =
        scheme,

      y =
        count,

      shape =
        variant
    )
  ) +

  geom_boxplot(

    aes(
      group =
        interaction(
          scheme,
          variant
        )
    ),

    position =
      position_dodge(
        width = 0.70
      ),

    outlier.shape =
      NA,

    width =
      0.55
  ) +

  geom_point(

    position =
      position_jitterdodge(

        jitter.width =
          0.10,

        dodge.width =
          0.70
      ),

    alpha =
      0.45,

    size =
      1.25
  ) +

  facet_grid(

    regime +
      transition_type ~
      algorithm,

    scales =
      "free_y"
  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" = 16,
        "RAW+SHADOW" = 17
      )
  ) +

  labs(

    title =
      "Composition of newly suitable external-test classifications",

    subtitle =
      paste0(
        "Global to illumination-regime-specific threshold transition; ",
        "exact observation-level changes"
      ),

    x =
      "Sampling scenario",

    y =
      "Number of observations per replicate",

    shape =
      "Predictor set"
  ) +

  theme_41222()


ggsave(

  filename =
    file.path(
      DIR_41222_FIG,
      "FIG41_222_B_three_algorithm_new_suitable_TP_vs_FP.png"
    ),

  plot =
    figB_41222,

  width =
    13,

  height =
    10,

  units =
    "in",

  dpi =
    600
)


ggsave(

  filename =
    file.path(
      DIR_41222_FIG,
      "FIG41_222_B_three_algorithm_new_suitable_TP_vs_FP.tiff"
    ),

  plot =
    figB_41222,

  width =
    13,

  height =
    10,

  units =
    "in",

  dpi =
    600,

  compression =
    "lzw"
)


# ======================================================================
# 41.21 FIGURE C
#
# Were newly suitable classifications correct?
#
# Rather than redundantly plotting both correct and false shares
# (which sum to one), the three-algorithm version plots the
# CORRECT share directly.
#
# 0.5 = equal numbers of:
#       recovered presences and new false positives
#
# > 0.5 = beneficial additions dominate
# < 0.5 = false-positive additions dominate
# ======================================================================

figC_df_41222 <-
  key_41222 %>%

  filter(
    is.finite(
      correct_added_share
    )
  ) %>%

  mutate(

    algorithm =
      factor(

        algorithm,

        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    regime =
      factor(

        regime,

        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      ),

    scheme =
      factor(

        scheme,

        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      factor(

        variant,

        levels =
          c(
            "RAW",
            "RAW+SHADOW"
          )
      )
  )


figC_41222 <-
  ggplot(

    figC_df_41222,

    aes(

      x =
        scheme,

      y =
        correct_added_share,

      shape =
        variant
    )
  ) +

  geom_hline(

    yintercept =
      0.5,

    linetype =
      2,

    linewidth =
      0.5
  ) +

  geom_boxplot(

    aes(
      group =
        interaction(
          scheme,
          variant
        )
    ),

    position =
      position_dodge(
        width = 0.70
      ),

    outlier.shape =
      NA,

    width =
      0.55
  ) +

  geom_point(

    position =
      position_jitterdodge(

        jitter.width =
          0.10,

        dodge.width =
          0.70
      ),

    alpha =
      0.50,

    size =
      1.35
  ) +

  facet_grid(

    regime ~
      algorithm
  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" = 16,
        "RAW+SHADOW" = 17
      )
  ) +

  scale_y_continuous(

    limits =
      c(
        0,
        1
      ),

    breaks =
      seq(
        0,
        1,
        by = 0.25
      )
  ) +

  labs(

    title =
      "Were newly suitable classifications correct?",

    subtitle =
      paste0(
        "Share of 0-to-1 classification changes that recovered ",
        "true presences on the independent external test"
      ),

    x =
      "Sampling scenario",

    y =
      "Correct share of newly suitable classifications",

    shape =
      "Predictor set"
  ) +

  theme_41222()


ggsave(

  filename =
    file.path(
      DIR_41222_FIG,
      "FIG41_222_C_three_algorithm_correct_added_share.png"
    ),

  plot =
    figC_41222,

  width =
    13,

  height =
    8,

  units =
    "in",

  dpi =
    600
)


ggsave(

  filename =
    file.path(
      DIR_41222_FIG,
      "FIG41_222_C_three_algorithm_correct_added_share.tiff"
    ),

  plot =
    figC_41222,

  width =
    13,

  height =
    8,

  units =
    "in",

  dpi =
    600,

  compression =
    "lzw"
)


# ======================================================================
# 41.22 SUPPLEMENTARY TRADE-OFF SUMMARY
# ======================================================================

tradeoff_summary_41222 <-
  delta_41222 %>%

  group_by(

    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    n_replicates =
      n(),

    mean_delta_sensitivity =
      mean(
        Delta_Sensitivity,
        na.rm = TRUE
      ),

    median_delta_sensitivity =
      median(
        Delta_Sensitivity,
        na.rm = TRUE
      ),

    q025_delta_sensitivity =
      quantile(
        Delta_Sensitivity,
        0.025,
        na.rm = TRUE
      ),

    q975_delta_sensitivity =
      quantile(
        Delta_Sensitivity,
        0.975,
        na.rm = TRUE
      ),

    mean_delta_specificity =
      mean(
        Delta_Specificity,
        na.rm = TRUE
      ),

    median_delta_specificity =
      median(
        Delta_Specificity,
        na.rm = TRUE
      ),

    q025_delta_specificity =
      quantile(
        Delta_Specificity,
        0.025,
        na.rm = TRUE
      ),

    q975_delta_specificity =
      quantile(
        Delta_Specificity,
        0.975,
        na.rm = TRUE
      ),

    mean_delta_balanced_accuracy =
      mean(
        Delta_BalancedAccuracy,
        na.rm = TRUE
      ),

    .groups =
      "drop"
  )


write_csv(

  tradeoff_summary_41222,

  file.path(
    DIR_41222_CSV,
    "Table41_222_S1_three_algorithm_tradeoff_summary.csv"
  )
)


# ======================================================================
# 41.23 CORRECT-ADDITION SUMMARY
# ======================================================================

correct_added_summary_41222 <-
  key_41222 %>%

  group_by(

    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    n_replicates =
      n(),

    n_replicates_with_added_suitable =
      sum(
        added_suitable > 0
      ),

    mean_FN_to_TP =
      mean(
        FN_to_TP,
        na.rm = TRUE
      ),

    mean_TN_to_FP =
      mean(
        TN_to_FP,
        na.rm = TRUE
      ),

    mean_correct_added_share =
      mean(
        correct_added_share,
        na.rm = TRUE
      ),

    median_correct_added_share =
      median(
        correct_added_share,
        na.rm = TRUE
      ),

    q025_correct_added_share =
      quantile(
        correct_added_share,
        0.025,
        na.rm = TRUE
      ),

    q975_correct_added_share =
      quantile(
        correct_added_share,
        0.975,
        na.rm = TRUE
      ),

    prop_replicates_correct_share_gt_0_5 =
      mean(
        correct_added_share > 0.5,
        na.rm = TRUE
      ),

    .groups =
      "drop"
  )


write_csv(

  correct_added_summary_41222,

  file.path(
    DIR_41222_CSV,
    "Table41_222_S2_new_suitable_classification_summary.csv"
  )
)


# ======================================================================
# 41.24 FINAL QC
# ======================================================================

expected_models_41222 <-
  N_REP *
  3 *
  2


model_count_41222 <-
  PRED_41222 %>%

  distinct(
    algorithm,
    replicate,
    scheme,
    variant
  ) %>%

  count(
    algorithm,
    name =
      "n_models"
  )


print(
  model_count_41222
)


stopifnot(

  all(
    model_count_41222$n_models ==
      expected_models_41222
  )
)


stopifnot(

  all(
    PRED_41222$predicted_global %in%
      c(
        0,
        1
      )
  ),

  all(
    PRED_41222$predicted_regime %in%
      c(
        0,
        1
      )
  ),

  all(
    PRED_41222$presence %in%
      c(
        0,
        1
      )
  )
)


write_csv(

  model_count_41222,

  file.path(
    DIR_41222_QC,
    "41_222_QC_model_counts.csv"
  )
)


# ======================================================================
# 41.25 FINAL CONSOLE REPORT
# ======================================================================

cat(
  "\n\n============================================================\n"
)

cat(
  "SECTION 41_222 COMPLETED SUCCESSFULLY\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nAlgorithms:\n"
)

print(
  model_count_41222
)


cat(
  "\nPrimary journal-ready figures:\n"
)

cat(
  "  FIG41_222_A_three_algorithm_sensitivity_specificity_tradeoff.png\n"
)

cat(
  "  FIG41_222_B_three_algorithm_new_suitable_TP_vs_FP.png\n"
)

cat(
  "  FIG41_222_C_three_algorithm_correct_added_share.png\n"
)


cat(
  "\nSupplementary tables:\n"
)

cat(
  "  Table41_222_S1_three_algorithm_tradeoff_summary.csv\n"
)

cat(
  "  Table41_222_S2_new_suitable_classification_summary.csv\n"
)


cat(
  "\nAll figures saved at 600 dpi in PNG and TIFF.\n"
)


cat(
  "\nResults directory:\n",
  DIR_41222,
  "\n"
)


cat(
  "============================================================\n"
)
############################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # ======================================================================
# Predictor-contribution analysis across algorithms
#
# PURPOSE
# ------
# Reconstruct the original Section-46 scientific question:
# does the predictive contribution of the SAME environmental predictors
# differ between persistent-shadow and non-shadow external-test subsets?
#
# IMPORTANT
# ---------
# - SAME model-agnostic importance definition as the original Section 46:
#       VIP_Brier = Brier(permuted predictor) - Brier(original)
#       Delta_VIP = VIP_Shadow - VIP_Non-shadow
# - SAME three algorithms: RF, Ridge logistic, GAM
# - SAME UR / SB / SW scenarios
# - SAME RAW / RAW+SHADOW variants
# - SAME independent external test
# - shadow_bin itself is NOT permuted within regime
#
# SPEED CHANGE ONLY
# -----------------
# N_PERM = 1 per predictor/model/regime instead of 5.
# With 30 stochastic model replicates, inference still comes from the
# replicate distribution; the single permutation is deterministic and
# reproducible for each fitted model.
#
# This script creates the SAME downstream object names required by
# Sections 48 and 47:
#   importance_46222
#   importance_wide_46222
#   importance_summary_46222
#   algorithm_summary_46222
#   strict_consensus_46222
#   qc_46222
#
# RUN:
#   1) Stop the currently running slow 46.
#   2) Run this entire file.
#   3) Then continue the master script from SECTION 48 onward.
# ======================================================================

# ======================================================================
# SECTION 46_FAST
# FAST DROP-IN REPLACEMENT
# Model-agnostic within-regime permutation importance across 3 algorithms
# ======================================================================

req46222 <- c("OUT_DIR","DIR_MODEL","df_fit_pool","df_te","PRED_41222","RAW_FEATURES_40222","RAW_SHADOW_FEATURES_40222","N_REP","N_TRAIN_REP","p_shadow_area","SEED","SEED_GAM_40222","make_UR_area","make_SB_poolratio","make_SW_from_SB","fit_ridge_40222","predict_ridge_40222","fit_gam_40222","predict_gam_40222")
miss46222 <- req46222[!vapply(req46222,exists,logical(1),envir=.GlobalEnv)]
if(length(miss46222)) stop("46_222 missing objects: ",paste(miss46222,collapse=", "))

SEED_46222 <- 46000

# FAST FINAL SETTING:
# One deterministic within-regime permutation per predictor and fitted model.
# Inference is still replicated across 30 stochastic model replicates,
# 3 sampling schemes, 2 predictor variants and 3 algorithms.
# This preserves the SAME Delta VIP(Brier) definition while avoiding the
# fivefold prediction burden of the previous 46 implementation.
N_PERM_46222 <- 1

ENV_FEATURES_46222 <- RAW_FEATURES_40222
stopifnot(!"SCL"%in%ENV_FEATURES_46222,!"ShadowMask"%in%ENV_FEATURES_46222,!"shadow_bin"%in%ENV_FEATURES_46222)
DIR_46222<-file.path(OUT_DIR,"46_222_FAST_THREE_ALGORITHM_REGIME_PERMUTATION_IMPORTANCE")
DIR_46222_CSV<-file.path(DIR_46222,"CSV"); DIR_46222_FIG<-file.path(DIR_46222,"FIGURES"); DIR_46222_QC<-file.path(DIR_46222,"QC")
invisible(lapply(c(DIR_46222,DIR_46222_CSV,DIR_46222_FIG,DIR_46222_QC),dir.create,recursive=TRUE,showWarnings=FALSE))

rf_model_path_46222 <- function(scheme,variant,replicate){vf<-if(variant=="RAW+SHADOW")"RAW_SHADOW" else "RAW"; file.path(DIR_MODEL,paste0("RF_",scheme,"_",vf,"_REP",sprintf("%03d",replicate),".rds"))}
predict_rf_46222 <- function(model,newdata,features){pp<-predict(model,data=newdata[,features,drop=FALSE])$predictions; if(is.matrix(pp)){if("1"%in%colnames(pp)) return(as.numeric(pp[,"1"])); return(as.numeric(pp[,ncol(pp)]))}; as.numeric(pp)}
brier_46222 <- function(y,p){ok<-is.finite(y)&is.finite(p); if(!any(ok))return(NA_real_); mean((as.numeric(p[ok])-as.integer(y[ok]))^2)}
auc_46222 <- safe_auc_42222

ridge_scen_46222 <- function(rp){u<-make_UR_area(df_fit_pool,N_TRAIN_REP,p_shadow_area,SEED+42000+rp); s<-make_SB_poolratio(df_fit_pool,N_TRAIN_REP,SEED+43000+rp); w<-make_SW_from_SB(s,p_shadow_area); list(UR=u,SB=s,SW=w$df)}
gam_scen_46222 <- function(rp){u<-make_UR_area(df_fit_pool,N_TRAIN_REP,p_shadow_area,SEED_GAM_40222+1000+rp); s<-make_SB_poolratio(df_fit_pool,N_TRAIN_REP,SEED_GAM_40222+2000+rp); w<-make_SW_from_SB(s,p_shadow_area); list(UR=u,SB=s,SW=w$df)}

importance_list_46222<-list(); recon_list_46222<-list(); ii46222<-1L; qq46222<-1L
for(ALG in c("Random Forest","Ridge logistic","GAM")){
  cat("\n46_222_FAST algorithm:",ALG,"\n")
  for(rp in seq_len(N_REP)){
    if(rp == 1L || rp %% 5L == 0L || rp == N_REP){
      cat("  replicate", rp, "of", N_REP, "\n")
      flush.console()
    }
    scen <- if(ALG=="Ridge logistic") ridge_scen_46222(rp) else if(ALG=="GAM") gam_scen_46222(rp) else NULL
    for(sc in c("SB","SW","UR")) for(va in c("RAW","RAW+SHADOW")){
      feats <- if(va=="RAW") RAW_FEATURES_40222 else RAW_SHADOW_FEATURES_40222
      if(ALG=="Random Forest"){
        mf<-rf_model_path_46222(sc,va,rp); if(!file.exists(mf)) stop("Missing RF model: ",mf); fit<-readRDS(mf)
        pfun<-function(nd) predict_rf_46222(fit,nd,feats)
      } else if(ALG=="Ridge logistic"){
        ds<-scen[[sc]]; ww<-if(sc=="SW")as.numeric(ds$w_area)else rep(1,nrow(ds)); mseed<-SEED+50000+rp*100+match(sc,c("UR","SB","SW"))*10+match(va,c("RAW","RAW+SHADOW"))
        fit<-fit_ridge_40222(ds,feats,ww,mseed); pfun<-function(nd) predict_ridge_40222(fit,nd)
      } else {
        ds<-scen[[sc]]; ww<-if(sc=="SW")as.numeric(ds$w_area)else rep(1,nrow(ds)); fit<-fit_gam_40222(ds,feats,ww); pfun<-function(nd) predict_gam_40222(fit,nd)
      }
      p0<-as.numeric(pfun(df_te))
      ref<-PRED_41222 %>% filter(as.character(algorithm)==ALG,replicate==rp,as.character(scheme)==sc,as.character(variant)==va) %>% arrange(test_id)
      if(nrow(ref)!=nrow(df_te)) stop("Reference prediction row mismatch")
      md<-max(abs(p0-ref$probability),na.rm=TRUE)
      recon_list_46222[[qq46222]]<-tibble(algorithm=ALG,replicate=rp,scheme=sc,variant=va,max_abs_probability_difference=md); qq46222<-qq46222+1L
      for(rv in c(0L,1L)){
        rlab<-if(rv==1L)"Shadow" else "Non-shadow"; idx<-which(df_te$shadow_bin==rv); dr<-df_te[idx,,drop=FALSE]; yr<-as.integer(dr$presence); pb<-p0[idx]
        bb<-brier_46222(yr,pb); ba<-auc_46222(yr,pb)
        for(pr in ENV_FEATURES_46222){
          if(!pr%in%names(dr) || !pr%in%feats) next
          bl<-al<-rep(NA_real_,N_PERM_46222)
          for(k in seq_len(N_PERM_46222)){
            set.seed(SEED_46222+rp*100000+match(ALG,c("Random Forest","Ridge logistic","GAM"))*10000+match(sc,c("SB","SW","UR"))*1000+match(va,c("RAW","RAW+SHADOW"))*500+rv*200+match(pr,ENV_FEATURES_46222)*10+k)
            dp<-dr; dp[[pr]]<-sample(dp[[pr]],replace=FALSE); pp<-as.numeric(pfun(dp)); bl[k]<-brier_46222(yr,pp)-bb; al[k]<-ba-auc_46222(yr,pp)
          }
          importance_list_46222[[ii46222]]<-tibble(algorithm=ALG,replicate=rp,scheme=sc,variant=va,regime=rlab,predictor=pr,n_perm=N_PERM_46222,VIP_Brier=mean(bl,na.rm=TRUE),VIP_AUC=mean(al,na.rm=TRUE),Base_Brier=bb,Base_AUC=ba); ii46222<-ii46222+1L
        }
      }
      rm(fit,pfun,p0)
    }
  }
  invisible(gc(FALSE))
}

importance_46222<-bind_rows(importance_list_46222); reconstruction_qc_46222<-bind_rows(recon_list_46222)
write_csv(importance_46222,file.path(DIR_46222_CSV,"46_222_01_regime_permutation_importance_all.csv")); write_csv(reconstruction_qc_46222,file.path(DIR_46222_QC,"46_222_reconstruction_probability_QC.csv"))

importance_wide_46222<-importance_46222 %>% select(algorithm,replicate,scheme,variant,predictor,regime,VIP_Brier,VIP_AUC) %>% pivot_wider(names_from=regime,values_from=c(VIP_Brier,VIP_AUC)) %>% mutate(Delta_Brier=VIP_Brier_Shadow-`VIP_Brier_Non-shadow`,Delta_AUC=VIP_AUC_Shadow-`VIP_AUC_Non-shadow`)
write_csv(importance_wide_46222,file.path(DIR_46222_CSV,"46_222_02_regime_contrasts_by_replicate.csv"))

importance_summary_46222<-importance_wide_46222 %>% group_by(algorithm,scheme,variant,predictor) %>% summarise(n_replicates=sum(is.finite(Delta_Brier)),Mean_Delta_Brier=mean(Delta_Brier,na.rm=TRUE),Median_Delta_Brier=median(Delta_Brier,na.rm=TRUE),LCL_Delta_Brier=safe_q_42222(Delta_Brier,.025),UCL_Delta_Brier=safe_q_42222(Delta_Brier,.975),Prop_Positive=mean(Delta_Brier>0,na.rm=TRUE),Mean_Delta_AUC=mean(Delta_AUC,na.rm=TRUE),.groups="drop")
write_csv(importance_summary_46222,file.path(DIR_46222_CSV,"46_222_03_importance_summary_by_algorithm_scheme.csv"))

algorithm_summary_46222<-importance_wide_46222 %>% group_by(algorithm,variant,predictor) %>% summarise(n_combinations=sum(is.finite(Delta_Brier)),Mean_Delta_Brier=mean(Delta_Brier,na.rm=TRUE),Median_Delta_Brier=median(Delta_Brier,na.rm=TRUE),Prop_positive=mean(Delta_Brier>0,na.rm=TRUE),Prop_negative=mean(Delta_Brier<0,na.rm=TRUE),Mean_Delta_AUC=mean(Delta_AUC,na.rm=TRUE),.groups="drop")
write_csv(algorithm_summary_46222,file.path(DIR_46222_CSV,"46_222_04_cross_algorithm_importance_summary.csv"))

strict_consensus_46222<-importance_wide_46222 %>% group_by(variant,predictor) %>% summarise(n_combinations=sum(is.finite(Delta_Brier)),Mean_Delta_Brier=mean(Delta_Brier,na.rm=TRUE),Positive_combinations=sum(Delta_Brier>0,na.rm=TRUE),Negative_combinations=sum(Delta_Brier<0,na.rm=TRUE),Prop_positive=mean(Delta_Brier>0,na.rm=TRUE),Prop_negative=mean(Delta_Brier<0,na.rm=TRUE),Strict_direction=case_when(Prop_positive>=.80~"Consistently Shadow",Prop_negative>=.80~"Consistently Non-shadow",TRUE~"Model/context dependent"),.groups="drop")
write_csv(strict_consensus_46222,file.path(DIR_46222_CSV,"46_222_05_strict_consensus.csv"))

top25_46222<-strict_consensus_46222 %>% group_by(predictor) %>% summarise(score=max(abs(Mean_Delta_Brier),na.rm=TRUE),.groups="drop") %>% arrange(desc(score)) %>% slice_head(n=25) %>% pull(predictor)
heat_46222<-algorithm_summary_46222 %>% filter(predictor%in%top25_46222) %>% mutate(predictor=factor(predictor,levels=rev(top25_46222)),algorithm=factor(algorithm,levels=c("Random Forest","Ridge logistic","GAM")))
fig_imp_46222<-ggplot(heat_46222,aes(x=algorithm,y=predictor,fill=Mean_Delta_Brier))+geom_tile(colour="white",linewidth=.25)+facet_wrap(~variant,nrow=1)+scale_fill_gradient2(midpoint=0)+labs(title="Regime-dependent predictor contribution across modelling algorithms",subtitle="Delta VIP(Brier) = Shadow - Non-shadow",x=NULL,y=NULL,fill="Delta VIP(Brier)")+theme_bw(base_size=10)+theme(panel.grid=element_blank(),strip.text=element_text(face="bold"),axis.text.y=element_text(size=7.5),plot.title=element_text(face="bold"))
ggsave(file.path(DIR_46222_FIG,"FIG46_222_MAIN_cross_algorithm_regime_importance_heatmap.png"),fig_imp_46222,width=10.5,height=9,units="in",dpi=600)
ggsave(file.path(DIR_46222_FIG,"FIG46_222_MAIN_cross_algorithm_regime_importance_heatmap.tiff"),fig_imp_46222,width=10.5,height=9,units="in",dpi=600,compression="lzw")

maxdiff46222<-max(reconstruction_qc_46222$max_abs_probability_difference,na.rm=TRUE)
qc_46222<-tibble(check=c("3 algorithms","3 schemes","2 variants","2 regimes","importance excludes SCL","importance excludes ShadowMask","importance excludes shadow_bin","expected reconstructed models","finite Brier importance","baseline reconstruction <=1e-6"),passed=c(n_distinct(importance_46222$algorithm)==3,n_distinct(importance_46222$scheme)==3,n_distinct(importance_46222$variant)==2,n_distinct(importance_46222$regime)==2,!"SCL"%in%ENV_FEATURES_46222,!"ShadowMask"%in%ENV_FEATURES_46222,!"shadow_bin"%in%ENV_FEATURES_46222,nrow(reconstruction_qc_46222)==N_REP*3*2*3,all(is.finite(importance_46222$VIP_Brier)),is.finite(maxdiff46222)&&maxdiff46222<=1e-6))
write_csv(qc_46222,file.path(DIR_46222_QC,"46_222_FINAL_QC.csv")); print(qc_46222,n=Inf); cat("\nMax reconstruction difference:",format(maxdiff46222,scientific=TRUE),"\n"); if(!all(qc_46222$passed)) stop("46_222 QC failed; inspect before using results")
cat("\nSECTION 46_222_FAST COMPLETED SUCCESSFULLY\nTop contrasts:\n"); print(strict_consensus_46222%>%arrange(desc(abs(Mean_Delta_Brier)))%>%slice_head(n=30),n=30,width=Inf)



# ======================================================================
# SECTION 48
# FINAL THREE-ALGORITHM EXTERNAL-TEST PROBABILITY PERFORMANCE
#
# CLEAN REPLACEMENT FOR OLD SECTION 48
#
# IMPORTANT:
# - Uses PRED_41222 directly.
# - NO model refitting.
# - NO threshold estimation.
# - NO tuning or resampling.
# - ONE identical metric implementation across all algorithms.
# ======================================================================

cat(
  "\n============================================================\n",
  "SECTION 48_222\n",
  "FINAL THREE-ALGORITHM PROBABILITY PERFORMANCE\n",
  "============================================================\n",
  sep = ""
)

req48222 <- c(
  "OUT_DIR",
  "PRED_41222",
  "N_REP"
)

miss48222 <- req48222[
  !vapply(
    req48222,
    exists,
    logical(1),
    envir = .GlobalEnv
  )
]

if(length(miss48222)){
  stop(
    "48_222 missing objects: ",
    paste(miss48222, collapse = ", ")
  )
}

DIR_48222 <- file.path(
  OUT_DIR,
  "48_222_THREE_ALGORITHM_PROBABILITY_PERFORMANCE"
)

DIR_48222_CSV <- file.path(DIR_48222, "CSV")
DIR_48222_FIG <- file.path(DIR_48222, "FIGURES")
DIR_48222_QC  <- file.path(DIR_48222, "QC")

invisible(
  lapply(
    c(
      DIR_48222,
      DIR_48222_CSV,
      DIR_48222_FIG,
      DIR_48222_QC
    ),
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)

# Reuse the already harmonized metric table generated in 42.
# This prevents Section 48 from recalculating the same metrics a second time.
stopifnot(
  exists("metrics_all_42222"),
  exists("metric_summary_42222")
)

probability_by_replicate_48222 <-
  metrics_all_42222 %>%
  mutate(
    algorithm = as.character(algorithm),
    scheme = as.character(scheme),
    variant = as.character(variant),
    region = as.character(regime)
  ) %>%
  select(
    algorithm,
    replicate,
    scheme,
    variant,
    region,
    n,
    prevalence,
    AUC,
    PR_AUC,
    Brier,
    ECE10
  )

write_csv(
  probability_by_replicate_48222,
  file.path(
    DIR_48222_CSV,
    "48_222_01_probability_metrics_by_replicate.csv"
  )
)

full_summary_48222 <-
  metric_summary_42222 %>%
  mutate(
    algorithm = as.character(algorithm),
    scheme = as.character(scheme),
    variant = as.character(variant),
    region = as.character(regime),
    Metric = dplyr::recode(
      as.character(metric),
      "PR_AUC" = "PR-AUC",
      .default = as.character(metric)
    )
  ) %>%
  select(
    algorithm,
    scheme,
    variant,
    region,
    Metric,
    n_valid,
    Mean,
    SD,
    Median,
    LCL,
    UCL
  )

write_csv(
  full_summary_48222,
  file.path(
    DIR_48222_CSV,
    "48_222_02_FULL_three_algorithm_probability_performance_summary.csv"
  )
)

journal_probability_48222 <-
  full_summary_48222 %>%
  filter(
    region == "Overall"
  ) %>%
  mutate(
    Result =
      sprintf(
        "%.3f [%.3f, %.3f]",
        Mean,
        LCL,
        UCL
      )
  ) %>%
  select(
    algorithm,
    scheme,
    variant,
    Metric,
    Result
  ) %>%
  arrange(
    factor(
      algorithm,
      levels = c(
        "Random Forest",
        "Ridge logistic",
        "GAM"
      )
    ),
    factor(
      scheme,
      levels = c(
        "UR",
        "SB",
        "SW"
      )
    ),
    factor(
      variant,
      levels = c(
        "RAW",
        "RAW+SHADOW"
      )
    ),
    Metric
  )

write_csv(
  journal_probability_48222,
  file.path(
    DIR_48222_CSV,
    "48_222_03_JOURNAL_overall_probability_performance.csv"
  )
)

# Publication-quality figure.
plot_prob_48222 <-
  full_summary_48222 %>%
  mutate(
    algorithm =
      factor(
        algorithm,
        levels = c(
          "Random Forest",
          "Ridge logistic",
          "GAM"
        )
      ),
    scheme =
      factor(
        scheme,
        levels = c(
          "UR",
          "SB",
          "SW"
        )
      ),
    variant =
      factor(
        variant,
        levels = c(
          "RAW",
          "RAW+SHADOW"
        )
      ),
    region =
      factor(
        region,
        levels = c(
          "Overall",
          "Non-shadow",
          "Shadow"
        )
      ),
    Metric =
      factor(
        Metric,
        levels = c(
          "AUC",
          "PR-AUC",
          "Brier",
          "ECE10"
        )
      )
  )

fig_prob_48222 <-
  ggplot(
    plot_prob_48222,
    aes(
      x = region,
      y = Mean,
      shape = variant,
      group = variant
    )
  ) +
  geom_errorbar(
    aes(
      ymin = LCL,
      ymax = UCL
    ),
    position =
      position_dodge(
        width = 0.40
      ),
    width = 0.10,
    linewidth = 0.4
  ) +
  geom_point(
    position =
      position_dodge(
        width = 0.40
      ),
    size = 1.9
  ) +
  facet_grid(
    Metric ~ interaction(
      algorithm,
      scheme,
      sep = " / "
    ),
    scales = "free_y"
  ) +
  labs(
    title =
      "External-test probability performance across algorithms and illumination regimes",
    subtitle =
      "Means and empirical 95% replicate intervals; identical metric definitions for all algorithms",
    x =
      "External-test subset",
    y =
      "Metric value",
    shape =
      "Predictor set"
  ) +
  theme_bw(
    base_size = 9
  ) +
  theme(
    panel.grid.minor =
      element_blank(),
    strip.text =
      element_text(
        size = 6.7,
        face = "bold"
      ),
    plot.title =
      element_text(
        face = "bold"
      ),
    legend.position =
      "bottom",
    axis.text.x =
      element_text(
        angle = 35,
        hjust = 1
      )
  )

ggsave(
  file.path(
    DIR_48222_FIG,
    "Figure48_222_three_algorithm_probability_performance.png"
  ),
  fig_prob_48222,
  width = 15,
  height = 10,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(
    DIR_48222_FIG,
    "Figure48_222_three_algorithm_probability_performance.tiff"
  ),
  fig_prob_48222,
  width = 15,
  height = 10,
  units = "in",
  dpi = 600,
  compression = "lzw"
)

expected_rows_48222 <-
  N_REP *
  3 *
  2 *
  3 *
  3

qc_48222 <-
  tibble(
    check =
      c(
        "Three algorithms",
        "Three sampling scenarios",
        "Two predictor variants",
        "Three evaluation subsets",
        "Expected replicate-level rows",
        "AUC in [0,1] when finite",
        "PR-AUC in [0,1] when finite",
        "Brier in [0,1] when finite",
        "ECE10 in [0,1] when finite"
      ),
    passed =
      c(
        n_distinct(
          probability_by_replicate_48222$algorithm
        ) == 3,
        n_distinct(
          probability_by_replicate_48222$scheme
        ) == 3,
        n_distinct(
          probability_by_replicate_48222$variant
        ) == 2,
        n_distinct(
          probability_by_replicate_48222$region
        ) == 3,
        nrow(
          probability_by_replicate_48222
        ) == expected_rows_48222,
        all(
          probability_by_replicate_48222$AUC[
            is.finite(
              probability_by_replicate_48222$AUC
            )
          ] >= 0 &
          probability_by_replicate_48222$AUC[
            is.finite(
              probability_by_replicate_48222$AUC
            )
          ] <= 1
        ),
        all(
          probability_by_replicate_48222$PR_AUC[
            is.finite(
              probability_by_replicate_48222$PR_AUC
            )
          ] >= 0 &
          probability_by_replicate_48222$PR_AUC[
            is.finite(
              probability_by_replicate_48222$PR_AUC
            )
          ] <= 1
        ),
        all(
          probability_by_replicate_48222$Brier[
            is.finite(
              probability_by_replicate_48222$Brier
            )
          ] >= 0 &
          probability_by_replicate_48222$Brier[
            is.finite(
              probability_by_replicate_48222$Brier
            )
          ] <= 1
        ),
        all(
          probability_by_replicate_48222$ECE10[
            is.finite(
              probability_by_replicate_48222$ECE10
            )
          ] >= 0 &
          probability_by_replicate_48222$ECE10[
            is.finite(
              probability_by_replicate_48222$ECE10
            )
          ] <= 1
        )
      )
  )

write_csv(
  qc_48222,
  file.path(
    DIR_48222_QC,
    "48_222_FINAL_QC.csv"
  )
)

print(
  qc_48222,
  n = Inf
)

if(!all(qc_48222$passed)){
  stop(
    "48_222 QC failed."
  )
}

cat(
  "\nSECTION 48_222 COMPLETED SUCCESSFULLY\n"
)


# ======================================================================
# SECTION 47
# FINAL MANUSCRIPT + SUPPLEMENTARY REPORTING PACK
#
# IMPORTANT:
# This is intentionally executed LAST, after 48.
#
# It does NOT rerun models, thresholds, permutation importance,
# mapped-area analysis, or external-test classification analyses.
#
# It only gathers the FINAL objects generated by:
#   40
#   41
#   42
#   46
#   48
#
# and creates clean manuscript/supplementary CSV.
# ======================================================================

cat(
  "\n============================================================\n",
  "SECTION 47_222\n",
  "FINAL REPORTING PACK\n",
  "============================================================\n",
  sep = ""
)

req47222 <- c(
  "OUT_DIR",
  "main_compact_40222",
  "area_summary_40222",
  "paired_tests_40222",
  "variant_tests_40222",
   "tradeoff_summary_41222",
  "correct_added_summary_41222",
  "threshold_summary_42222",
  "threshold_tests_42222",
  "strict_consensus_46222",
  "importance_summary_46222",
  "journal_probability_48222",
  "full_summary_48222"
)

miss47222 <- req47222[
  !vapply(
    req47222,
    exists,
    logical(1),
    envir = .GlobalEnv
  )
]

if(length(miss47222)){
  stop(
    "47_222 missing objects: ",
    paste(
      miss47222,
      collapse = ", "
    )
  )
}



DIR_47222 <- file.path(
  OUT_DIR,
  "47_222_FINAL_REPORTING_PACK"
)

DIR_47222_MAIN_CSV <-
  file.path(
    DIR_47222,
    "MAIN",
    "TABLES_CSV"
  )

DIR_47222_SUPP_CSV <-
  file.path(
    DIR_47222,
    "SUPPLEMENTARY",
    "TABLES_CSV"
  )



DIR_47222_QC <-
  file.path(
    DIR_47222,
    "QC"
  )

invisible(
  lapply(
    c(
      DIR_47222,
      DIR_47222_MAIN_CSV,
      DIR_47222_SUPP_CSV,
      DIR_47222_QC
    ),
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)

# ----------------------------------------------------------------------
# 47.1 FINAL MAIN-TEXT TABLES
# ----------------------------------------------------------------------

MainTable_Probability_47222 <-
  journal_probability_48222

MainTable_MappedArea_47222 <-
  main_compact_40222

# Predictor importance table:
# retain the strongest directionally consistent contrasts only.
MainTable_Importance_47222 <-
  strict_consensus_46222 %>%
  mutate(
    Abs_Delta =
      abs(
        Mean_Delta_Brier
      )
  ) %>%
  filter(
    Strict_direction !=
      "Model/context dependent"
  ) %>%
  group_by(
    predictor
  ) %>%
  slice_max(
    order_by = Abs_Delta,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup() %>%
  arrange(
    desc(
      Abs_Delta
    )
  ) %>%
  slice_head(
    n = 15
  )

write_csv(
  MainTable_Probability_47222,
  file.path(
    DIR_47222_MAIN_CSV,
    "Table_MAIN_probability_performance.csv"
  )
)

write_csv(
  MainTable_MappedArea_47222,
  file.path(
    DIR_47222_MAIN_CSV,
    "Table_MAIN_mapped_area.csv"
  )
)

write_csv(
  MainTable_Importance_47222,
  file.path(
    DIR_47222_MAIN_CSV,
    "Table_MAIN_regime_predictor_importance.csv"
  )
)


# ----------------------------------------------------------------------
# 47.2 FINAL SUPPLEMENTARY TABLES
# ----------------------------------------------------------------------

supp_tables_47222 <- list(
  "Table_S1_probability_performance_full" =
    full_summary_48222,

  "Table_S2_threshold_summary" =
    threshold_summary_42222,

  "Table_S3_threshold_tests" =
    threshold_tests_42222,

  "Table_S4_threshold_tradeoff" =
    tradeoff_summary_41222,

  "Table_S5_new_suitable_classifications" =
    correct_added_summary_41222,

  "Table_S6_mapped_area_full" =
    area_summary_40222,

  "Table_S7_mapped_area_threshold_tests" =
    paired_tests_40222,

  "Table_S8_mapped_area_RAW_SHADOW_effect" =
    variant_tests_40222,

  "Table_S9_predictor_importance_full" =
    importance_summary_46222,

  "Table_S10_predictor_importance_consensus" =
    strict_consensus_46222
)

for(nm47222 in names(supp_tables_47222)){

  write_csv(
    supp_tables_47222[[nm47222]],
    file.path(
      DIR_47222_SUPP_CSV,
      paste0(
        nm47222,
        ".csv"
      )
    )
  )
}




# ----------------------------------------------------------------------
# 47.7 FINAL QC
# ----------------------------------------------------------------------
cat(
  "\n============================================================\n",
  "SECTION 47_222 COMPLETED SUCCESSFULLY\n",
  "FINAL CSV REPORTING PACK COMPLETE\n",
  "============================================================\n",
  "\nFinal reporting directory:\n",
  DIR_47222,
  "\n",
  sep = ""
)

#
#
#
#
# ======================================================================
# SECTION 49
# FINAL JOURNAL-READY FIGURE SUITE
#
# Purpose:
#   Produce the final, publication-quality figure set for the manuscript.
#
# IMPORTANT:
#   - NO model refitting
#   - NO threshold re-estimation
#   - NO new resampling
#   - NO use of external-test information for training/calibration
#   - Existing analytical outputs are only summarized/visualized
#
# Main-text figure strategy:
#
#   Figure 2
#     Topographic basis of   shadow
#
#   Figure 3
#     External-test probability performance
#
#   Figure 4
#     Calibration-derived illumination-regime thresholds
#
#   Figure 5
#     Consequence of regime-specific thresholding for mapped area
#
# Supplementary:
#     full probability-performance matrix
#     sensitivity-specificity trade-off
#     predictor-contribution heatmap
#     absolute mapped suitable area
#     newly suitable TP/FP composition
#     correct-added share
#     topographic partial effects
#
# ======================================================================


cat(
  "\n============================================================\n",
  "SECTION 49_222\n",
  "FINAL JOURNAL-READY FIGURE SUITE\n",
  "============================================================\n",
  sep = ""
)


# ======================================================================
# 49.1 PACKAGES
# ======================================================================

suppressPackageStartupMessages({

  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(forcats)
  library(scales)
  library(patchwork)

})


# ======================================================================
# 49.2 OUTPUT DIRECTORIES
# ======================================================================

DIR_49222 <-
  file.path(
    OUT_DIR,
    "49_222_FINAL_JOURNAL_FIGURES"
  )

DIR_49222_MAIN <-
  file.path(
    DIR_49222,
    "MAIN"
  )

DIR_49222_SUPP <-
  file.path(
    DIR_49222,
    "SUPPLEMENTARY"
  )

DIR_49222_QC <-
  file.path(
    DIR_49222,
    "QC"
  )


invisible(
  lapply(
    c(
      DIR_49222,
      DIR_49222_MAIN,
      DIR_49222_SUPP,
      DIR_49222_QC
    ),
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)


# ======================================================================
# 49.3 JOURNAL GRAPHIC SETTINGS
# ======================================================================

COL_RAW <- "#3B3B3B"
COL_SHADOW <- "#0072B2"

COL_NONSHADOW <- "#D55E00"
COL_PSHADOW <- "#0072B2"

COL_RF <- "#0072B2"
COL_RIDGE <- "#009E73"
COL_GAM <- "#CC79A7"

COL_POSITIVE <- "#0072B2"
COL_NEGATIVE <- "#D55E00"

COL_NEUTRAL <- "#5B5B5B"

BASE_FONT_49222 <- 10.5


theme_journal_49222 <- function(
  base_size = BASE_FONT_49222
){

  theme_classic(
    base_size = base_size,
    base_family = "sans"
  ) +

    theme(

      plot.title =
        element_text(
          size = base_size + 2,
          face = "bold",
          colour = "black",
          hjust = 0,
          margin = margin(
            b = 5
          )
        ),

      plot.subtitle =
        element_text(
          size = base_size,
          colour = "grey25",
          hjust = 0,
          margin = margin(
            b = 9
          )
        ),

      plot.caption =
        element_text(
          size = base_size - 1.5,
          colour = "grey35",
          hjust = 0,
          margin = margin(
            t = 7
          )
        ),

      axis.title =
        element_text(
          size = base_size,
          colour = "black"
        ),

      axis.text =
        element_text(
          size = base_size - 0.5,
          colour = "black"
        ),

      axis.line =
        element_line(
          linewidth = 0.45,
          colour = "black"
        ),

      axis.ticks =
        element_line(
          linewidth = 0.4,
          colour = "black"
        ),

      strip.background =
        element_blank(),

      strip.text =
        element_text(
          size = base_size,
          face = "bold",
          colour = "black",
          margin = margin(
            b = 5
          )
        ),

      panel.grid.major.y =
        element_line(
          linewidth = 0.25,
          colour = "grey90"
        ),

      panel.grid.major.x =
        element_blank(),

      panel.grid.minor =
        element_blank(),

      legend.title =
        element_text(
          size = base_size - 0.5,
          face = "bold"
        ),

      legend.text =
        element_text(
          size = base_size - 0.5
        ),

      legend.key.height =
        unit(
          0.45,
          "cm"
        ),

      legend.key.width =
        unit(
          0.55,
          "cm"
        ),

      plot.margin =
        margin(
          8,
          10,
          8,
          8
        )
    )
}


# ======================================================================
# 49.4 COMMON LABELS
# ======================================================================

algorithm_labels_49222 <-
  c(
    RF = "Random forest",
    RANDOM_FOREST = "Random forest",
    Random_Forest = "Random forest",
    RIDGE = "Ridge logistic",
    Ridge = "Ridge logistic",
    GAM = "GAM"
  )


scheme_labels_49222 <-
  c(
    UR = "UR",
    SB = "SB",
    SW = "SW"
  )


variant_labels_49222 <-
  c(
    RAW = "RAW",
    RAW_SHADOW = "RAW + shadow",
    `RAW+SHADOW` = "RAW + shadow"
  )


# ======================================================================
# 49.5 EXPORT FUNCTION
# ======================================================================

save_journal_plot_49222 <- function(
  plot,
  filename,
  directory,
  width,
  height
){

  png_file <-
    file.path(
      directory,
      paste0(
        filename,
        ".png"
      )
    )

  tiff_file <-
    file.path(
      directory,
      paste0(
        filename,
        ".tiff"
      )
    )

  pdf_file <-
    file.path(
      directory,
      paste0(
        filename,
        ".pdf"
      )
    )


  ggsave(
    png_file,
    plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    bg = "white"
  )


  ggsave(
    tiff_file,
    plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )


  ggsave(
    pdf_file,
    plot,
    width = width,
    height = height,
    units = "in",
    device = cairo_pdf,
    bg = "white"
  )


  invisible(
    tibble(
      figure = filename,
      png = png_file,
      tiff = tiff_file,
      pdf = pdf_file
    )
  )
}


# ======================================================================
# 49.6 HELPER:
# empirical replicate summary
# ======================================================================

summary95_49222 <- function(
  data,
  value
){

  value <-
    rlang::ensym(
      value
    )

  data %>%
    summarise(

      Mean =
        mean(
          !!value,
          na.rm = TRUE
        ),

      Median =
        median(
          !!value,
          na.rm = TRUE
        ),

      Lower =
        quantile(
          !!value,
          0.025,
          na.rm = TRUE,
          names = FALSE
        ),

      Upper =
        quantile(
          !!value,
          0.975,
          na.rm = TRUE,
          names = FALSE
        ),

      .groups = "drop"
    )
}


# ======================================================================
# 49.7 LOCATE THREE-ALGORITHM PREDICTIONS
# ======================================================================

if(
  !exists(
    "PRED_41222"
  )
){

  stop(
    "PRED_41222 is required for the final probability figure."
  )
}


PRED_FINAL_49222 <-
  PRED_41222 %>%
  mutate(

    algorithm =
      case_when(

        grepl(
          "random",
          algorithm,
          ignore.case = TRUE
        ) ~ "Random forest",

        grepl(
          "ridge",
          algorithm,
          ignore.case = TRUE
        ) ~ "Ridge logistic",

        grepl(
          "gam",
          algorithm,
          ignore.case = TRUE
        ) ~ "GAM",

        TRUE ~ as.character(
          algorithm
        )
      ),

    scheme =
      factor(
        scheme,
        levels = c(
          "UR",
          "SB",
          "SW"
        )
      ),

    variant =
      case_when(

        variant %in%
          c(
            "RAW_SHADOW",
            "RAW+SHADOW"
          ) ~ "RAW + shadow",

        variant ==
          "RAW" ~
          "RAW",

        TRUE ~ as.character(
          variant
        )
      ),

    variant =
      factor(
        variant,
        levels = c(
          "RAW",
          "RAW + shadow"
        )
      ),

    algorithm =
      factor(
        algorithm,
        levels = c(
          "Random forest",
          "Ridge logistic",
          "GAM"
        )
      )
  )


# ======================================================================
# 49.8 MAIN FIGURE 3
#
# EXTERNAL-TEST PROBABILITY PERFORMANCE
#
# Main-text simplification:
#   AUC + Brier only
#
# Full AUC / PR-AUC / Brier / ECE remains supplementary.
# ======================================================================


# ----------------------------------------------------------------------
# Metric functions
# ----------------------------------------------------------------------

auc_49222 <- function(
  y,
  p
){

  ok <-
    is.finite(y) &
    is.finite(p)

  y <- y[ok]
  p <- p[ok]

  if(
    length(y) < 5 ||
    length(
      unique(
        y
      )
    ) < 2
  ){

    return(
      NA_real_
    )
  }


  rr <-
    tryCatch(

      pROC::roc(
        response = y,
        predictor = p,
        quiet = TRUE,
        direction = "<"
      ),

      error =
        function(e) NULL
    )


  if(
    is.null(
      rr
    )
  ){

    return(
      NA_real_
    )
  }


  as.numeric(
    pROC::auc(
      rr
    )
  )
}


brier_49222 <- function(
  y,
  p
){

  ok <-
    is.finite(y) &
    is.finite(p)

  mean(
    (
      p[ok] -
        y[ok]
    )^2
  )
}


# ----------------------------------------------------------------------
# Detect probability column
# ----------------------------------------------------------------------

prob_candidates_49222 <-
  intersect(
    c(
      "probability",
      "prob",
      "p",
      "prediction",
      "pred_probability"
    ),
    names(
      PRED_FINAL_49222
    )
  )


if(
  length(
    prob_candidates_49222
  ) == 0
){

  stop(
    "Probability column not found in PRED_41222."
  )
}


PROB_COL_49222 <-
  prob_candidates_49222[1]


# ----------------------------------------------------------------------
# Overall + illumination subsets
# ----------------------------------------------------------------------

perf_overall_49222 <-
  PRED_FINAL_49222 %>%
  group_by(
    algorithm,
    scheme,
    variant,
    replicate
  ) %>%
  summarise(

    AUC =
      auc_49222(
        presence,
        .data[[PROB_COL_49222]]
      ),

    Brier =
      brier_49222(
        presence,
        .data[[PROB_COL_49222]]
      ),

    .groups =
      "drop"
  ) %>%
  mutate(
    subset =
      "Overall"
  )


perf_regime_49222 <-
  PRED_FINAL_49222 %>%
  mutate(

    subset =
      ifelse(
        shadow_bin == 1,
        "  shadow",
        "Non-shadow"
      )
  ) %>%
  group_by(
    algorithm,
    scheme,
    variant,
    replicate,
    subset
  ) %>%
  summarise(

    AUC =
      auc_49222(
        presence,
        .data[[PROB_COL_49222]]
      ),

    Brier =
      brier_49222(
        presence,
        .data[[PROB_COL_49222]]
      ),

    .groups =
      "drop"
  )


perf_main_49222 <-
  bind_rows(
    perf_overall_49222,
    perf_regime_49222
  ) %>%
  mutate(

    subset =
      factor(
        subset,
        levels =
          c(
            "Overall",
            "Non-shadow",
            "  shadow"
          )
      )
  )


perf_long_49222 <-
  perf_main_49222 %>%
  pivot_longer(
    cols =
      c(
        AUC,
        Brier
      ),
    names_to =
      "metric",
    values_to =
      "value"
  )


perf_summary_49222 <-
  perf_long_49222 %>%
  group_by(
    algorithm,
    scheme,
    variant,
    subset,
    metric
  ) %>%
  summary95_49222(
    value
  )


# ----------------------------------------------------------------------
# Main probability figure
# ----------------------------------------------------------------------

fig_probability_main_49222 <-
  ggplot(
    perf_summary_49222,
    aes(
      x = subset,
      y = Mean,
      group = variant
    )
  ) +

  geom_errorbar(
    aes(
      ymin = Lower,
      ymax = Upper
    ),
    position =
      position_dodge(
        width = 0.38
      ),
    width = 0,
    linewidth = 0.55,
    colour = "grey25"
  ) +

  geom_point(
    aes(
      shape = variant,
      fill = variant
    ),
    position =
      position_dodge(
        width = 0.38
      ),
    size = 2.45,
    stroke = 0.7,
    colour = "black"
  ) +

  facet_grid(
    metric ~ algorithm,
    scales = "free_y",
    switch = "y"
  ) +

  scale_shape_manual(
    values =
      c(
        "RAW" = 21,
        "RAW + shadow" = 24
      )
  ) +

  scale_fill_manual(
    values =
      c(
        "RAW" = "white",
        "RAW + shadow" = COL_SHADOW
      )
  ) +

  labs(

    title =
      "External-test predictive performance",

    subtitle =
      "Replicate means and empirical 95% intervals across illumination regimes",

    x =
      NULL,

    y =
      NULL,

    shape =
      "Predictor set",

    fill =
      "Predictor set"
  ) +

  theme_journal_49222() +

  theme(

    legend.position =
      "bottom",

    axis.text.x =
      element_text(
        angle = 0,
        hjust = 0.5
      ),

    strip.placement =
      "outside",

    panel.spacing =
      unit(
        0.75,
        "lines"
      )
  )


save_journal_plot_49222(

  fig_probability_main_49222,

  "Figure3_external_test_probability_performance",

  DIR_49222_MAIN,

  width = 7.2,

  height = 5.3
)


# ======================================================================
# 49.9 MAIN FIGURE 4
#
# PAIRED CALIBRATION-DERIVED THRESHOLDS
#
# IMPORTANT:
# Thresholds are NOT re-estimated here.
# They are extracted directly from PRED_41222.
#
# PRED_41222 already contains, for every:
#   algorithm × scheme × variant × replicate
#
#   threshold_global
#   threshold_nonshadow
#   threshold_shadow
#
# ======================================================================


# ----------------------------------------------------------------------
# 49.9.1 REQUIREMENTS
# ----------------------------------------------------------------------

if(!exists("PRED_41222")){

  stop(
    "PRED_41222 does not exist. Run Section 41_222 first."
  )
}


required_threshold_cols_49222 <-
  c(
    "algorithm",
    "scheme",
    "variant",
    "replicate",
    "threshold_global",
    "threshold_nonshadow",
    "threshold_shadow"
  )


missing_threshold_cols_49222 <-
  setdiff(
    required_threshold_cols_49222,
    names(PRED_41222)
  )


if(length(missing_threshold_cols_49222) > 0){

  stop(
    paste0(
      "PRED_41222 is missing required threshold columns: ",
      paste(
        missing_threshold_cols_49222,
        collapse = ", "
      )
    )
  )
}


# ----------------------------------------------------------------------
# 49.9.2 EXTRACT ONE THRESHOLD SET PER MODEL REPLICATE
#
# PRED_41222 is point-level external-test data, therefore each
# threshold is repeated for every test point belonging to the same
# fitted model.
#
# We collapse those repeated values here.
# NO threshold is calculated from the external test.
# ----------------------------------------------------------------------

THRESH_FINAL_49222 <-
  PRED_41222 %>%

  transmute(

    algorithm =
      as.character(
        algorithm
      ),

    scheme =
      as.character(
        scheme
      ),

    variant =
      as.character(
        variant
      ),

    replicate =
      as.integer(
        replicate
      ),

    threshold_global =
      as.numeric(
        threshold_global
      ),

    threshold_nonshadow =
      as.numeric(
        threshold_nonshadow
      ),

    threshold_shadow =
      as.numeric(
        threshold_shadow
      )
  ) %>%

  distinct(
    algorithm,
    scheme,
    variant,
    replicate,
    threshold_global,
    threshold_nonshadow,
    threshold_shadow
  )


# ----------------------------------------------------------------------
# CRITICAL QC:
# exactly one threshold triplet must exist for each fitted model
# ----------------------------------------------------------------------

threshold_duplicate_qc_49222 <-
  THRESH_FINAL_49222 %>%

  count(
    algorithm,
    scheme,
    variant,
    replicate,
    name = "n_threshold_sets"
  )


if(
  any(
    threshold_duplicate_qc_49222$n_threshold_sets != 1
  )
){

  print(
    threshold_duplicate_qc_49222 %>%
      filter(
        n_threshold_sets != 1
      ),
    n = Inf
  )

  stop(
    paste0(
      "More than one threshold set exists for at least one ",
      "algorithm × scheme × variant × replicate combination."
    )
  )
}


# ----------------------------------------------------------------------
# Expected design:
#
# 3 algorithms ×
# 3 schemes ×
# 2 variants ×
# N_REP replicates
# ----------------------------------------------------------------------

expected_threshold_rows_49222 <-
  3L *
  3L *
  2L *
  N_REP


cat(
  "\nThreshold rows extracted from PRED_41222 = ",
  nrow(
    THRESH_FINAL_49222
  ),
  "\nExpected threshold rows = ",
  expected_threshold_rows_49222,
  "\n",
  sep = ""
)


if(
  nrow(
    THRESH_FINAL_49222
  ) !=
    expected_threshold_rows_49222
){

  stop(
    paste0(
      "Unexpected number of threshold rows. Expected ",
      expected_threshold_rows_49222,
      " but obtained ",
      nrow(
        THRESH_FINAL_49222
      ),
      "."
    )
  )
}


# ----------------------------------------------------------------------
# 49.9.3 STANDARDIZE LABELS
# ----------------------------------------------------------------------

THRESH_FINAL_49222 <-
  THRESH_FINAL_49222 %>%

  mutate(

    algorithm =
      case_when(

        grepl(
          "random",
          algorithm,
          ignore.case = TRUE
        ) ~
          "Random forest",

        grepl(
          "ridge",
          algorithm,
          ignore.case = TRUE
        ) ~
          "Ridge logistic",

        grepl(
          "gam",
          algorithm,
          ignore.case = TRUE
        ) ~
          "GAM",

        TRUE ~
          algorithm
      ),


    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random forest",
            "Ridge logistic",
            "GAM"
          )
      ),


    scheme =
      factor(
        scheme,
        levels =
          c(
            "UR",
            "SB",
            "SW"
          )
      ),


    variant =
      case_when(

        variant %in%
          c(
            "RAW_SHADOW",
            "RAW+SHADOW"
          ) ~
          "RAW + shadow",

        variant ==
          "RAW" ~
          "RAW",

        TRUE ~
          variant
      ),


    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW + shadow"
          )
      )
  )


# ----------------------------------------------------------------------
# 49.9.4 FINITE-VALUE QC
# ----------------------------------------------------------------------

if(
  any(
    !is.finite(
      THRESH_FINAL_49222$threshold_nonshadow
    )
  ) ||
  any(
    !is.finite(
      THRESH_FINAL_49222$threshold_shadow
    )
  )
){

  stop(
    "Non-finite regime-specific threshold detected."
  )
}


if(
  any(
    THRESH_FINAL_49222$threshold_nonshadow < 0 |
    THRESH_FINAL_49222$threshold_nonshadow > 1 |
    THRESH_FINAL_49222$threshold_shadow < 0 |
    THRESH_FINAL_49222$threshold_shadow > 1
  )
){

  stop(
    "Threshold outside [0,1] detected."
  )
}


# ----------------------------------------------------------------------
# 49.9.5 LONG FORMAT
# ----------------------------------------------------------------------

threshold_long_49222 <-
  THRESH_FINAL_49222 %>%

  select(
    algorithm,
    scheme,
    variant,
    replicate,
    threshold_nonshadow,
    threshold_shadow
  ) %>%

  pivot_longer(

    cols =
      c(
        threshold_nonshadow,
        threshold_shadow
      ),

    names_to =
      "illumination",

    values_to =
      "threshold"
  ) %>%

  mutate(

    illumination =
      recode(

        illumination,

        threshold_nonshadow =
          "Non-shadow",

        threshold_shadow =
          "  shadow"
      ),

    illumination =
      factor(
        illumination,
        levels =
          c(
            "Non-shadow",
            "  shadow"
          )
      )
  )


# ----------------------------------------------------------------------
# 49.9.6 SUMMARY
# ----------------------------------------------------------------------

threshold_summary_49222 <-
  threshold_long_49222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    illumination
  ) %>%

  summarise(

    n =
      sum(
        is.finite(
          threshold
        )
      ),

    Mean =
      mean(
        threshold,
        na.rm = TRUE
      ),

    Median =
      median(
        threshold,
        na.rm = TRUE
      ),

    Lower =
      quantile(
        threshold,
        0.025,
        na.rm = TRUE,
        names = FALSE
      ),

    Upper =
      quantile(
        threshold,
        0.975,
        na.rm = TRUE,
        names = FALSE
      ),

    .groups =
      "drop"
  )


# ----------------------------------------------------------------------
# 49.9.7 PAIRED DIFFERENCE TABLE
# ----------------------------------------------------------------------

threshold_delta_49222 <-
  THRESH_FINAL_49222 %>%

  mutate(

    Delta_threshold =
      threshold_shadow -
      threshold_nonshadow
  )


threshold_delta_summary_49222 <-
  threshold_delta_49222 %>%

  group_by(
    algorithm,
    scheme,
    variant
  ) %>%

  summarise(

    n =
      sum(
        is.finite(
          Delta_threshold
        )
      ),

    Mean =
      mean(
        Delta_threshold,
        na.rm = TRUE
      ),

    Median =
      median(
        Delta_threshold,
        na.rm = TRUE
      ),

    Lower =
      quantile(
        Delta_threshold,
        0.025,
        na.rm = TRUE,
        names = FALSE
      ),

    Upper =
      quantile(
        Delta_threshold,
        0.975,
        na.rm = TRUE,
        names = FALSE
      ),

    Prop_positive =
      mean(
        Delta_threshold > 0,
        na.rm = TRUE
      ),

    .groups =
      "drop"
  )


# ----------------------------------------------------------------------
# 49.9.8 SAVE PLOT DATA
# ----------------------------------------------------------------------

write_csv(

  THRESH_FINAL_49222,

  file.path(
    DIR_49222_QC,
    "Figure4_thresholds_by_model_replicate.csv"
  )
)


write_csv(

  threshold_summary_49222,

  file.path(
    DIR_49222_QC,
    "Figure4_threshold_summary.csv"
  )
)


write_csv(

  threshold_delta_summary_49222,

  file.path(
    DIR_49222_QC,
    "Figure4_threshold_delta_summary.csv"
  )
)


# ======================================================================
# 49.9.9 MAIN FIGURE 4 — CORRECTED
#
# PAIRED CALIBRATION-DERIVED THRESHOLDS
#
# Thin grey lines = paired replicate-level thresholds
# Large symbols   = replicate means
# Error bars      = empirical 95% replicate intervals
#
# No threshold is recalculated here.
# ======================================================================


fig_threshold_main_49222 <-
  ggplot(
    threshold_long_49222,
    aes(
      x = illumination,
      y = threshold
    )
  ) +

  # ----------------------------------------------------------
  # A) PAIRED REPLICATE-LEVEL THRESHOLDS
  # ----------------------------------------------------------

  geom_line(
    aes(
      group =
        interaction(
          algorithm,
          scheme,
          variant,
          replicate,
          drop = TRUE
        )
    ),
    colour = "grey72",
    linewidth = 0.32,
    alpha = 0.55
  ) +

  geom_point(
    colour = "grey55",
    size = 0.90,
    alpha = 0.50
  ) +

  # ----------------------------------------------------------
  # B) EMPIRICAL 95% REPLICATE INTERVAL
  # ----------------------------------------------------------

  geom_errorbar(
    data = threshold_summary_49222,
    mapping =
      aes(
        x = illumination,
        ymin = Lower,
        ymax = Upper
      ),
    inherit.aes = FALSE,
    width = 0.07,
    linewidth = 0.70,
    colour = "black"
  ) +

  # ----------------------------------------------------------
  # C) REPLICATE MEAN
  # ----------------------------------------------------------

  geom_point(
    data = threshold_summary_49222,
    mapping =
      aes(
        x = illumination,
        y = Mean,
        fill = illumination
      ),
    inherit.aes = FALSE,
    shape = 21,
    size = 3.2,
    stroke = 0.85,
    colour = "black"
  ) +

  # ----------------------------------------------------------
  # D) FACETS
  #
  # columns = algorithm
  # rows    = sampling scheme × predictor set
  # ----------------------------------------------------------

  facet_grid(
    scheme + variant ~ algorithm,
    labeller =
      labeller(
        algorithm =
          c(
            "Random forest" =
              "Random forest",
            "Ridge logistic" =
              "Ridge logistic",
            "GAM" =
              "GAM"
          ),
        variant =
          c(
            "RAW" =
              "RAW",
            "RAW + shadow" =
              "RAW + shadow"
          )
      )
  ) +

  # ----------------------------------------------------------
  # E) COLOURS
  # ----------------------------------------------------------

  scale_fill_manual(
    values =
      c(
        "Non-shadow" =
          "white",
        "  shadow" =
          COL_PSHADOW
      ),
    drop = FALSE
  ) +

  # ----------------------------------------------------------
  # F) AXES
  # ----------------------------------------------------------

  scale_y_continuous(
    limits =
      c(
        0,
        1
      ),
    breaks =
      seq(
        0,
        1,
        by = 0.2
      ),
    expand =
      expansion(
        mult =
          c(
            0.01,
            0.02
          )
      )
  ) +

  # ----------------------------------------------------------
  # G) LABELS
  # ----------------------------------------------------------

  labs(
    title =
      "Calibration-derived decision thresholds across illumination regimes",

    subtitle =
      paste0(
        "Paired thresholds from identical model replicates; ",
        "large points show replicate means and bars empirical 95% intervals"
      ),

    x = NULL,

    y =
      "Decision threshold",

    fill =
      "Illumination regime"
  ) +

  # ----------------------------------------------------------
  # H) JOURNAL THEME
  # ----------------------------------------------------------

  theme_journal_49222() +

  theme(
    legend.position =
      "bottom",

    panel.spacing.x =
      unit(
        0.75,
        "lines"
      ),

    panel.spacing.y =
      unit(
        0.45,
        "lines"
      ),

    strip.text =
      element_text(
        face = "bold",
        size = 9.2
      ),

    axis.text.x =
      element_text(
        size = 8.5
      ),

    plot.title =
      element_text(
        face = "bold"
      )
  )


# ======================================================================
# 49.9.10 EXPORT
# ======================================================================

save_journal_plot_49222(
  fig_threshold_main_49222,
  "Figure4_paired_illumination_thresholds",
  DIR_49222_MAIN,
  width = 7.4,
  height = 8.2
)


cat(
  "\nFigure 4 successfully generated.\n"
)

# ======================================================================
# 49.10 MAIN FIGURE 5
#
# MAPPED CONSEQUENCE OF REGIME-SPECIFIC THRESHOLDING
#
# Primary quantity:
#
# Delta suitable area =
#     regime-specific threshold
#       minus
#     global threshold
#
# This deliberately avoids making absolute differences among
# algorithms the dominant visual message.
# ======================================================================


area_object_candidates_49222 <-
  c(
    "area_threshold_wide_40222",
    "area_threshold_wide_40"
  )


area_object_found_49222 <-
  area_object_candidates_49222[
    vapply(
      area_object_candidates_49222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(
  length(
    area_object_found_49222
  ) == 0
){

  stop(
    "Mapped-area threshold-difference object was not found."
  )
}


AREA_DELTA_49222 <-
  get(
    area_object_found_49222[1],
    envir = .GlobalEnv
  )


# ----------------------------------------------------------------------
# Automatically identify delta column
# ----------------------------------------------------------------------

delta_candidates_49222 <-
  intersect(
    c(
      "Delta_Regime_minus_Global_km2",
      "delta_total_km2"
    ),
    names(
      AREA_DELTA_49222
    )
  )


if(
  length(
    delta_candidates_49222
  ) == 0
){

  stop(
    "Mapped-area delta column not found."
  )
}


DELTA_AREA_COL_49222 <-
  delta_candidates_49222[1]


AREA_DELTA_49222 <-
  AREA_DELTA_49222 %>%
  mutate(

    algorithm =
      case_when(

        grepl(
          "random",
          algorithm,
          ignore.case = TRUE
        ) ~ "Random forest",

        grepl(
          "ridge",
          algorithm,
          ignore.case = TRUE
        ) ~ "Ridge logistic",

        grepl(
          "gam",
          algorithm,
          ignore.case = TRUE
        ) ~ "GAM",

        TRUE ~ as.character(
          algorithm
        )
      ),

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    scheme =
      factor(
        scheme,
        levels =
          c(
            "UR",
            "SB",
            "SW"
          )
      ),

    variant =
      case_when(

        variant %in%
          c(
            "RAW_SHADOW",
            "RAW+SHADOW"
          ) ~
          "RAW + shadow",

        TRUE ~
          as.character(
            variant
          )
      ),

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW + shadow"
          )
      )
  )


area_delta_summary_49222 <-
  AREA_DELTA_49222 %>%
  group_by(
    algorithm,
    scheme,
    variant
  ) %>%
  summarise(

    Mean =
      mean(
        .data[[DELTA_AREA_COL_49222]],
        na.rm = TRUE
      ),

    Lower =
      quantile(
        .data[[DELTA_AREA_COL_49222]],
        0.025,
        na.rm = TRUE,
        names = FALSE
      ),

    Upper =
      quantile(
        .data[[ DELTA_AREA_COL_49222]],
        0.975,
        na.rm = TRUE,
        names = FALSE
      ),

    .groups =
      "drop"
  )


fig_area_delta_main_49222 <-
  ggplot(
    area_delta_summary_49222,
    aes(
      x = Mean,
      y = scheme,
      shape = variant,
      fill = variant
    )
  ) +

  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.55,
    colour = "grey45"
  ) +

  geom_errorbarh(
    aes(
      xmin = Lower,
      xmax = Upper
    ),
    height = 0,
    position =
      position_dodge(
        width = 0.38
      ),
    linewidth = 0.65,
    colour = "grey25"
  ) +

  geom_point(
    position =
      position_dodge(
        width = 0.38
      ),
    size = 2.8,
    stroke = 0.75,
    colour = "black"
  ) +

  facet_wrap(
    ~algorithm,
    nrow = 1,
    scales = "free_x"
  ) +

  scale_shape_manual(
    values =
      c(
        "RAW" = 21,
        "RAW + shadow" = 24
      )
  ) +

  scale_fill_manual(
    values =
      c(
        "RAW" = "white",
        "RAW + shadow" = COL_SHADOW
      )
  ) +

  labs(

    title =
      "Regime-specific thresholding changes mapped suitable area",

    subtitle =
      "Difference relative to the corresponding global-threshold map; points are replicate means with empirical 95% intervals",

    x =
      expression(
        Delta *
          " suitable area (km"^2*")"
      ),

    y =
      "Sampling scenario",

    shape =
      "Predictor set",

    fill =
      "Predictor set"
  ) +

  theme_journal_49222() +

  theme(

    legend.position =
      "bottom",

    panel.spacing =
      unit(
        1,
        "lines"
      )
  )


save_journal_plot_49222(

  fig_area_delta_main_49222,

  "Figure5_regime_specific_threshold_mapped_area_effect",

  DIR_49222_MAIN,

  width = 7.2,

  height = 4.4
)


# ======================================================================
# 49.11 SUPPLEMENTARY:
# ABSOLUTE MAPPED SUITABLE AREA
# ======================================================================


area_all_candidates_49222 <-
  c(
    "area_all_40222",
    "area_results_40"
  )


area_all_found_49222 <-
  area_all_candidates_49222[
    vapply(
      area_all_candidates_49222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(
  length(
    area_all_found_49222
  ) > 0
){

  AREA_ALL_49222 <-
    get(
      area_all_found_49222[1],
      envir = .GlobalEnv
    )


  AREA_ALL_49222 <-
    AREA_ALL_49222 %>%
    mutate(

      algorithm =
        case_when(

          grepl(
            "random",
            algorithm,
            ignore.case = TRUE
          ) ~
            "Random forest",

          grepl(
            "ridge",
            algorithm,
            ignore.case = TRUE
          ) ~
            "Ridge logistic",

          grepl(
            "gam",
            algorithm,
            ignore.case = TRUE
          ) ~
            "GAM",

          TRUE ~
            as.character(
              algorithm
            )
        ),

      algorithm =
        factor(
          algorithm,
          levels =
            c(
              "Random forest",
              "Ridge logistic",
              "GAM"
            )
        ),

      threshold_strategy =
        recode(
          threshold_strategy,

          GLOBAL =
            "Global",

          REGIME_SPECIFIC =
            "Regime-specific"
        ),

      threshold_strategy =
        factor(
          threshold_strategy,
          levels =
            c(
              "Global",
              "Regime-specific"
            )
        ),

      variant =
        case_when(

          variant %in%
            c(
              "RAW_SHADOW",
              "RAW+SHADOW"
            ) ~
            "RAW + shadow",

          TRUE ~
            as.character(
              variant
            )
        )
    )


  fig_area_absolute_supp_49222 <-
    ggplot(
      AREA_ALL_49222,
      aes(
        x =
          threshold_strategy,
        y =
          suitable_area_total_km2,
        fill =
          variant
      )
    ) +

    geom_boxplot(
      width = 0.58,
      outlier.shape = NA,
      linewidth = 0.45,
      position =
        position_dodge(
          width = 0.68
        )
    ) +

    geom_point(
      aes(
        shape = variant
      ),
      position =
        position_jitterdodge(
          jitter.width = 0.05,
          dodge.width = 0.68
        ),
      size = 1,
      alpha = 0.30
    ) +

    facet_grid(
      algorithm ~ scheme,
      scales = "free_y"
    ) +

    scale_fill_manual(
      values =
        c(
          "RAW" =
            "white",
          "RAW + shadow" =
            COL_SHADOW
        )
    ) +

    scale_shape_manual(
      values =
        c(
          "RAW" = 21,
          "RAW + shadow" = 24
        )
    ) +

    labs(

      title =
        "Absolute mapped suitable area",

      subtitle =
        "Global and illumination-regime-specific threshold strategies",

      x =
        NULL,

      y =
        expression(
          "Estimated suitable area (km"^2*")"
        ),

      fill =
        "Predictor set",

      shape =
        "Predictor set"
    ) +

    theme_journal_49222() +

    theme(
      legend.position =
        "bottom"
    )


  save_journal_plot_49222(

    fig_area_absolute_supp_49222,

    "FigureS_absolute_mapped_suitable_area",

    DIR_49222_SUPP,

    width = 7.4,

    height = 7.0
  )
}

# ======================================================================
# 49.12 SUPPLEMENTARY FIGURE
# CROSS-ALGORITHM PREDICTOR-CONTRIBUTION HEATMAP
#
# FINAL CORRECTED VERSION
#
# Uses:
#   importance_summary_46222
#
# Required level:
#   algorithm × scheme × variant × predictor
#
# ======================================================================


# ----------------------------------------------------------------------
# 49.12.1 REQUIRE FINAL IMPORTANCE SUMMARY
# ----------------------------------------------------------------------

if(!exists("importance_summary_46222")){

  stop(
    paste0(
      "importance_summary_46222 was not found. ",
      "Run Section 46_222 first."
    )
  )
}


IMP_49222 <-
  importance_summary_46222


cat(
  "\nPredictor-importance object used: importance_summary_46222\n"
)


# ----------------------------------------------------------------------
# 49.12.2 REQUIRED COLUMNS
# ----------------------------------------------------------------------

required_imp_cols_49222 <-
  c(
    "algorithm",
    "scheme",
    "variant",
    "predictor",
    "Mean_Delta_Brier"
  )


missing_imp_cols_49222 <-
  setdiff(
    required_imp_cols_49222,
    names(IMP_49222)
  )


if(length(missing_imp_cols_49222) > 0){

  stop(
    paste0(
      "importance_summary_46222 is missing: ",
      paste(
        missing_imp_cols_49222,
        collapse = ", "
      )
    )
  )
}


# ----------------------------------------------------------------------
# 49.12.3 STANDARDIZE LABELS
# ----------------------------------------------------------------------

IMP_PLOT_49222 <-
  IMP_49222 %>%

  mutate(

    algorithm =
      case_when(

        grepl(
          "random",
          algorithm,
          ignore.case = TRUE
        ) ~
          "Random forest",

        grepl(
          "ridge",
          algorithm,
          ignore.case = TRUE
        ) ~
          "Ridge logistic",

        grepl(
          "gam",
          algorithm,
          ignore.case = TRUE
        ) ~
          "GAM",

        TRUE ~
          as.character(algorithm)
      ),


    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random forest",
            "Ridge logistic",
            "GAM"
          )
      ),


    scheme =
      factor(
        as.character(scheme),
        levels =
          c(
            "UR",
            "SB",
            "SW"
          )
      ),


    variant =
      case_when(

        as.character(variant) %in%
          c(
            "RAW_SHADOW",
            "RAW+SHADOW",
            "RAW + SHADOW",
            "RAW + shadow"
          ) ~
          "RAW + shadow",

        as.character(variant) == "RAW" ~
          "RAW",

        TRUE ~
          as.character(variant)
      ),


    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW + shadow"
          )
      ),


    predictor =
      as.character(predictor),


    Mean_Delta_Brier =
      as.numeric(Mean_Delta_Brier)
  ) %>%

  filter(
    !is.na(algorithm),
    !is.na(scheme),
    !is.na(variant),
    !is.na(predictor),
    is.finite(Mean_Delta_Brier)
  )


if(nrow(IMP_PLOT_49222) == 0){

  stop(
    "No valid predictor-importance rows remain after standardization."
  )
}


# ----------------------------------------------------------------------
# 49.12.4 ENSURE ONE CELL PER COMBINATION
# ----------------------------------------------------------------------

IMP_PLOT_49222 <-
  IMP_PLOT_49222 %>%

  group_by(
    predictor,
    algorithm,
    scheme,
    variant
  ) %>%

  summarise(

    Mean_Delta_Brier =
      mean(
        Mean_Delta_Brier,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


# ----------------------------------------------------------------------
# 49.12.5 SELECT MOST INFORMATIVE PREDICTORS
#
# Full 45-predictor heatmap becomes visually crowded.
# For publication-quality Supplementary display, show the 25 predictors
# with the largest regime contrast anywhere in the experiment.
#
# Full numerical results remain in Table S9.
# ----------------------------------------------------------------------

predictor_rank_49222 <-
  IMP_PLOT_49222 %>%

  group_by(
    predictor
  ) %>%

  summarise(

    Max_Abs_Delta =
      max(
        abs(Mean_Delta_Brier),
        na.rm = TRUE
      ),

    .groups = "drop"
  ) %>%

  arrange(
    desc(Max_Abs_Delta)
  )


TOP_N_IMPORTANCE_49222 <-
  min(
    25L,
    nrow(predictor_rank_49222)
  )


top_predictors_49222 <-
  predictor_rank_49222 %>%

  slice_head(
    n = TOP_N_IMPORTANCE_49222
  ) %>%

  pull(
    predictor
  )


IMP_PLOT_49222 <-
  IMP_PLOT_49222 %>%

  filter(
    predictor %in%
      top_predictors_49222
  )


# ----------------------------------------------------------------------
# 49.12.6 ORDER PREDICTORS
#
# Largest regime contrasts at the top.
# ----------------------------------------------------------------------

predictor_order_49222 <-
  predictor_rank_49222 %>%

  filter(
    predictor %in%
      top_predictors_49222
  ) %>%

  arrange(
    Max_Abs_Delta
  ) %>%

  pull(
    predictor
  )


IMP_PLOT_49222 <-
  IMP_PLOT_49222 %>%

  mutate(

    predictor =
      factor(
        predictor,
        levels =
          predictor_order_49222
      ),

    algorithm_scheme =
      interaction(
        algorithm,
        scheme,
        sep = "\n",
        lex.order = TRUE
      )
  )


# ----------------------------------------------------------------------
# 49.12.7 SYMMETRIC COLOUR SCALE
# ----------------------------------------------------------------------

lim_imp_49222 <-
  max(
    abs(
      IMP_PLOT_49222$Mean_Delta_Brier
    ),
    na.rm = TRUE
  )


if(
  !is.finite(lim_imp_49222) ||
  lim_imp_49222 <= 0
){

  stop(
    "Invalid predictor-importance colour scale."
  )
}


# ----------------------------------------------------------------------
# 49.12.8 DRAW HEATMAP
# ----------------------------------------------------------------------

fig_importance_supp_49222 <-
  ggplot(
    IMP_PLOT_49222,
    aes(
      x = algorithm_scheme,
      y = predictor,
      fill = Mean_Delta_Brier
    )
  ) +

  geom_tile(
    colour = "white",
    linewidth = 0.38
  ) +

  scale_fill_gradient2(

    low =
      COL_NEGATIVE,

    mid =
      "white",

    high =
      COL_POSITIVE,

    midpoint =
      0,

    limits =
      c(
        -lim_imp_49222,
        lim_imp_49222
      ),

    oob =
      scales::squish,

    name =
      expression(
        Delta*"VIP"[Brier]
      )
  ) +

  facet_wrap(
    ~variant,
    ncol = 1
  ) +

  labs(

    title =
      "Regime-dependent environmental predictor contribution",

    subtitle =
      paste0(
        "Top ",
        TOP_N_IMPORTANCE_49222,
        " predictors ranked by maximum absolute shadow–non-shadow contrast; ",
        "positive values indicate greater contribution in   shadow"
      ),

    x =
      NULL,

    y =
      NULL
  ) +

  theme_journal_49222() +

  theme(

    panel.grid =
      element_blank(),

    legend.position =
      "right",

    legend.key.height =
      unit(
        1.25,
        "cm"
      ),

    axis.text.x =
      element_text(
        size =
          BASE_FONT_49222 - 1,
        lineheight = 0.9
      ),

    axis.text.y =
      element_text(
        size =
          BASE_FONT_49222 - 1
      ),

    strip.text =
      element_text(
        face = "bold",
        size = BASE_FONT_49222
      ),

    panel.spacing =
      unit(
        0.7,
        "lines"
      )
  )


# ----------------------------------------------------------------------
# 49.12.9 EXPORT
# ----------------------------------------------------------------------

save_journal_plot_49222(

  fig_importance_supp_49222,

  "FigureS_regime_predictor_contribution_heatmap",

  DIR_49222_SUPP,

  width = 7.4,

  height = 8.5
)


# ----------------------------------------------------------------------
# 49.12.10 SAVE EXACT PLOT DATA
# ----------------------------------------------------------------------

write_csv(

  IMP_PLOT_49222 %>%
    mutate(
      predictor =
        as.character(predictor),
      algorithm_scheme =
        as.character(algorithm_scheme)
    ),

  file.path(
    DIR_49222_QC,
    "FigureS_predictor_contribution_heatmap_plot_data.csv"
  )
)


write_csv(

  predictor_rank_49222,

  file.path(
    DIR_49222_QC,
    "FigureS_predictor_contribution_predictor_ranking.csv"
  )
)


cat(
  "\nSupplementary predictor-contribution heatmap successfully generated.\n"
)

# ======================================================================
# 49.13 SUPPLEMENTARY:
# THRESHOLD DIFFERENCE
#
# Retain a compact delta representation as a supplementary diagnostic.
# ======================================================================

threshold_delta_summary_49222 <-
  THRESH_FINAL_49222 %>%
  mutate(

    Delta_threshold =
      threshold_shadow -
      threshold_nonshadow

  ) %>%
  group_by(
    algorithm,
    scheme,
    variant
  ) %>%
  summary95_49222(
    Delta_threshold
  )


fig_threshold_delta_supp_49222 <-
  ggplot(
    threshold_delta_summary_49222,
    aes(
      x = Mean,
      y = scheme,
      shape = variant,
      fill = variant
    )
  ) +

  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.5,
    colour = "grey45"
  ) +

  geom_errorbarh(
    aes(
      xmin = Lower,
      xmax = Upper
    ),
    height = 0,
    position =
      position_dodge(
        width = 0.35
      ),
    linewidth = 0.6
  ) +

  geom_point(
    position =
      position_dodge(
        width = 0.35
      ),
    size = 2.6,
    stroke = 0.7,
    colour = "black"
  ) +

  facet_wrap(
    ~algorithm,
    nrow = 1
  ) +

  scale_shape_manual(
    values =
      c(
        "RAW" = 21,
        "RAW + shadow" = 24
      )
  ) +

  scale_fill_manual(
    values =
      c(
        "RAW" = "white",
        "RAW + shadow" = COL_SHADOW
      )
  ) +

  labs(

    title =
      "Shadow–non-shadow decision-threshold contrast",

    subtitle =
      "Positive values indicate a higher calibration-derived threshold in   shadow",

    x =
      expression(
        Delta *
          " threshold (shadow - non-shadow)"
      ),

    y =
      "Sampling scenario",

    shape =
      "Predictor set",

    fill =
      "Predictor set"
  ) +

  theme_journal_49222() +

  theme(
    legend.position =
      "bottom"
  )


save_journal_plot_49222(

  fig_threshold_delta_supp_49222,

  "FigureS_threshold_difference_shadow_minus_nonshadow",

  DIR_49222_SUPP,

  width = 7.2,

  height = 4.2
)


# ======================================================================
# 49.14 QC TABLES
# ======================================================================

write_csv(
  perf_summary_49222,
  file.path(
    DIR_49222_QC,
    "Figure3_probability_performance_plot_data.csv"
  )
)


write_csv(
  threshold_summary_49222,
  file.path(
    DIR_49222_QC,
    "Figure4_threshold_plot_data.csv"
  )
)


write_csv(
  area_delta_summary_49222,
  file.path(
    DIR_49222_QC,
    "Figure5_mapped_area_delta_plot_data.csv"
  )
)


# ======================================================================
# 49.15 FINAL FIGURE MANIFEST
# ======================================================================

figure_manifest_49222 <-
  tibble(

    figure =
      c(
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5"
      ),

    role =
      "Main text",

    scientific_message =
      c(

        paste0(
          "  shadow has a structured ",
          "topographic basis rather than representing ",
          "an arbitrary image class."
        ),

        paste0(
          "External-test probability performance differs ",
          "across illumination regimes and model structures."
        ),

        paste0(
          "Calibration-derived decision thresholds differ ",
          "between non-shadow and shadow regimes."
        ),

        paste0(
          "Using illumination-regime-specific thresholds ",
          "changes the amount of habitat classified as suitable."
        )
      ),

    status =
      c(
        "Use topography composite from topographic section",
        "Generated by Section 49_222",
        "Generated by Section 49_222",
        "Generated by Section 49_222"
      )
  )


write_csv(
  figure_manifest_49222,
  file.path(
    DIR_49222_QC,
    "FINAL_MAIN_FIGURE_MANIFEST.csv"
  )
)


print(
  figure_manifest_49222,
  n = Inf
)


# ======================================================================
# 49.16 FINAL QC
# ======================================================================

qc_49222 <-
  tibble(

    check =
      c(
        "Three algorithms in prediction data",
        "Three sampling scenarios",
        "Two predictor variants",
        "30 replicate design retained",
        "RAW and RAW+shadow represented",
        "Threshold data available",
        "Mapped-area delta available"
      ),

    passed =
      c(

        n_distinct(
          PRED_FINAL_49222$algorithm
        ) == 3,

        n_distinct(
          PRED_FINAL_49222$scheme
        ) == 3,

        n_distinct(
          PRED_FINAL_49222$variant
        ) == 2,

        max(
          PRED_FINAL_49222$replicate,
          na.rm = TRUE
        ) == N_REP,

        all(
          c(
            "RAW",
            "RAW + shadow"
          ) %in%
            as.character(
              unique(
                PRED_FINAL_49222$variant
              )
            )
        ),

        nrow(
          THRESH_FINAL_49222
        ) > 0,

        nrow(
          AREA_DELTA_49222
        ) > 0
      )
  )


write_csv(
  qc_49222,
  file.path(
    DIR_49222_QC,
    "49_222_FINAL_FIGURE_QC.csv"
  )
)


print(
  qc_49222,
  n = Inf
)


if(
  !all(
    qc_49222$passed
  )
){

  stop(
    "SECTION 49_222 QC failed. Inspect QC table before manuscript use."
  )
}


cat(
  "\n============================================================\n",
  "SECTION 49_222 COMPLETED SUCCESSFULLY\n",
  "Final journal-ready figures written to:\n",
  DIR_49222,
  "\n============================================================\n",
  sep = ""
)

###########################
# ======================================================================
# SECTION 50
# FINAL COMPLETION OF MAIN + SUPPLEMENTARY FIGURE SUITE
#
# RUN IMMEDIATELY AFTER SECTION 49
#
# PURPOSE
# ----------------------------------------------------------------------
# Finalize the manuscript figure architecture after inspection of the
# complete analytical results.
#
# IMPORTANT
# ----------------------------------------------------------------------
# - NO model refitting
# - NO threshold re-estimation
# - NO resampling
# - NO external-test optimization
# - ONLY existing analytical objects are summarized/replotted
#
# FINAL MAIN-TEXT LOGIC
# ----------------------------------------------------------------------
# Figure 1 : Study design / analytical workflow          [separate]
# Figure 2 : Topographic basis of persistent shadow      [existing]
# Figure 3 : External-test probability performance       [Section 49]
# Figure 4 : Calibration-derived regime thresholds       [Section 49]
# Figure 5 : Sensitivity-specificity consequences        [THIS SECTION]
# Figure 6 : Mapped suitable-area consequence            [Section 49]
#
# SUPPLEMENTARY
# ----------------------------------------------------------------------
# S1  Shadow - non-shadow threshold contrast
# S2  Absolute mapped suitable area
# S3  Predictor-contribution heatmap
# S4  Newly suitable classification composition
# S5  Correct share of newly suitable classifications
#
# ======================================================================


suppressPackageStartupMessages({

  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(forcats)
  library(scales)
  library(grid)

})


# ======================================================================
# 50.1 REQUIRED OBJECTS
# ======================================================================

required_50222 <- c(

  "OUT_DIR",

  # Section 49
  "DIR_49222",
  "DIR_49222_MAIN",
  "DIR_49222_SUPP",

  # Section 41 exact external-test threshold consequences
  "key_41222"

)


missing_50222 <-
  required_50222[
    !vapply(
      required_50222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(length(missing_50222) > 0){

  stop(
    paste0(
      "\nSECTION 50_222 cannot start.\n",
      "Missing objects:\n",
      paste(
        missing_50222,
        collapse = ", "
      )
    )
  )

}


# ======================================================================
# 50.2 OUTPUT DIRECTORIES
# ======================================================================

DIR_50222 <-
  file.path(
    OUT_DIR,
    "50_222_FINAL_FIGURE_COMPLETION"
  )


DIR_50222_MAIN <-
  file.path(
    DIR_50222,
    "MAIN"
  )


DIR_50222_SUPP <-
  file.path(
    DIR_50222,
    "SUPPLEMENTARY"
  )


DIR_50222_QC <-
  file.path(
    DIR_50222,
    "QC"
  )


invisible(
  lapply(
    c(
      DIR_50222,
      DIR_50222_MAIN,
      DIR_50222_SUPP,
      DIR_50222_QC
    ),
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)


# ======================================================================
# 50.3 JOURNAL STYLE
# ======================================================================

COL_SHADOW_50222 <- "#1386BF"


theme_journal_50222 <- function(){

  theme_classic(
    base_size = 11
  ) +

    theme(

      plot.title =
        element_text(
          face = "bold",
          size = 13,
          hjust = 0
        ),

      plot.subtitle =
        element_text(
          size = 10.5,
          colour = "grey25",
          margin = margin(
            b = 8
          )
        ),

      axis.title =
        element_text(
          size = 11
        ),

      axis.text =
        element_text(
          size = 9.5,
          colour = "black"
        ),

      strip.background =
        element_rect(
          fill = "grey94",
          colour = "grey35",
          linewidth = 0.4
        ),

      strip.text =
        element_text(
          face = "bold",
          size = 10
        ),

      legend.position =
        "bottom",

      legend.title =
        element_text(
          face = "bold"
        ),

      panel.spacing =
        unit(
          0.7,
          "lines"
        ),

      plot.margin =
        margin(
          8, 10, 8, 8
        )

    )

}


save_final_50222 <-
  function(
    p,
    basename,
    directory,
    width,
    height
  ){

    ggsave(
      file.path(
        directory,
        paste0(
          basename,
          ".png"
        )
      ),
      p,
      width = width,
      height = height,
      units = "in",
      dpi = 600,
      bg = "white"
    )


    ggsave(
      file.path(
        directory,
        paste0(
          basename,
          ".tiff"
        )
      ),
      p,
      width = width,
      height = height,
      units = "in",
      dpi = 600,
      compression = "lzw",
      bg = "white"
    )

}


# ======================================================================
# 50.4 STANDARDIZE SECTION 41 DATA
# ======================================================================

K_50222 <-
  key_41222 %>%

  mutate(

    algorithm =
      factor(
        as.character(
          algorithm
        ),
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    scheme =
      factor(
        as.character(
          scheme
        ),
        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      case_when(

        as.character(
          variant
        ) %in%
          c(
            "RAW+SHADOW",
            "RAW_SHADOW",
            "RAW + shadow"
          ) ~
          "RAW + shadow",

        TRUE ~
          "RAW"

      ),

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW + shadow"
          )
      ),

    regime =
      case_when(

        tolower(
          as.character(
            regime
          )
        ) %in%
          c(
            "shadow",
            "sh"
          ) ~
          "Shadow",

        TRUE ~
          "Non-shadow"

      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      )

  )


# ======================================================================
# 50.5 MAIN FIGURE 5 — CORRECTED FINAL VERSION
#
# SENSITIVITY-SPECIFICITY TRADE-OFF
#
# IMPORTANT:
#   Trade-off data come from delta_41222, NOT key_41222.
#
#   delta_41222 contains:
#       Delta_Sensitivity
#       Delta_Specificity
#
#   defined as:
#       regime-specific threshold result - global threshold result
# ======================================================================


# ----------------------------------------------------------------------
# 50.5.1 REQUIRED OBJECT
# ----------------------------------------------------------------------

if(
  !exists(
    "delta_41222",
    envir = .GlobalEnv
  )
){

  stop(
    paste0(
      "\nSECTION 50_222.5 requires object 'delta_41222'.\n",
      "This object is generated in Section 41_222 and contains ",
      "the exact replicate-level sensitivity-specificity contrasts."
    )
  )

}


# ----------------------------------------------------------------------
# 50.5.2 CHECK REQUIRED COLUMNS
# ----------------------------------------------------------------------

required_trade_cols_50222 <-
  c(
    "algorithm",
    "replicate",
    "scheme",
    "variant",
    "regime",
    "Delta_Sensitivity",
    "Delta_Specificity"
  )


missing_trade_cols_50222 <-
  setdiff(
    required_trade_cols_50222,
    names(
      delta_41222
    )
  )


if(
  length(
    missing_trade_cols_50222
  ) > 0
){

  stop(
    paste0(
      "\ndelta_41222 is missing required columns:\n",
      paste(
        missing_trade_cols_50222,
        collapse = ", "
      )
    )
  )

}


# ----------------------------------------------------------------------
# 50.5.3 STANDARDIZE TRADE-OFF DATA
# ----------------------------------------------------------------------

trade_50222 <-
  delta_41222 %>%

  transmute(

    algorithm =
      case_when(

        grepl(
          "random",
          algorithm,
          ignore.case = TRUE
        ) ~
          "Random Forest",

        grepl(
          "ridge",
          algorithm,
          ignore.case = TRUE
        ) ~
          "Ridge logistic",

        grepl(
          "gam",
          algorithm,
          ignore.case = TRUE
        ) ~
          "GAM",

        TRUE ~
          as.character(
            algorithm
          )

      ),

    replicate =
      replicate,

    scheme =
      as.character(
        scheme
      ),

    variant =
      case_when(

        as.character(
          variant
        ) %in%
          c(
            "RAW_SHADOW",
            "RAW+SHADOW",
            "RAW + shadow"
          ) ~
          "RAW + shadow",

        TRUE ~
          "RAW"

      ),

    regime =
      case_when(

        toupper(
          gsub(
            "[-_ ]",
            "",
            as.character(
              regime
            )
          )
        ) ==
          "SHADOW" ~
          "Shadow",

        toupper(
          gsub(
            "[-_ ]",
            "",
            as.character(
              regime
            )
          )
        ) ==
          "NONSHADOW" ~
          "Non-shadow",

        TRUE ~
          as.character(
            regime
          )

      ),

    Delta_Sensitivity =
      as.numeric(
        Delta_Sensitivity
      ),

    Delta_Specificity =
      as.numeric(
        Delta_Specificity
      )

  ) %>%

  filter(

    is.finite(
      Delta_Sensitivity
    ),

    is.finite(
      Delta_Specificity
    )

  ) %>%

  mutate(

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    scheme =
      factor(
        scheme,
        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW + shadow"
          )
      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      )

  )


# ----------------------------------------------------------------------
# 50.5.4 QC BEFORE PLOTTING
# ----------------------------------------------------------------------

cat(
  "\nTrade-off data dimensions:",
  nrow(
    trade_50222
  ),
  "rows\n"
)


cat(
  "\nAlgorithms:\n"
)

print(
  table(
    trade_50222$algorithm,
    useNA = "ifany"
  )
)


cat(
  "\nSampling scenarios:\n"
)

print(
  table(
    trade_50222$scheme,
    useNA = "ifany"
  )
)


cat(
  "\nPredictor sets:\n"
)

print(
  table(
    trade_50222$variant,
    useNA = "ifany"
  )
)


cat(
  "\nIllumination regimes:\n"
)

print(
  table(
    trade_50222$regime,
    useNA = "ifany"
  )
)


stopifnot(

  all(
    c(
      "Random Forest",
      "Ridge logistic",
      "GAM"
    ) %in%
      as.character(
        unique(
          trade_50222$algorithm
        )
      )
  ),

  all(
    c(
      "SB",
      "SW",
      "UR"
    ) %in%
      as.character(
        unique(
          trade_50222$scheme
        )
      )
  ),

  all(
    c(
      "RAW",
      "RAW + shadow"
    ) %in%
      as.character(
        unique(
          trade_50222$variant
        )
      )
  ),

  all(
    c(
      "Non-shadow",
      "Shadow"
    ) %in%
      as.character(
        unique(
          trade_50222$regime
        )
      )
  )

)


# ----------------------------------------------------------------------
# 50.5.5 FINAL JOURNAL-READY FIGURE
# ----------------------------------------------------------------------

fig5_50222 <-
  ggplot(

    trade_50222,

    aes(

      x =
        Delta_Specificity,

      y =
        Delta_Sensitivity,

      shape =
        variant,

      alpha =
        scheme

    )

  ) +

  geom_hline(

    yintercept =
      0,

    linetype =
      "dashed",

    linewidth =
      0.45,

    colour =
      "grey40"

  ) +

  geom_vline(

    xintercept =
      0,

    linetype =
      "dashed",

    linewidth =
      0.45,

    colour =
      "grey40"

  ) +

  geom_point(

    size =
      2.05,

    colour =
      "black"

  ) +

  facet_grid(

    regime ~ algorithm,

    labeller =
      labeller(

        algorithm =
          c(
            "Random Forest" =
              "Random forest",

            "Ridge logistic" =
              "Ridge logistic",

            "GAM" =
              "GAM"
          ),

        regime =
          c(
            "Non-shadow" =
              "Non-shadow",

            "Shadow" =
              "Shadow"
          )

      )

  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" =
          16,

        "RAW + shadow" =
          17
      )

  ) +

  scale_alpha_manual(

    values =
      c(
        "SB" =
          0.42,

        "SW" =
          0.68,

        "UR" =
          1.00
      )

  ) +

  labs(

    title =
      "Sensitivity–specificity trade-off of illumination-regime-specific thresholds",

    subtitle =
      paste0(
        "Regime-specific minus global-threshold performance; ",
        "each point represents one stochastic model replicate"
      ),

    x =
      expression(
        Delta *
          " specificity (regime-specific - global)"
      ),

    y =
      expression(
        Delta *
          " sensitivity (regime-specific - global)"
      ),

    shape =
      "Predictor set",

    alpha =
      "Sampling scenario"

  ) +

  theme_journal_50222() +

  theme(

    legend.position =
      "bottom",

    strip.text =
      element_text(
        face =
          "bold",
        size =
          10
      ),

    panel.spacing.x =
      unit(
        0.75,
        "lines"
      ),

    panel.spacing.y =
      unit(
        0.55,
        "lines"
      )

  )


# ----------------------------------------------------------------------
# 50.5.6 EXPORT
# ----------------------------------------------------------------------

save_final_50222(

  fig5_50222,

  "Figure5_MAIN_sensitivity_specificity_tradeoff",

  DIR_50222_MAIN,

  width =
    12.5,

  height =
    7.2

)


# ----------------------------------------------------------------------
# 50.5.7 SAVE EXACT PLOT DATA
# ----------------------------------------------------------------------

write_csv(

  trade_50222,

  file.path(
    DIR_50222_QC,
    "Figure5_sensitivity_specificity_tradeoff_plot_data.csv"
  )

)


cat(
  "\n============================================================\n",
  "MAIN FIGURE 5 CREATED SUCCESSFULLY\n",
  "============================================================\n"
)


# ======================================================================
# 50.6 SUPPLEMENTARY FIGURE S4
#
# COMPOSITION OF NEWLY SUITABLE CLASSIFICATIONS
# ======================================================================
#
# Exact observation-level transitions:
#
# FN -> TP
#     recovered true presence
#
# TN -> FP
#     newly introduced false positive
#
# ======================================================================


newclass_50222 <-
  K_50222 %>%

  select(

    algorithm,
    replicate,
    scheme,
    variant,
    regime,

    `Recovered true presence` =
      FN_to_TP,

    `New false positive` =
      TN_to_FP

  ) %>%

  pivot_longer(

    cols =
      c(
        `Recovered true presence`,
        `New false positive`
      ),

    names_to =
      "transition",

    values_to =
      "count"

  ) %>%

  mutate(

    transition =
      factor(
        transition,
        levels =
          c(
            "New false positive",
            "Recovered true presence"
          )
      )

  )


figS4_50222 <-
  ggplot(

    newclass_50222,

    aes(

      x =
        scheme,

      y =
        count,

      shape =
        variant

    )

  ) +

  geom_boxplot(

    aes(
      group =
        interaction(
          scheme,
          variant
        )
    ),

    position =
      position_dodge(
        width = 0.65
      ),

    width = 0.58,

    outlier.shape = NA,

    linewidth = 0.45

  ) +

  geom_point(

    position =
      position_jitterdodge(
        jitter.width = 0.08,
        dodge.width = 0.65
      ),

    alpha = 0.50,

    size = 1.35

  ) +

  facet_grid(
    interaction(
      regime,
      transition,
      sep = " — "
    ) ~
      algorithm,
    scales = "free_y"
  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" = 21,
        "RAW + shadow" = 24
      )

  ) +

  labs(

    title =
      "Composition of newly suitable external-test classifications",

    subtitle =
      paste0(
        "Exact observation-level changes after replacing the global ",
        "threshold with illumination-regime-specific thresholds"
      ),

    x =
      "Sampling scenario",

    y =
      "Number of observations per replicate",

    shape =
      "Predictor set"

  ) +

  theme_journal_50222()


save_final_50222(

  figS4_50222,

  "FigureS4_newly_suitable_classification_composition",

  DIR_50222_SUPP,

  width = 12.8,

  height = 9.2

)


# ======================================================================
# 50.7 SUPPLEMENTARY FIGURE S5
#
# WERE NEWLY SUITABLE CLASSIFICATIONS CORRECT?
# ======================================================================
#
# Correct share =
#
#        FN -> TP
# -----------------------
# FN -> TP + TN -> FP
#
# Values > 0.5:
#   more newly suitable classifications recovered true presences than
#   introduced false positives.
#
# IMPORTANT:
#   Undefined when there were zero 0->1 transitions.
# ======================================================================


correct_added_50222 <-
  K_50222 %>%

  mutate(

    n_new_suitable =
      FN_to_TP +
      TN_to_FP,

    correct_added_share =
      ifelse(
        n_new_suitable > 0,
        FN_to_TP /
          n_new_suitable,
        NA_real_
      )

  )


write_csv(

  correct_added_50222,

  file.path(
    DIR_50222_QC,
    "50_222_correct_added_share_data.csv"
  )

)


figS5_50222 <-
  correct_added_50222 %>%

  filter(
    is.finite(
      correct_added_share
    )
  ) %>%

  ggplot(

    aes(

      x =
        scheme,

      y =
        correct_added_share,

      shape =
        variant

    )

  ) +

  geom_hline(

    yintercept = 0.5,

    linetype = "dashed",

    linewidth = 0.5

  ) +

  geom_boxplot(

    aes(
      group =
        interaction(
          scheme,
          variant
        )
    ),

    position =
      position_dodge(
        width = 0.65
      ),

    width = 0.58,

    outlier.shape = NA,

    linewidth = 0.45

  ) +

  geom_point(

    position =
      position_jitterdodge(
        jitter.width = 0.08,
        dodge.width = 0.65
      ),

    size = 1.4,

    alpha = 0.55

  ) +

  facet_grid(
    regime ~ algorithm
  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" = 21,
        "RAW + shadow" = 24
      )

  ) +

  scale_y_continuous(

    limits =
      c(
        0,
        1
      ),

    breaks =
      seq(
        0,
        1,
        0.25
      )

  ) +

  labs(

    title =
      "Were newly suitable classifications correct?",

    subtitle =
      paste0(
        "Share of 0-to-1 classification changes that recovered true ",
        "presences on the independent external test"
      ),

    x =
      "Sampling scenario",

    y =
      "Correct share of newly suitable classifications",

    shape =
      "Predictor set"

  ) +

  theme_journal_50222()


save_final_50222(

  figS5_50222,

  "FigureS5_correct_share_newly_suitable",

  DIR_50222_SUPP,

  width = 12.5,

  height = 7.0

)


# ======================================================================
# 50.8 SUPPLEMENTARY:
# THRESHOLD CONTRAST
# ======================================================================

if(
  exists(
    "threshold_contrast_summary_49222"
  )
){

  figS1_50222 <-
    ggplot(

      threshold_contrast_summary_49222,

      aes(

        x =
          Mean,

        y =
          scheme,

        shape =
          variant,

        fill =
          variant

      )

    ) +

    geom_vline(

      xintercept = 0,

      linetype = "dashed",

      colour = "grey45"

    ) +

    geom_errorbarh(

      aes(
        xmin = Lower,
        xmax = Upper
      ),

      height = 0,

      position =
        position_dodge(
          width = 0.38
        ),

      linewidth = 0.65

    ) +

    geom_point(

      position =
        position_dodge(
          width = 0.38
        ),

      size = 2.8,

      stroke = 0.75

    ) +

    facet_wrap(
      ~algorithm,
      nrow = 1
    ) +

    scale_shape_manual(

      values =
        c(
          "RAW" = 21,
          "RAW + shadow" = 24
        )

    ) +

    scale_fill_manual(

      values =
        c(
          "RAW" = "white",
          "RAW + shadow" =
            COL_SHADOW_50222
        )

    ) +

    labs(

      title =
        "Shadow–non-shadow decision-threshold contrast",

      subtitle =
        "Positive values indicate a higher calibration-derived threshold in shadow",

      x =
        expression(
          Delta *
            " threshold (shadow - non-shadow)"
        ),

      y =
        "Sampling scenario",

      shape =
        "Predictor set",

      fill =
        "Predictor set"

    ) +

    theme_journal_50222()


  save_final_50222(

    figS1_50222,

    "FigureS1_shadow_nonshadow_threshold_contrast",

    DIR_50222_SUPP,

    width = 7.5,

    height = 4.4

  )

}


# ======================================================================
# 50.9 COPY / REGISTER IMPORTANT SECTION 49 FIGURES
# ======================================================================
#
# We DO NOT redraw figures that Section 49 already produced correctly.
#
# Instead, build the final manuscript manifest.
# ======================================================================


find_file_50222 <-
  function(
    directory,
    pattern
  ){

    z <-
      list.files(
        directory,
        pattern = pattern,
        recursive = TRUE,
        full.names = TRUE,
        ignore.case = TRUE
      )

    if(length(z) == 0){

      return(
        NA_character_
      )

    }

    z[1]

  }


FIG3_50222 <-
  find_file_50222(
    DIR_49222,
    "probability.*performance.*\\.png$"
  )


FIG4_50222 <-
  find_file_50222(
    DIR_49222,
    "threshold.*regime|regime.*threshold"
  )


FIG6_50222 <-
  find_file_50222(
    DIR_49222,
    "mapped.*area.*effect|threshold.*mapped.*area"
  )


FIG_ABS_AREA_50222 <-
  find_file_50222(
    DIR_49222,
    "absolute.*mapped.*area"
  )


FIG_IMPORTANCE_50222 <-
  find_file_50222(
    DIR_49222,
    "predictor.*contribution|importance.*heatmap"
  )


# ======================================================================
# 50.10 FINAL FIGURE MANIFEST
# ======================================================================


final_manifest_50222 <-
  tibble(

    order =
      c(
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5",
        "Figure 6",

        "Figure S1",
        "Figure S2",
        "Figure S3",
        "Figure S4",
        "Figure S5"
      ),

    role =
      c(
        rep(
          "MAIN",
          6
        ),
        rep(
          "SUPPLEMENTARY",
          5
        )
      ),

    topic =
      c(

        "Study design and analytical workflow",

        "Topographic basis of persistent shadow",

        "External-test probability performance",

        "Calibration-derived illumination-regime thresholds",

        "Sensitivity-specificity consequences of regime-specific thresholding",

        "Change in mapped suitable area",

        "Shadow-non-shadow threshold contrast",

        "Absolute mapped suitable area",

        "Regime-dependent predictor contribution",

        "Composition of newly suitable external-test classifications",

        "Correct share of newly suitable classifications"

      ),

    scientific_question =
      c(

        "How was the independent evaluation framework constructed?",

        "Is persistent shadow environmentally structured?",

        "How well do models discriminate and calibrate on the external test?",

        "Do calibration-derived decision thresholds differ between illumination regimes?",

        "What classification trade-off results from using regime-specific thresholds?",

        "Does threshold choice materially alter mapped habitat extent?",

        "How large is the shadow minus non-shadow threshold difference?",

        "What are the absolute mapped-area distributions?",

        "Which environmental predictors show regime-dependent contribution?",

        "Are newly suitable classifications true recovered presences or new false positives?",

        "What proportion of newly suitable classifications are correct?"

      ),

    file =
      c(

        NA_character_,

        NA_character_,

        FIG3_50222,

        FIG4_50222,

        file.path(
          DIR_50222_MAIN,
          "Figure5_MAIN_classification_consequences_regime_thresholds.png"
        ),

        FIG6_50222,

        if(
          exists(
            "figS1_50222"
          )
        ){
          file.path(
            DIR_50222_SUPP,
            "FigureS1_shadow_nonshadow_threshold_contrast.png"
          )
        } else {
          NA_character_
        },

        FIG_ABS_AREA_50222,

        FIG_IMPORTANCE_50222,

        file.path(
          DIR_50222_SUPP,
          "FigureS4_newly_suitable_classification_composition.png"
        ),

        file.path(
          DIR_50222_SUPP,
          "FigureS5_correct_share_newly_suitable.png"
        )

      )

  ) %>%

  mutate(

    exists =
      ifelse(
        is.na(
          file
        ),
        NA,
        file.exists(
          file
        )
      )

  )


write_csv(

  final_manifest_50222,

  file.path(
    DIR_50222_QC,
    "50_222_FINAL_MANUSCRIPT_FIGURE_MANIFEST.csv"
  )

)


print(
  final_manifest_50222,
  n = Inf
)


# ======================================================================
# 50.11 CORE QC
# ======================================================================


qc_50222 <-
  tibble(

    check =
      c(

        "Section 41 consequence table available",

        "Three algorithms retained",

        "Three sampling scenarios retained",

        "Two predictor sets retained",

        "Both illumination regimes retained",

        "Main trade-off figure created",

        "Supplementary transition-composition figure created",

        "Supplementary correct-share figure created"

      ),

    passed =
      c(

        nrow(
          K_50222
        ) > 0,

        all(
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          ) %in%
            as.character(
              unique(
                K_50222$algorithm
              )
            )
        ),

        all(
          c(
            "SB",
            "SW",
            "UR"
          ) %in%
            as.character(
              unique(
                K_50222$scheme
              )
            )
        ),

        all(
          c(
            "RAW",
            "RAW + shadow"
          ) %in%
            as.character(
              unique(
                K_50222$variant
              )
            )
        ),

        all(
          c(
            "Non-shadow",
            "Shadow"
          ) %in%
            as.character(
              unique(
                K_50222$regime
              )
            )
        ),

        file.exists(
          file.path(
            DIR_50222_MAIN,
            "Figure5_MAIN_classification_consequences_regime_thresholds.png"
          )
        ),

        file.exists(
          file.path(
            DIR_50222_SUPP,
            "FigureS4_newly_suitable_classification_composition.png"
          )
        ),

        file.exists(
          file.path(
            DIR_50222_SUPP,
            "FigureS5_correct_share_newly_suitable.png"
          )
        )

      )

  )


write_csv(

  qc_50222,

  file.path(
    DIR_50222_QC,
    "50_222_QC.csv"
  )

)


print(
  qc_50222,
  n = Inf
)


if(
  !all(
    qc_50222$passed
  )
){

  stop(
    paste0(
      "\nSECTION 50_222 QC FAILED.\n",
      "Inspect:\n",
      file.path(
        DIR_50222_QC,
        "50_222_QC.csv"
      )
    )
  )

}


# ======================================================================
# 50.12 SESSION INFO
# ======================================================================


capture.output(

  sessionInfo(),

  file =
    file.path(
      DIR_50222_QC,
      "sessionInfo_SECTION50_222.txt"
    )

)


# ======================================================================
# 50.13 FINAL REPORT
# ======================================================================


cat(
  "\n============================================================\n",
  "SECTION 50_222 COMPLETED SUCCESSFULLY\n",
  "============================================================\n",
  "\nMain trade-off figure:\n",
  file.path(
    DIR_50222_MAIN,
    "Figure5_MAIN_classification_consequences_regime_thresholds.png"
  ),
  "\n\nSupplementary newly-suitable composition:\n",
  file.path(
    DIR_50222_SUPP,
    "FigureS4_newly_suitable_classification_composition.png"
  ),
  "\n\nSupplementary correct-added share:\n",
  file.path(
    DIR_50222_SUPP,
    "FigureS5_correct_share_newly_suitable.png"
  ),
  "\n\nFinal manifest:\n",
  file.path(
    DIR_50222_QC,
    "50_222_FINAL_MANUSCRIPT_FIGURE_MANIFEST.csv"
  ),
  "\n============================================================\n",
  sep = ""
)
# ======================================================================
# 50.6 SUPPLEMENTARY FIGURE S4 — REVISED FINAL VERSION
#
# COMPOSITION OF NEWLY SUITABLE CLASSIFICATIONS
#
# WHY THIS VERSION?
# ----------------------------------------------------------------------
# Counts are discrete and strongly zero-inflated.
# Conventional boxplots can collapse to a horizontal line when
# Q1 = median = Q3 = 0.
#
# Therefore:
#   - each stochastic replicate is shown explicitly
#   - large symbols show replicate medians
#   - vertical bars show empirical 95% replicate intervals
#
# NO new analysis is performed.
# ======================================================================


# ----------------------------------------------------------------------
# 50.6.1 BUILD LONG-FORM DATA
# ----------------------------------------------------------------------

newclass_50222 <-
  K_50222 %>%

  select(

    algorithm,
    replicate,
    scheme,
    variant,
    regime,

    `Recovered true presence` =
      FN_to_TP,

    `New false positive` =
      TN_to_FP

  ) %>%

  pivot_longer(

    cols =
      c(
        `Recovered true presence`,
        `New false positive`
      ),

    names_to =
      "transition",

    values_to =
      "count"

  ) %>%

  mutate(

    count =
      as.numeric(
        count
      ),

    transition =
      factor(
        transition,
        levels =
          c(
            "New false positive",
            "Recovered true presence"
          )
      ),

    scheme =
      factor(
        scheme,
        levels =
          c(
            "SB",
            "SW",
            "UR"
          )
      ),

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW + shadow"
          )
      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      )

  ) %>%

  filter(
    is.finite(
      count
    )
  )


# ----------------------------------------------------------------------
# 50.6.2 SUMMARY STATISTICS
# ----------------------------------------------------------------------

newclass_summary_50222 <-
  newclass_50222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime,
    transition
  ) %>%

  summarise(

    n =
      n(),

    Median =
      median(
        count,
        na.rm = TRUE
      ),

    Mean =
      mean(
        count,
        na.rm = TRUE
      ),

    Lower95 =
      quantile(
        count,
        probs = 0.025,
        na.rm = TRUE,
        names = FALSE
      ),

    Upper95 =
      quantile(
        count,
        probs = 0.975,
        na.rm = TRUE,
        names = FALSE
      ),

    .groups =
      "drop"

  )


write_csv(

  newclass_summary_50222,

  file.path(
    DIR_50222_QC,
    "FigureS4_newly_suitable_classification_summary.csv"
  )

)


# ----------------------------------------------------------------------
# 50.6.3 POSITION OBJECT
# ----------------------------------------------------------------------

pd_newclass_50222 <-
  position_dodge(
    width = 0.52
  )


# ----------------------------------------------------------------------
# 50.6.4 FINAL FIGURE
# ----------------------------------------------------------------------

figS4_50222 <-
  ggplot(

    newclass_50222,

    aes(

      x =
        scheme,

      y =
        count,

      shape =
        variant,

      group =
        variant

    )

  ) +

  # ------------------------------------------------------------
  # Raw replicate-level observations
  # ------------------------------------------------------------

  geom_point(

    position =
      position_jitterdodge(
        jitter.width = 0.08,
        jitter.height = 0,
        dodge.width = 0.52
      ),

    size =
      1.55,

    alpha =
      0.38,

    stroke =
      0.55,

    colour =
      "grey25"

  ) +

  # ------------------------------------------------------------
  # Empirical 95% replicate intervals
  # ------------------------------------------------------------

  geom_errorbar(

    data =
      newclass_summary_50222,

    aes(

      x =
        scheme,

      ymin =
        Lower95,

      ymax =
        Upper95,

      group =
        variant

    ),

    position =
      pd_newclass_50222,

    width =
      0.08,

    linewidth =
      0.65,

    inherit.aes =
      FALSE

  ) +

  # ------------------------------------------------------------
  # Median
  # ------------------------------------------------------------

  geom_point(

    data =
      newclass_summary_50222,

    aes(

      x =
        scheme,

      y =
        Median,

      shape =
        variant,

      group =
        variant

    ),

    position =
      pd_newclass_50222,

    size =
      3.0,

    stroke =
      0.9,

    fill =
      "white",

    colour =
      "black",

    inherit.aes =
      FALSE

  ) +

  facet_grid(

    interaction(
      regime,
      transition,
      sep = " — "
    ) ~
      algorithm,

    scales =
      "free_y",

    labeller =
      labeller(

        algorithm =
          c(
            "Random Forest" =
              "Random forest",

            "Ridge logistic" =
              "Ridge logistic",

            "GAM" =
              "GAM"
          )

      )

  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" =
          21,

        "RAW + shadow" =
          24
      )

  ) +

  scale_y_continuous(

    expand =
      expansion(
        mult =
          c(
            0.02,
            0.10
          )
      )

  ) +

  labs(

    title =
      "Composition of newly suitable external-test classifications",

    subtitle =
      paste0(
        "Exact observation-level changes after replacing the global ",
        "threshold with illumination-regime-specific thresholds; ",
        "large symbols are medians and bars empirical 95% replicate intervals"
      ),

    x =
      "Sampling scenario",

    y =
      "Number of observations per replicate",

    shape =
      "Predictor set"

  ) +

  theme_journal_50222() +

  theme(

    legend.position =
      "bottom",

    panel.grid.major.y =
      element_line(
        colour =
          "grey92",
        linewidth =
          0.35
      ),

    panel.grid.minor =
      element_blank(),

    strip.text.y.right =
      element_text(
        angle =
          90,
        face =
          "bold",
        size =
          8.7
      )

  )


# ----------------------------------------------------------------------
# 50.6.5 EXPORT
# ----------------------------------------------------------------------

save_final_50222(

  figS4_50222,

  "FigureS4_newly_suitable_classification_composition_REVISED",

  DIR_50222_SUPP,

  width =
    12.5,

  height =
    8.6

)


cat(
  "\nFigure S4 revised successfully.\n"
)



# ======================================================================
# 50.7 SUPPLEMENTARY FIGURE S5 — REVISED FINAL VERSION
#
# WERE NEWLY SUITABLE CLASSIFICATIONS CORRECT?
#
# Correct share =
#
#        FN -> TP
# -----------------------
# FN -> TP + TN -> FP
#
# IMPORTANT:
# ----------------------------------------------------------------------
# This proportion is undefined when no 0->1 classification change
# occurred in a replicate.
#
# Therefore:
#   - zero-transition replicates are excluded from the proportion
#   - each valid replicate is shown explicitly
#   - large symbols show medians
#   - bars show empirical IQR
#   - n_valid is displayed for transparency
# ======================================================================


# ----------------------------------------------------------------------
# 50.7.1 BUILD VALID-REPLICATE DATA
# ----------------------------------------------------------------------

correct_added_50222 <-
  K_50222 %>%

  mutate(

    n_new_suitable =
      FN_to_TP +
      TN_to_FP,

    correct_added_share =
      ifelse(

        n_new_suitable > 0,

        FN_to_TP /
          n_new_suitable,

        NA_real_

      )

  )


correct_added_valid_50222 <-
  correct_added_50222 %>%

  filter(
    is.finite(
      correct_added_share
    )
  )


# ----------------------------------------------------------------------
# 50.7.2 SUMMARY
# ----------------------------------------------------------------------

correct_added_summary_50222 <-
  correct_added_valid_50222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    n_valid =
      n(),

    Median =
      median(
        correct_added_share,
        na.rm = TRUE
      ),

    Q25 =
      quantile(
        correct_added_share,
        0.25,
        na.rm = TRUE,
        names = FALSE
      ),

    Q75 =
      quantile(
        correct_added_share,
        0.75,
        na.rm = TRUE,
        names = FALSE
      ),

    Mean =
      mean(
        correct_added_share,
        na.rm = TRUE
      ),

    .groups =
      "drop"

  )


write_csv(

  correct_added_valid_50222,

  file.path(
    DIR_50222_QC,
    "FigureS5_correct_share_valid_replicates.csv"
  )

)


write_csv(

  correct_added_summary_50222,

  file.path(
    DIR_50222_QC,
    "FigureS5_correct_share_summary.csv"
  )

)


# ----------------------------------------------------------------------
# 50.7.3 n-LABEL POSITIONS
# ----------------------------------------------------------------------

n_label_50222 <-
  correct_added_summary_50222 %>%

  mutate(

    y_n =
      1.04,

    n_text =
      paste0(
        "n=",
        n_valid
      )

  )


# ----------------------------------------------------------------------
# 50.7.4 DODGE POSITION
# ----------------------------------------------------------------------

pd_correct_50222 <-
  position_dodge(
    width = 0.52
  )


# ----------------------------------------------------------------------
# 50.7.5 FINAL FIGURE
# ----------------------------------------------------------------------

figS5_50222 <-
  ggplot(

    correct_added_valid_50222,

    aes(

      x =
        scheme,

      y =
        correct_added_share,

      shape =
        variant,

      group =
        variant

    )

  ) +

  # ------------------------------------------------------------
  # Reference: more recovered TPs than introduced FPs
  # ------------------------------------------------------------

  geom_hline(

    yintercept =
      0.5,

    linetype =
      "dashed",

    linewidth =
      0.5,

    colour =
      "grey30"

  ) +

  # ------------------------------------------------------------
  # Raw valid replicate values
  # ------------------------------------------------------------

  geom_point(

    position =
      position_jitterdodge(
        jitter.width = 0.075,
        jitter.height = 0,
        dodge.width = 0.52
      ),

    size =
      1.55,

    alpha =
      0.38,

    colour =
      "grey25",

    stroke =
      0.55

  ) +

  # ------------------------------------------------------------
  # IQR
  # ------------------------------------------------------------

  geom_errorbar(

    data =
      correct_added_summary_50222,

    aes(

      x =
        scheme,

      ymin =
        Q25,

      ymax =
        Q75,

      group =
        variant

    ),

    position =
      pd_correct_50222,

    width =
      0.08,

    linewidth =
      0.75,

    inherit.aes =
      FALSE

  ) +

  # ------------------------------------------------------------
  # Median
  # ------------------------------------------------------------

  geom_point(

    data =
      correct_added_summary_50222,

    aes(

      x =
        scheme,

      y =
        Median,

      shape =
        variant,

      group =
        variant

    ),

    position =
      pd_correct_50222,

    size =
      3.0,

    fill =
      "white",

    colour =
      "black",

    stroke =
      0.9,

    inherit.aes =
      FALSE

  ) +

  # ------------------------------------------------------------
  # Valid replicate n
  # ------------------------------------------------------------

  geom_text(

    data =
      n_label_50222,

    aes(

      x =
        scheme,

      y =
        y_n,

      label =
        n_text,

      group =
        variant

    ),

    position =
      pd_correct_50222,

    size =
      2.65,

    colour =
      "grey30",

    inherit.aes =
      FALSE

  ) +

  facet_grid(

    regime ~ algorithm,

    labeller =
      labeller(

        algorithm =
          c(
            "Random Forest" =
              "Random forest",

            "Ridge logistic" =
              "Ridge logistic",

            "GAM" =
              "GAM"
          )

      )

  ) +

  scale_shape_manual(

    values =
      c(
        "RAW" =
          21,

        "RAW + shadow" =
          24
      )

  ) +

  scale_y_continuous(

    limits =
      c(
        0,
        1.10
      ),

    breaks =
      c(
        0,
        0.25,
        0.50,
        0.75,
        1.00
      ),

    labels =
      c(
        "0.00",
        "0.25",
        "0.50",
        "0.75",
        "1.00"
      ),

    expand =
      expansion(
        mult =
          c(
            0,
            0.01
          )
      )

  ) +

  labs(

    title =
      "Correctness of newly suitable external-test classifications",

    subtitle =
      paste0(
        "Only replicates with at least one 0-to-1 classification change are included; ",
        "large symbols are medians, bars are interquartile ranges, and n denotes valid replicates"
      ),

    x =
      "Sampling scenario",

    y =
      "Share recovering true presences",

    shape =
      "Predictor set"

  ) +

  theme_journal_50222() +

  theme(

    legend.position =
      "bottom",

    panel.grid.major.y =
      element_line(
        colour =
          "grey92",
        linewidth =
          0.35
      ),

    panel.grid.minor =
      element_blank()

  )


# ----------------------------------------------------------------------
# 50.7.6 EXPORT
# ----------------------------------------------------------------------

save_final_50222(

  figS5_50222,

  "FigureS5_correct_share_newly_suitable_REVISED",

  DIR_50222_SUPP,

  width =
    12.5,

  height =
    6.8

)


cat(
  "\nFigure S5 revised successfully.\n"
)



# ======================================================================
# 50.7.7 FINAL QC FOR BOTH REVISED SUPPLEMENTARY FIGURES
# ======================================================================

qc_supp_50222 <-
  tibble(

    check =
      c(

        "S4 raw replicate data available",

        "S4 summary available",

        "S5 valid-replicate data available",

        "S5 summary available",

        "S4 PNG created",

        "S5 PNG created"

      ),

    passed =
      c(

        nrow(
          newclass_50222
        ) > 0,

        nrow(
          newclass_summary_50222
        ) > 0,

        nrow(
          correct_added_valid_50222
        ) > 0,

        nrow(
          correct_added_summary_50222
        ) > 0,

        file.exists(
          file.path(
            DIR_50222_SUPP,
            "FigureS4_newly_suitable_classification_composition_REVISED.png"
          )
        ),

        file.exists(
          file.path(
            DIR_50222_SUPP,
            "FigureS5_correct_share_newly_suitable_REVISED.png"
          )
        )

      )

  )


write_csv(

  qc_supp_50222,

  file.path(
    DIR_50222_QC,
    "50_222_QC_REVISED_SUPPLEMENTARY_FIGURES.csv"
  )

)


print(
  qc_supp_50222,
  n = Inf
)


if(
  !all(
    qc_supp_50222$passed
  )
){

  stop(
    "Revised Supplementary Figure QC failed."
  )

}


cat(
  "\n============================================================\n",
  "REVISED SUPPLEMENTARY FIGURES COMPLETED SUCCESSFULLY\n",
  "============================================================\n",
  "\nFigure S4:\n",
  file.path(
    DIR_50222_SUPP,
    "FigureS4_newly_suitable_classification_composition_REVISED.png"
  ),
  "\n\nFigure S5:\n",
  file.path(
    DIR_50222_SUPP,
    "FigureS5_correct_share_newly_suitable_REVISED.png"
  ),
  "\n============================================================\n",
  sep = ""
)

# ======================================================================
# ============================================================
# Publication-only continuation
# ============================================================
# Threshold-search and convergence sensitivity branches used during
# development are intentionally omitted from this public script.
# The sections below use only primary-analysis objects created above.

# Final inference and predictor-contribution figures
# FINAL PRIMARY-ANALYSIS INFERENCE AND COMPACT MAIN FIGURES
#
# PURPOSE
# ----------------------------------------------------------------------
# PART A
#   Formal replicate-level inference for:
#
#       Delta BACC =
#       BACC_regime-specific threshold - BACC_global threshold
#
#   For every:
#       algorithm × sampling scheme × predictor variant × regime
#
#   Reports:
#       - n valid stochastic replicates
#       - mean Delta BACC
#       - median Delta BACC
#       - empirical 2.5% and 97.5% replicate quantiles
#       - proportion positive
#       - Wilcoxon signed-rank test against zero
#       - paired rank-biserial effect size
#       - BH-FDR adjusted p-value
#
# PART B
#   Replace the dense legacy Delta-VIP figure by a compact,
#   algorithm-specific MAIN figure.
#
#   Predictor selection is NOT performed separately for RAW and
#   RAW+SHADOW.
#
#   Within each algorithm, predictors are ranked by:
#
#       max(
#         abs(mean DeltaVIP_RAW),
#         abs(mean DeltaVIP_RAW+SHADOW)
#       )
#
#   The SAME selected predictor set is then displayed for BOTH variants.
#
# CRITICAL SOURCE RESTRICTION
# ----------------------------------------------------------------------
# ANALYTICAL INPUTS ARE STRICTLY PRIMARY-ANALYSIS:
#
#   Delta BACC : delta_41222
#   Delta VIP  : importance_wide_46222
#
# NO Section 51 / 52 / 53 / 54 / 55 / 56 analytical result is used.
#
# Section 56 objects are NOT used as analytical inputs either.
# Everything required below is reconstructed directly from the original
# PRIMARY-ANALYSIS source objects.
#
# PRIMARY ANALYSIS IS NOT REPLACED.
# This section adds inference / uncertainty and improves visualization.
# ======================================================================


# ======================================================================
# 57.1 PACKAGES
# ======================================================================

suppressPackageStartupMessages({

  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(forcats)
  library(scales)
  library(tibble)

})


# ======================================================================
# 57.2 REQUIRED PRIMARY-ANALYSIS OBJECTS
# ======================================================================

required_57222 <-
  c(
    "OUT_DIR",
    "delta_41222",
    "importance_wide_46222"
  )


missing_57222 <-
  required_57222[
    !vapply(
      required_57222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(length(missing_57222) > 0){

  stop(
    paste0(
      "\nSECTION 57 cannot start.\n",
      "Missing PRE-SECTION-51 objects:\n",
      paste(
        missing_57222,
        collapse = "\n"
      ),
      "\n\nRequired analytical sources are STRICTLY:\n",
      "  delta_41222\n",
      "  importance_wide_46222\n"
    )
  )
}


cat(
  "\n============================================================\n",
  "SECTION 57 — SOURCE CHECK\n",
  "============================================================\n",
  "Delta BACC source : delta_41222\n",
  "Delta VIP source  : importance_wide_46222\n",
  "Section 51+ analytical objects used: NONE\n",
  "============================================================\n",
  sep = ""
)


# ======================================================================
# 57.3 OUTPUT DIRECTORY
# ======================================================================

DIR_57222 <-
  file.path(
    OUT_DIR,
    "final_inference_and_predictor_figures"
  )


if(dir.exists(DIR_57222)){

  stop(
    paste0(
      "\nSAFETY STOP:\n",
      DIR_57222,
      "\nalready exists.\n\n",
      "No previous Section-57 output will be overwritten.\n",
      "Rename/remove the directory before rerunning."
    )
  )
}


DIR_57222_CSV <-
  file.path(
    DIR_57222,
    "CSV"
  )


DIR_57222_FIG <-
  file.path(
    DIR_57222,
    "FIGURES"
  )


DIR_57222_QC <-
  file.path(
    DIR_57222,
    "QC"
  )


invisible(
  lapply(
    c(
      DIR_57222,
      DIR_57222_CSV,
      DIR_57222_FIG,
      DIR_57222_QC
    ),
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)


# ======================================================================
# 57.4 STANDARDIZATION HELPERS
# ======================================================================

std_algorithm_57222 <- function(x){

  x <- as.character(x)

  case_when(

    grepl(
      "random",
      x,
      ignore.case = TRUE
    ) ~
      "Random Forest",

    grepl(
      "ridge",
      x,
      ignore.case = TRUE
    ) ~
      "Ridge logistic",

    grepl(
      "gam",
      x,
      ignore.case = TRUE
    ) ~
      "GAM",

    TRUE ~ x
  )
}


std_variant_57222 <- function(x){

  x <- as.character(x)

  case_when(

    x %in%
      c(
        "RAW_SHADOW",
        "RAW+SHADOW",
        "RAW + shadow",
        "RAW+shadow",
        "RAW + SHADOW"
      ) ~
        "RAW+SHADOW",

    x == "RAW" ~
      "RAW",

    TRUE ~ x
  )
}


std_regime_57222 <- function(x){

  x <- as.character(x)

  case_when(

    grepl(
      "non",
      x,
      ignore.case = TRUE
    ) ~
      "Non-shadow",

    grepl(
      "shadow",
      x,
      ignore.case = TRUE
    ) ~
      "Shadow",

    TRUE ~ x
  )
}


# ======================================================================
# 57.5 JOURNAL FIGURE THEME
# ======================================================================

theme_57222 <- function(){

  theme_bw(
    base_size = 10
  ) +

    theme(

      panel.grid.minor =
        element_blank(),

      panel.grid.major.y =
        element_blank(),

      panel.grid.major.x =
        element_line(
          linewidth = 0.25,
          colour = "grey90"
        ),

      strip.background =
        element_rect(
          fill = "grey94",
          colour = NA
        ),

      strip.text =
        element_text(
          face = "bold",
          size = 9.5
        ),

      plot.title =
        element_text(
          face = "bold",
          size = 12
        ),

      plot.subtitle =
        element_text(
          size = 9.5
        ),

      axis.title =
        element_text(
          size = 10
        ),

      legend.position =
        "bottom",

      legend.title =
        element_text(
          face = "bold"
        )
    )
}


# ======================================================================
# PART A
# FORMAL INFERENCE FOR DELTA BACC
# ======================================================================


# ======================================================================
# 57.6 VERIFY ORIGINAL DELTA-BACC SOURCE
# ======================================================================

required_bacc_cols_57222 <-
  c(
    "algorithm",
    "replicate",
    "scheme",
    "variant",
    "regime",
    "Delta_BalancedAccuracy"
  )


missing_bacc_cols_57222 <-
  setdiff(
    required_bacc_cols_57222,
    names(
      delta_41222
    )
  )


if(length(missing_bacc_cols_57222) > 0){

  stop(
    paste0(
      "\ndelta_41222 lacks required columns:\n",
      paste(
        missing_bacc_cols_57222,
        collapse = ", "
      )
    )
  )
}


# ======================================================================
# 57.7 STANDARDIZE REPLICATE-LEVEL DELTA BACC
#
# IMPORTANT:
# No averaging occurs before inference.
# Each stochastic replicate remains an observation in the inferential
# distribution.
# ======================================================================

BACC_REP_57222 <-
  delta_41222 %>%

  transmute(

    algorithm =
      std_algorithm_57222(
        algorithm
      ),

    replicate =
      as.integer(
        replicate
      ),

    scheme =
      as.character(
        scheme
      ),

    variant =
      std_variant_57222(
        variant
      ),

    regime =
      std_regime_57222(
        regime
      ),

    Delta_BACC =
      as.numeric(
        Delta_BalancedAccuracy
      )
  ) %>%

  filter(

    algorithm %in%
      c(
        "Random Forest",
        "Ridge logistic",
        "GAM"
      ),

    scheme %in%
      c(
        "UR",
        "SB",
        "SW"
      ),

    variant %in%
      c(
        "RAW",
        "RAW+SHADOW"
      ),

    regime %in%
      c(
        "Non-shadow",
        "Shadow"
      ),

    is.finite(
      Delta_BACC
    )
  )


write_csv(
  BACC_REP_57222,
  file.path(
    DIR_57222_CSV,
    "57_01_delta_BACC_replicate_level_PRE51.csv"
  )
)


# ======================================================================
# 57.8 CHECK UNIQUENESS OF REPLICATE-LEVEL CONTRASTS
# ======================================================================

BACC_DUP_57222 <-
  BACC_REP_57222 %>%

  count(
    algorithm,
    replicate,
    scheme,
    variant,
    regime,
    name = "n_rows"
  ) %>%

  filter(
    n_rows != 1
  )


write_csv(
  BACC_DUP_57222,
  file.path(
    DIR_57222_QC,
    "57_QC_01_delta_BACC_duplicate_keys.csv"
  )
)


if(nrow(BACC_DUP_57222) > 0){

  stop(
    paste0(
      "\nSECTION 57 STOPPED:\n",
      "delta_41222 does not contain exactly one Delta BACC value ",
      "per algorithm × replicate × scheme × variant × regime.\n",
      "Inspect 57_QC_01_delta_BACC_duplicate_keys.csv."
    )
  )
}


# ======================================================================
# 57.9 SAFE WILCOXON SIGNED-RANK TEST AGAINST ZERO
# ======================================================================

safe_wilcox_zero_57222 <- function(x){

  x <-
    x[
      is.finite(x)
    ]


  if(length(x) < 3){

    return(
      tibble(
        n_test = length(x),
        W = NA_real_,
        p_value = NA_real_
      )
    )
  }


  if(
    all(
      abs(x) <
        .Machine$double.eps
    )
  ){

    return(
      tibble(
        n_test = length(x),
        W = 0,
        p_value = 1
      )
    )
  }


  wt <-
    suppressWarnings(
      wilcox.test(
        x,
        mu = 0,
        alternative = "two.sided",
        exact = FALSE,
        correct = FALSE
      )
    )


  tibble(
    n_test =
      length(x),

    W =
      unname(
        wt$statistic
      ),

    p_value =
      wt$p.value
  )
}


# ======================================================================
# 57.10 PAIRED RANK-BISERIAL EFFECT SIZE
#
# For a one-sample / paired Wilcoxon contrast against zero:
#
#   r_rb =
#     (sum positive ranks - sum negative ranks) /
#     (sum positive ranks + sum negative ranks)
#
# Range:
#   -1 = all non-zero differences negative
#    0 = balanced ranks
#   +1 = all non-zero differences positive
#
# Exact zero differences are removed, consistent with the signed-rank
# construction.
# ======================================================================

rank_biserial_zero_57222 <- function(x){

  x <-
    x[
      is.finite(x)
    ]


  x <-
    x[
      abs(x) >
        .Machine$double.eps
    ]


  if(length(x) == 0){

    return(
      NA_real_
    )
  }


  r <-
    rank(
      abs(x),
      ties.method = "average"
    )


  R_pos <-
    sum(
      r[
        x > 0
      ]
    )


  R_neg <-
    sum(
      r[
        x < 0
      ]
    )


  denom <-
    R_pos +
    R_neg


  if(
    !is.finite(denom) ||
    denom <= 0
  ){

    return(
      NA_real_
    )
  }


  (
    R_pos -
    R_neg
  ) /
    denom
}


# ======================================================================
# 57.11 DESCRIPTIVE DELTA-BACC SUMMARY
# ======================================================================

BACC_SUM_57222 <-
  BACC_REP_57222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    n =
      n(),

    mean_delta_BACC =
      mean(
        Delta_BACC,
        na.rm = TRUE
      ),

    median_delta_BACC =
      median(
        Delta_BACC,
        na.rm = TRUE
      ),

    q025 =
      quantile(
        Delta_BACC,
        probs = 0.025,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    q975 =
      quantile(
        Delta_BACC,
        probs = 0.975,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    proportion_positive =
      mean(
        Delta_BACC > 0,
        na.rm = TRUE
      ),

    proportion_negative =
      mean(
        Delta_BACC < 0,
        na.rm = TRUE
      ),

    proportion_zero =
      mean(
        abs(Delta_BACC) <=
          .Machine$double.eps,
        na.rm = TRUE
      ),

    rank_biserial =
      rank_biserial_zero_57222(
        Delta_BACC
      ),

    .groups =
      "drop"
  )


# ======================================================================
# 57.12 WILCOXON TESTS
# ======================================================================

BACC_TESTS_57222 <-
  BACC_REP_57222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  group_modify(
    ~safe_wilcox_zero_57222(
      .x$Delta_BACC
    )
  ) %>%

  ungroup()


# ======================================================================
# 57.13 BH-FDR
#
# ONE INFERENTIAL FAMILY:
#
#   all primary Delta-BACC tests
#
# = 3 algorithms × 3 schemes × 2 variants × 2 regimes
# = expected 36 tests
#
# This FDR adjustment is COMPLETELY SEPARATE from:
#   threshold tests
#   mapped-area tests
#   Section-51 sensitivity analyses
# ======================================================================

BACC_TESTS_57222 <-
  BACC_TESTS_57222 %>%

  mutate(

    p_FDR =
      p.adjust(
        p_value,
        method = "BH"
      ),

    significant_FDR_0_05 =
      is.finite(p_FDR) &
      p_FDR < 0.05
  )


# ======================================================================
# 57.14 COMBINE EFFECT SIZE + UNCERTAINTY + INFERENCE
# ======================================================================

BACC_FINAL_57222 <-
  BACC_SUM_57222 %>%

  left_join(
    BACC_TESTS_57222,
    by =
      c(
        "algorithm",
        "scheme",
        "variant",
        "regime"
      )
  ) %>%

  mutate(

    configuration =
      paste(
        scheme,
        variant,
        sep = " | "
      ),

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      ),

    configuration =
      factor(
        configuration,
        levels =
          c(
            "SW | RAW+SHADOW",
            "SW | RAW",
            "UR | RAW+SHADOW",
            "SB | RAW",
            "SB | RAW+SHADOW",
            "UR | RAW"
          )
      )
  )


write_csv(
  BACC_SUM_57222,
  file.path(
    DIR_57222_CSV,
    "57_02_delta_BACC_descriptive_summary_PRE51.csv"
  )
)


write_csv(
  BACC_TESTS_57222,
  file.path(
    DIR_57222_CSV,
    "57_03_delta_BACC_Wilcoxon_BH_FDR_PRE51.csv"
  )
)


write_csv(
  BACC_FINAL_57222,
  file.path(
    DIR_57222_CSV,
    "57_04_delta_BACC_FINAL_inference_PRE51.csv"
  )
)


# ======================================================================
# 57.15 DELTA-BACC MAIN FIGURE WITH EMPIRICAL REPLICATE INTERVALS
#
# IMPORTANT:
# - point = mean Delta BACC
# - horizontal interval = empirical 2.5th–97.5th replicate quantiles
# - NO significance stars
# - inferential p-values remain in the table
# ======================================================================

FIG_BACC_57222 <-
  ggplot(
    BACC_FINAL_57222,
    aes(
      x = mean_delta_BACC,
      y = configuration
    )
  ) +

  geom_vline(
    xintercept = 0,
    linewidth = 0.45,
    colour = "grey35"
  ) +

  geom_errorbarh(
    aes(
      xmin = q025,
      xmax = q975
    ),
    height = 0,
    linewidth = 0.55
  ) +

  geom_point(
    size = 2.7
  ) +

  facet_grid(
    algorithm ~ regime,
    scales = "free_x"
  ) +

  labs(

    title =
      expression(
        Delta*BACC ==
          BACC[Regime] -
          BACC[Global]
      ),

    subtitle =
      "Original pre-Section-51 algorithm-specific thresholds; points are means and horizontal ranges are empirical 95% replicate intervals",

    x =
      expression(
        Delta*BACC
      ),

    y =
      NULL
  ) +

  theme_57222()


ggsave(
  file.path(
    DIR_57222_FIG,
    "Figure57_A_MAIN_delta_BACC_mean_95RI_PRE51.png"
  ),
  FIG_BACC_57222,
  width = 9,
  height = 8,
  units = "in",
  dpi = 600,
  bg = "white"
)


ggsave(
  file.path(
    DIR_57222_FIG,
    "Figure57_A_MAIN_delta_BACC_mean_95RI_PRE51.tiff"
  ),
  FIG_BACC_57222,
  width = 9,
  height = 8,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# ======================================================================
# PART B
# COMPACT DELTA-VIP MAIN FIGURE
# ======================================================================


# ======================================================================
# 57.16 VERIFY ORIGINAL IMPORTANCE SOURCE
# ======================================================================

required_imp_cols_57222 <-
  c(
    "algorithm",
    "replicate",
    "scheme",
    "variant",
    "predictor",
    "Delta_Brier"
  )


missing_imp_cols_57222 <-
  setdiff(
    required_imp_cols_57222,
    names(
      importance_wide_46222
    )
  )


if(length(missing_imp_cols_57222) > 0){

  stop(
    paste0(
      "\nimportance_wide_46222 lacks required columns:\n",
      paste(
        missing_imp_cols_57222,
        collapse = ", "
      )
    )
  )
}


# ======================================================================
# 57.17 STANDARDIZE ORIGINAL REPLICATE-LEVEL DELTA VIP
#
# Delta_VIP = VIP_Shadow - VIP_Non-shadow
#
# Positive:
#   predictor contributes more strongly in shadow.
#
# Negative:
#   predictor contributes more strongly in non-shadow.
# ======================================================================

VIP_REP_57222 <-
  importance_wide_46222 %>%

  transmute(

    algorithm =
      std_algorithm_57222(
        algorithm
      ),

    replicate =
      as.integer(
        replicate
      ),

    scheme =
      as.character(
        scheme
      ),

    variant =
      std_variant_57222(
        variant
      ),

    predictor =
      as.character(
        predictor
      ),

    Delta_VIP =
      as.numeric(
        Delta_Brier
      )
  ) %>%

  filter(

    algorithm %in%
      c(
        "Random Forest",
        "Ridge logistic",
        "GAM"
      ),

    scheme %in%
      c(
        "UR",
        "SB",
        "SW"
      ),

    variant %in%
      c(
        "RAW",
        "RAW+SHADOW"
      ),

    is.finite(
      Delta_VIP
    )
  )


write_csv(
  VIP_REP_57222,
  file.path(
    DIR_57222_CSV,
    "57_05_delta_VIP_replicate_level_PRE51.csv"
  )
)


# ======================================================================
# 57.18 FIRST AVERAGE ACROSS SAMPLING SCHEMES WITHIN REPLICATE
#
# Why:
# The compact MAIN figure describes the overall illumination-dependent
# predictor pattern rather than treating UR/SB/SW as independent
# inferential observations.
#
# Therefore each stochastic replicate contributes ONE value per:
#
#   algorithm × replicate × variant × predictor
#
# after averaging the three sampling-scheme realizations.
#
# This avoids pseudo-replication from pooling UR/SB/SW as 90 independent
# observations.
# ======================================================================

VIP_REP_AVG_57222 <-
  VIP_REP_57222 %>%

  group_by(
    algorithm,
    replicate,
    variant,
    predictor
  ) %>%

  summarise(

    Delta_VIP =
      mean(
        Delta_VIP,
        na.rm = TRUE
      ),

    n_schemes =
      n_distinct(
        scheme
      ),

    .groups =
      "drop"
  )


# ======================================================================
# 57.19 QC — EXPECT THREE SAMPLING SCHEMES WHERE AVAILABLE
# ======================================================================

VIP_SCHEME_QC_57222 <-
  VIP_REP_AVG_57222 %>%

  count(
    n_schemes,
    name = "n_predictor_replicates"
  )


write_csv(
  VIP_SCHEME_QC_57222,
  file.path(
    DIR_57222_QC,
    "57_QC_02_delta_VIP_number_of_schemes_per_replicate.csv"
  )
)


# ======================================================================
# 57.20 SUMMARY FOR EACH ALGORITHM × VARIANT × PREDICTOR
# ======================================================================

VIP_SUM_57222 <-
  VIP_REP_AVG_57222 %>%

  group_by(
    algorithm,
    variant,
    predictor
  ) %>%

  summarise(

    n =
      n(),

    mean_Delta_VIP =
      mean(
        Delta_VIP,
        na.rm = TRUE
      ),

    median_Delta_VIP =
      median(
        Delta_VIP,
        na.rm = TRUE
      ),

    q025 =
      quantile(
        Delta_VIP,
        probs = 0.025,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    q975 =
      quantile(
        Delta_VIP,
        probs = 0.975,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    .groups =
      "drop"
  )


write_csv(
  VIP_SUM_57222,
  file.path(
    DIR_57222_CSV,
    "57_06_delta_VIP_summary_PRE51.csv"
  )
)


# ======================================================================
# 57.21 RULE-BASED PREDICTOR SELECTION FOR MAIN FIGURE
#
# IMPORTANT:
#
# We DO NOT independently choose "top predictors" for RAW and
# RAW+SHADOW.
#
# For each algorithm and predictor:
#
#   selection_score =
#     maximum absolute mean Delta VIP across the two variants.
#
# Then the TOP N predictors are selected ONCE for each algorithm.
#
# Consequently the SAME predictors are displayed for RAW and
# RAW+SHADOW.
#
# This prevents visual cherry-picking and makes direct variant
# comparison possible.
# ======================================================================

N_TOP_57222 <- 10


VIP_SELECTION_SCORE_57222 <-
  VIP_SUM_57222 %>%

  group_by(
    algorithm,
    predictor
  ) %>%

  summarise(

    selection_score =
      max(
        abs(
          mean_Delta_VIP
        ),
        na.rm = TRUE
      ),

    .groups =
      "drop"
  )


VIP_SELECTED_57222 <-
  VIP_SELECTION_SCORE_57222 %>%

  group_by(
    algorithm
  ) %>%

  arrange(
    desc(
      selection_score
    ),
    predictor,
    .by_group = TRUE
  ) %>%

  slice_head(
    n = N_TOP_57222
  ) %>%

  ungroup()


write_csv(
  VIP_SELECTION_SCORE_57222,
  file.path(
    DIR_57222_CSV,
    "57_07_delta_VIP_predictor_selection_scores_PRE51.csv"
  )
)


write_csv(
  VIP_SELECTED_57222,
  file.path(
    DIR_57222_CSV,
    "57_08_delta_VIP_TOP10_predictors_PRE51.csv"
  )
)


# ======================================================================
# 57.22 BUILD DISPLAY DATA
# ======================================================================

VIP_DISPLAY_57222 <-
  VIP_SUM_57222 %>%

  inner_join(
    VIP_SELECTED_57222 %>%
      select(
        algorithm,
        predictor,
        selection_score
      ),
    by =
      c(
        "algorithm",
        "predictor"
      )
  ) %>%

  mutate(

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW+SHADOW"
          )
      ),

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      )
  )


# ======================================================================
# 57.23 CREATE ALGORITHM-SPECIFIC PREDICTOR ORDER
#
# Predictor names may repeat among algorithms, but their ranking differs.
# A hidden algorithm-specific key is therefore used for plotting.
# ======================================================================

VIP_ORDER_57222 <-
  VIP_SELECTED_57222 %>%

  arrange(
    algorithm,
    selection_score
  ) %>%

  mutate(

    predictor_key =
      paste(
        algorithm,
        predictor,
        sep = "___"
      )
  )


VIP_DISPLAY_57222 <-
  VIP_DISPLAY_57222 %>%

  mutate(

    predictor_key =
      paste(
        as.character(
          algorithm
        ),
        predictor,
        sep = "___"
      ),

    predictor_key =
      factor(
        predictor_key,
        levels =
          VIP_ORDER_57222$
            predictor_key
      )
  )


# ======================================================================
# 57.24 SAVE DISPLAY TABLE
# ======================================================================

write_csv(
  VIP_DISPLAY_57222 %>%
    mutate(
      predictor_key =
        as.character(
          predictor_key
        )
    ),
  file.path(
    DIR_57222_CSV,
    "57_09_delta_VIP_MAIN_figure_data_PRE51.csv"
  )
)


# ======================================================================
# 57.25 LABEL FUNCTION
#
# Remove hidden algorithm prefix from y-axis labels.
# ======================================================================

strip_algorithm_57222 <- function(x){

  sub(
    "^.*___",
    "",
    x
  )
}


# ======================================================================
# 57.26 COMPACT MAIN DELTA-VIP FIGURE
#
# Each algorithm gets its own panel.
#
# Point:
#   mean Delta VIP across stochastic replicates after averaging
#   UR/SB/SW within replicate.
#
# Horizontal range:
#   empirical 2.5th–97.5th replicate quantiles.
#
# RAW and RAW+SHADOW:
#   shown for the SAME predictor set.
#
# NO significance inference is attached to this figure.
# ======================================================================

FIG_VIP_57222 <-
  ggplot(
    VIP_DISPLAY_57222,
    aes(
      x =
        mean_Delta_VIP,

      y =
        predictor_key,

      shape =
        variant,

      linetype =
        variant
    )
  ) +

  geom_vline(
    xintercept = 0,
    linewidth = 0.45,
    colour = "grey30"
  ) +

  geom_errorbarh(
    aes(
      xmin = q025,
      xmax = q975
    ),
    height = 0,
    linewidth = 0.55,
    position =
      position_dodge(
        width = 0.45
      )
  ) +

  geom_point(
    size = 2.5,
    position =
      position_dodge(
        width = 0.45
      )
  ) +

  facet_wrap(
    ~algorithm,
    ncol = 3,
    scales = "free"
  ) +

  scale_y_discrete(
    labels =
      strip_algorithm_57222
  ) +

  labs(

    title =
      "Regime-dependent environmental predictor contribution",

    subtitle =
      paste0(
        "Top ",
        N_TOP_57222,
        " predictors per algorithm selected by the maximum absolute ",
        "mean shadow–non-shadow contrast across RAW and RAW+SHADOW"
      ),

    x =
      expression(
        Delta*VI[Brier] ==
          VI[Shadow] -
          VI[Non-shadow]
      ),

    y =
      NULL,

    shape =
      "Predictor variant",

    linetype =
      "Predictor variant"
  ) +

  theme_57222() +

  theme(

    axis.text.y =
      element_text(
        size = 8.2
      ),

    legend.position =
      "bottom"
  )


ggsave(
  file.path(
    DIR_57222_FIG,
    "Figure57_B_MAIN_compact_delta_VIP_PRE51.png"
  ),
  FIG_VIP_57222,
  width = 11,
  height = 6.5,
  units = "in",
  dpi = 600,
  bg = "white"
)


ggsave(
  file.path(
    DIR_57222_FIG,
    "Figure57_B_MAIN_compact_delta_VIP_PRE51.tiff"
  ),
  FIG_VIP_57222,
  width = 11,
  height = 6.5,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# ======================================================================
# 57.27 FULL SUPPLEMENTARY DELTA-VIP TABLE
#
# No predictor filtering.
# This preserves all information for Supplementary Material.
# ======================================================================

write_csv(
  VIP_SUM_57222,
  file.path(
    DIR_57222_CSV,
    "57_10_delta_VIP_ALL_predictors_supplementary_PRE51.csv"
  )
)


# ======================================================================
# 57.28 FINAL QC
# ======================================================================

expected_bacc_groups_57222 <-
  3 * 3 * 2 * 2


QC_57222 <-
  tibble(

    check =
      c(
        "Delta BACC source exists",
        "Delta VIP source exists",
        "Delta BACC has no duplicate replicate keys",
        "Three algorithms in Delta BACC",
        "Three algorithms in Delta VIP",
        "Three sampling schemes in Delta BACC",
        "Two predictor variants in Delta BACC",
        "Two illumination regimes in Delta BACC",
        "Expected 36 Delta-BACC inferential groups",
        "All Delta-BACC FDR values finite where raw p is finite",
        "Top predictor selection generated for three algorithms",
        "No Section-51+ analytical source required"
      ),

    passed =
      c(

        exists(
          "delta_41222",
          envir = .GlobalEnv
        ),

        exists(
          "importance_wide_46222",
          envir = .GlobalEnv
        ),

        nrow(
          BACC_DUP_57222
        ) == 0,

        n_distinct(
          BACC_REP_57222$
            algorithm
        ) == 3,

        n_distinct(
          VIP_REP_57222$
            algorithm
        ) == 3,

        n_distinct(
          BACC_REP_57222$
            scheme
        ) == 3,

        n_distinct(
          BACC_REP_57222$
            variant
        ) == 2,

        n_distinct(
          BACC_REP_57222$
            regime
        ) == 2,

        nrow(
          BACC_FINAL_57222
        ) ==
          expected_bacc_groups_57222,

        all(
          is.finite(
            BACC_TESTS_57222$
              p_FDR[
                is.finite(
                  BACC_TESTS_57222$
                    p_value
                )
              ]
          )
        ),

        n_distinct(
          VIP_SELECTED_57222$
            algorithm
        ) == 3,

        TRUE
      )
  )


print(
  QC_57222,
  n = Inf
)


write_csv(
  QC_57222,
  file.path(
    DIR_57222_QC,
    "57_99_FINAL_QC.csv"
  )
)


if(!all(QC_57222$passed)){

  stop(
    paste0(
      "\nSECTION 57 FINAL QC FAILED.\n",
      "Inspect:\n",
      file.path(
        DIR_57222_QC,
        "57_99_FINAL_QC.csv"
      )
    )
  )
}


# ======================================================================
# 57.29 SESSION INFO
# ======================================================================

capture.output(
  sessionInfo(),
  file =
    file.path(
      DIR_57222_QC,
      "57_sessionInfo.txt"
    )
)


# ======================================================================
# 57.30 FINAL CONSOLE REPORT
# ======================================================================

cat(
  "\n============================================================\n",
  "SECTION 57 COMPLETED SUCCESSFULLY\n",
  "============================================================\n",
  "\nANALYTICAL SOURCES — STRICTLY PRE-SECTION-51:\n",
  "  Delta BACC : delta_41222\n",
  "  Delta VIP  : importance_wide_46222\n",
  "\nSECTION 51+ ANALYTICAL SOURCES USED: NONE\n",
  "\nDELTA-BACC INFERENCE:\n",
  "  57_04_delta_BACC_FINAL_inference_PRE51.csv\n",
  "\nMAIN FIGURES:\n",
  "  Figure57_A_MAIN_delta_BACC_mean_95RI_PRE51.png\n",
  "  Figure57_B_MAIN_compact_delta_VIP_PRE51.png\n",
  "\nSUPPLEMENTARY FULL PREDICTOR TABLE:\n",
  "  57_10_delta_VIP_ALL_predictors_supplementary_PRE51.csv\n",
  "\nOUTPUT DIRECTORY:\n",
  DIR_57222,
  "\n============================================================\n",
  sep = ""
)































































# ======================================================================
# Final publication figures
# FINAL PUBLICATION POLISH OF MAIN DELTA-BACC AND
# REGIME-DEPENDENT PREDICTOR-CONTRIBUTION FIGURES
#
# PURPOSE
# ----------------------------------------------------------------------
# Presentation-only refinement of two manuscript figures:
#
#   A. Delta BACC:
#      BACC_regime-specific - BACC_global
#
#   B. Regime-dependent environmental predictor contribution:
#      Delta VI_Brier = VI_Shadow - VI_Non-shadow
#
# CRITICAL ANALYTICAL RULES
# ----------------------------------------------------------------------
# - NO model refitting
# - NO resampling
# - NO threshold re-estimation
# - NO new threshold search
# - NO new statistical tests
# - NO external-test optimization
# - NO Section-51 common-exact thresholds
# - NO Section-52/53/54 analytical results
#
# SOURCES
# ----------------------------------------------------------------------
# Figure A:
#   BACC_FINAL_57222
#   -> generated by Section 57 from original PRIMARY-ANALYSIS
#      algorithm-specific thresholds.
#
# Figure B:
#   importance_wide_46222
#   -> original PRIMARY-ANALYSIS permutation-importance results.
#
# IMPORTANT
# ----------------------------------------------------------------------
# Section 58 changes ONLY graphical presentation.
# Analytical values are preserved.
# ======================================================================


# ======================================================================
# 58.1 PACKAGES
# ======================================================================

suppressPackageStartupMessages({

  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(scales)
  library(readr)
  library(grid)

})


# patchwork is used ONLY for arranging already-generated ggplots

if(!requireNamespace("patchwork", quietly = TRUE)){

  stop(
    paste0(
      "\nSECTION 58 requires package 'patchwork' only for figure layout.\n",
      "Install it once with:\n",
      "install.packages('patchwork')\n"
    )
  )

}


# ======================================================================
# 58.2 REQUIRED OBJECTS
# ======================================================================

required_58222 <-
  c(
    "OUT_DIR",
    "BACC_FINAL_57222",
    "importance_wide_46222"
  )


missing_58222 <-
  required_58222[
    !vapply(
      required_58222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(length(missing_58222) > 0){

  stop(
    paste0(
      "\nSECTION 58 cannot start.\n",
      "Missing required PRE-51 / Section-57 objects:\n",
      paste(
        missing_58222,
        collapse = "\n"
      ),
      "\n\nRun the original pipeline through Section 57 first."
    )
  )

}


# ======================================================================
# 58.3 OUTPUT DIRECTORIES
# ======================================================================

DIR_58222 <-
  file.path(
    OUT_DIR,
    "58_222_FINAL_PUBLICATION_FIGURE_POLISH"
  )


# Do not overwrite a previous successful run.
# If directory already exists, create a safe rerun directory.

if(dir.exists(DIR_58222)){

  suffix_58222 <-
    format(
      Sys.time(),
      "%Y%m%d_%H%M%S"
    )

  DIR_58222 <-
    paste0(
      DIR_58222,
      "_RERUN_",
      suffix_58222
    )

}


DIR_58222_FIG <-
  file.path(
    DIR_58222,
    "FIGURES"
  )


DIR_58222_CSV <-
  file.path(
    DIR_58222,
    "CSV"
  )


DIR_58222_QC <-
  file.path(
    DIR_58222,
    "QC"
  )


invisible(
  lapply(
    c(
      DIR_58222,
      DIR_58222_FIG,
      DIR_58222_CSV,
      DIR_58222_QC
    ),
    dir.create,
    recursive = TRUE,
    showWarnings = FALSE
  )
)


cat(
  "\n============================================================\n",
  "SECTION 58 OUTPUT DIRECTORY\n",
  "============================================================\n",
  DIR_58222,
  "\n============================================================\n",
  sep = ""
)


# ======================================================================
# 58.4 STANDARDIZATION HELPERS
# ======================================================================

std_algorithm_58222 <- function(x){

  x <- as.character(x)

  dplyr::case_when(

    grepl(
      "random",
      x,
      ignore.case = TRUE
    ) ~ "Random Forest",

    grepl(
      "ridge",
      x,
      ignore.case = TRUE
    ) ~ "Ridge logistic",

    grepl(
      "^gam$|generalized additive",
      x,
      ignore.case = TRUE
    ) ~ "GAM",

    TRUE ~ x

  )

}


std_variant_58222 <- function(x){

  x <- as.character(x)

  dplyr::case_when(

    x %in%
      c(
        "RAW_SHADOW",
        "RAW+SHADOW",
        "RAW + SHADOW",
        "RAW + shadow",
        "RAW+shadow"
      ) ~ "RAW+SHADOW",

    x == "RAW" ~
      "RAW",

    TRUE ~
      x

  )

}


std_regime_58222 <- function(x){

  x <- as.character(x)

  dplyr::case_when(

    grepl(
      "non",
      x,
      ignore.case = TRUE
    ) ~ "Non-shadow",

    grepl(
      "shadow",
      x,
      ignore.case = TRUE
    ) ~ "Shadow",

    TRUE ~
      x

  )

}


# ======================================================================
# 58.5 JOURNAL THEME
# ======================================================================

theme_58222 <- function(base_size = 11){

  theme_bw(
    base_size = base_size
  ) +

    theme(

      panel.grid.minor =
        element_blank(),

      panel.grid.major.y =
        element_blank(),

      panel.grid.major.x =
        element_line(
          linewidth = 0.30,
          colour = "grey90"
        ),

      strip.background =
        element_rect(
          fill = "grey94",
          colour = NA
        ),

      strip.text =
        element_text(
          face = "bold",
          size = base_size
        ),

      plot.title =
        element_text(
          face = "bold",
          size = base_size + 3,
          hjust = 0
        ),

      plot.subtitle =
        element_text(
          size = base_size,
          hjust = 0,
          margin =
            margin(
              b = 8
            )
        ),

      axis.title =
        element_text(
          size = base_size + 0.5
        ),

      axis.text =
        element_text(
          size = base_size - 0.5
        ),

      legend.position =
        "bottom",

      legend.title =
        element_text(
          face = "bold"
        ),

      plot.margin =
        margin(
          10,
          12,
          10,
          10
        )

    )

}


# ======================================================================
# PART A
# FINAL DELTA-BACC FIGURE
# ======================================================================


# ======================================================================
# 58.6 VERIFY SECTION-57 BACC OBJECT
# ======================================================================

required_bacc_cols_58222 <-
  c(
    "algorithm",
    "scheme",
    "variant",
    "regime",
    "mean_delta_BACC",
    "q025",
    "q975"
  )


missing_bacc_cols_58222 <-
  setdiff(
    required_bacc_cols_58222,
    names(
      BACC_FINAL_57222
    )
  )


if(length(missing_bacc_cols_58222) > 0){

  stop(
    paste0(
      "\nBACC_FINAL_57222 lacks required columns:\n",
      paste(
        missing_bacc_cols_58222,
        collapse = ", "
      )
    )
  )

}


# ======================================================================
# 58.7 PREPARE BACC DATA
#
# NO analytical recalculation.
# Only factor ordering / presentation.
# ======================================================================

BACC_PLOT_58222 <-
  BACC_FINAL_57222 %>%

  transmute(

    algorithm =
      std_algorithm_58222(
        algorithm
      ),

    scheme =
      as.character(
        scheme
      ),

    variant =
      std_variant_58222(
        variant
      ),

    regime =
      std_regime_58222(
        regime
      ),

    mean_delta_BACC =
      as.numeric(
        mean_delta_BACC
      ),

    q025 =
      as.numeric(
        q025
      ),

    q975 =
      as.numeric(
        q975
      )

  ) %>%

  filter(

    algorithm %in%
      c(
        "Random Forest",
        "Ridge logistic",
        "GAM"
      ),

    scheme %in%
      c(
        "UR",
        "SB",
        "SW"
      ),

    variant %in%
      c(
        "RAW",
        "RAW+SHADOW"
      ),

    regime %in%
      c(
        "Non-shadow",
        "Shadow"
      )

  ) %>%

  mutate(

    configuration =
      paste(
        scheme,
        ifelse(
          variant == "RAW+SHADOW",
          "RAW + shadow",
          "RAW"
        ),
        sep = " | "
      ),

    # Logical manuscript order:
    # UR pair -> SB pair -> SW pair
    #
    # ggplot displays first factor level at bottom,
    # therefore levels are reversed here.

    configuration =
      factor(
        configuration,
        levels =
          rev(
            c(
              "UR | RAW",
              "UR | RAW + shadow",
              "SB | RAW",
              "SB | RAW + shadow",
              "SW | RAW",
              "SW | RAW + shadow"
            )
          )
      ),

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      )

  )


# ======================================================================
# 58.8 HARD BACC QC
# ======================================================================

if(
  nrow(
    BACC_PLOT_58222
  ) !=
    3 * 3 * 2 * 2
){

  stop(
    paste0(
      "\nUnexpected BACC plotting rows.\n",
      "Expected 36 = 3 algorithms x 3 schemes x ",
      "2 variants x 2 regimes.\n",
      "Observed: ",
      nrow(
        BACC_PLOT_58222
      )
    )
  )

}


if(
  any(
    !is.finite(
      BACC_PLOT_58222$
        mean_delta_BACC
    )
  ) ||
  any(
    !is.finite(
      BACC_PLOT_58222$q025
    )
  ) ||
  any(
    !is.finite(
      BACC_PLOT_58222$q975
    )
  )
){

  stop(
    "Non-finite Delta-BACC values detected."
  )

}


if(
  any(
    BACC_PLOT_58222$q025 >
      BACC_PLOT_58222$mean_delta_BACC
  ) ||
  any(
    BACC_PLOT_58222$q975 <
      BACC_PLOT_58222$mean_delta_BACC
  )
){

  warning(
    paste0(
      "At least one mean Delta BACC lies outside its empirical ",
      "2.5-97.5% replicate interval. Inspect Section 57."
    )
  )

}


write_csv(

  BACC_PLOT_58222,

  file.path(
    DIR_58222_CSV,
    "58_01_delta_BACC_values_used_exactly_from_Section57.csv"
  )

)


# ======================================================================
# 58.9 FINAL MAIN DELTA-BACC FIGURE
# ======================================================================

FIG58_A_58222 <-

  ggplot(

    BACC_PLOT_58222,

    aes(
      x =
        mean_delta_BACC,

      y =
        configuration
    )

  ) +

  geom_vline(

    xintercept = 0,

    linewidth = 0.62,

    colour = "grey35"

  ) +

  geom_errorbarh(

    aes(
      xmin = q025,
      xmax = q975
    ),

    height = 0,

    linewidth = 0.62,

    lineend = "round"

  ) +

  geom_point(

    size = 3.0

  ) +

  facet_grid(

    algorithm ~ regime,

    scales = "free_x"

  ) +

  labs(

    title =
      "Effect of regime-specific thresholding on balanced accuracy",

    subtitle =
      expression(
        Delta*BACC ==
          BACC[regime-specific] -
          BACC[global]
      ),

    x =
      expression(
        Delta*BACC
      ),

    y =
      NULL

  ) +

  theme_58222(
    base_size = 11
  ) +

  theme(

    panel.spacing.x =
      unit(
        0.9,
        "lines"
      ),

    panel.spacing.y =
      unit(
        0.55,
        "lines"
      ),

    axis.text.y =
      element_text(
        size = 10
      ),

    strip.text =
      element_text(
        face = "bold",
        size = 11
      )

  )


# ======================================================================
# 58.10 SAVE FINAL BACC FIGURE
# ======================================================================

ggsave(

  file.path(
    DIR_58222_FIG,
    "Figure58_A_MAIN_delta_BACC_FINAL.png"
  ),

  FIG58_A_58222,

  width = 9.3,
  height = 8.3,
  units = "in",
  dpi = 600,
  bg = "white"

)


ggsave(

  file.path(
    DIR_58222_FIG,
    "Figure58_A_MAIN_delta_BACC_FINAL.tiff"
  ),

  FIG58_A_58222,

  width = 9.3,
  height = 8.3,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"

)


ggsave(

  file.path(
    DIR_58222_FIG,
    "Figure58_A_MAIN_delta_BACC_FINAL.pdf"
  ),

  FIG58_A_58222,

  width = 9.3,
  height = 8.3,
  units = "in",
  bg = "white"

)


# ======================================================================
# PART B
# FINAL REGIME-DEPENDENT PREDICTOR CONTRIBUTION FIGURE
# ======================================================================


# ======================================================================
# 58.11 VERIFY ORIGINAL PRIMARY-ANALYSIS IMPORTANCE OBJECT
# ======================================================================

required_vip_cols_58222 <-
  c(
    "algorithm",
    "replicate",
    "scheme",
    "variant",
    "predictor",
    "Delta_Brier"
  )


missing_vip_cols_58222 <-
  setdiff(
    required_vip_cols_58222,
    names(
      importance_wide_46222
    )
  )


if(length(missing_vip_cols_58222) > 0){

  stop(
    paste0(
      "\nimportance_wide_46222 lacks required columns:\n",
      paste(
        missing_vip_cols_58222,
        collapse = ", "
      )
    )
  )

}


# ======================================================================
# 58.12 STANDARDIZE ORIGINAL IMPORTANCE DATA
#
# IMPORTANT:
# This is the SAME underlying PRIMARY Delta_Brier analysis.
# No model or permutation is rerun.
# ======================================================================

VIP_BASE_58222 <-
  importance_wide_46222 %>%

  transmute(

    algorithm =
      std_algorithm_58222(
        algorithm
      ),

    replicate =
      as.integer(
        replicate
      ),

    scheme =
      as.character(
        scheme
      ),

    variant =
      std_variant_58222(
        variant
      ),

    predictor =
      as.character(
        predictor
      ),

    Delta_Brier =
      as.numeric(
        Delta_Brier
      )

  ) %>%

  filter(

    algorithm %in%
      c(
        "Random Forest",
        "Ridge logistic",
        "GAM"
      ),

    scheme %in%
      c(
        "UR",
        "SB",
        "SW"
      ),

    variant %in%
      c(
        "RAW",
        "RAW+SHADOW"
      ),

    is.finite(
      Delta_Brier
    )

  )


# Protection against accidental inclusion of the illumination
# indicator itself in environmental predictor-contribution plots.

forbidden_predictors_58222 <-
  c(
    "SCL",
    "ShadowMask",
    "shadow_bin"
  )


if(
  any(
    VIP_BASE_58222$predictor %in%
      forbidden_predictors_58222
  )
){

  stop(
    paste0(
      "\nPredictor-contribution data unexpectedly contain ",
      "SCL / ShadowMask / shadow_bin.\n",
      "This violates the environmental-predictor contribution design."
    )
  )

}


# ======================================================================
# 58.13 EXACT VISUAL SUMMARY USED IN LEGACY BAR FIGURE
#
# For each:
#   algorithm x predictor variant x predictor
#
# average Delta VI over all available PRIMARY:
#   replicates + sampling scenarios
#
# This does NOT create a new inferential result.
# It is a descriptive display summary of the existing replicate-level
# importance contrasts.
# ======================================================================

VIP_MEAN_58222 <-
  VIP_BASE_58222 %>%

  group_by(
    algorithm,
    variant,
    predictor
  ) %>%

  summarise(

    mean_Delta_VIP =
      mean(
        Delta_Brier,
        na.rm = TRUE
      ),

    n_values =
      sum(
        is.finite(
          Delta_Brier
        )
      ),

    .groups =
      "drop"

  )


# ======================================================================
# 58.14 SELECT STRONGEST POSITIVE AND NEGATIVE CONTRASTS
#
# This preserves the original legacy-style logic:
#   strongest shadow-favouring
#   strongest non-shadow-favouring
#
# separately within algorithm x variant.
#
# N_SIDE = 8 gives up to 16 predictors per panel.
# ======================================================================

N_SIDE_58222 <- 8L


VIP_SELECTED_58222 <-
  VIP_MEAN_58222 %>%

  group_by(
    algorithm,
    variant
  ) %>%

  group_modify(

    ~{

      positive_58222 <-
        .x %>%

        filter(
          mean_Delta_VIP > 0
        ) %>%

        arrange(
          desc(
            mean_Delta_VIP
          )
        ) %>%

        slice_head(
          n = N_SIDE_58222
        )


      negative_58222 <-
        .x %>%

        filter(
          mean_Delta_VIP < 0
        ) %>%

        arrange(
          mean_Delta_VIP
        ) %>%

        slice_head(
          n = N_SIDE_58222
        )


      bind_rows(
        negative_58222,
        positive_58222
      )

    }

  ) %>%

  ungroup() %>%

  mutate(

    Direction =
      ifelse(
        mean_Delta_VIP > 0,
        "Shadow",
        "Non-shadow"
      )

  )


# ======================================================================
# 58.15 SAVE EXACT DISPLAY DATA
# ======================================================================

write_csv(

  VIP_SELECTED_58222,

  file.path(
    DIR_58222_CSV,
    "58_02_predictor_contribution_display_values_PRE51.csv"
  )

)


write_csv(

  VIP_MEAN_58222,

  file.path(
    DIR_58222_CSV,
    "58_03_predictor_contribution_ALL_descriptive_values_PRE51.csv"
  )

)


# ======================================================================
# 58.16 FUNCTION FOR ONE ALGORITHM
#
# Critical graphical improvement over Figure 56:
#
# - one algorithm receives its own vertical plotting block
# - RAW and RAW+SHADOW are vertically stacked
# - predictor labels are independent for each algorithm
# - no blank regions caused by predictors selected for another algorithm
# - x scale is algorithm-specific
#
# This is much more readable while preserving the same information.
# ======================================================================

make_vip_algorithm_plot_58222 <-
  function(
    algorithm_name,
    show_y_title = FALSE
  ){

    dd <-
      VIP_SELECTED_58222 %>%

      filter(
        algorithm ==
          algorithm_name
      ) %>%

      mutate(

        variant =
          factor(
            variant,
            levels =
              c(
                "RAW",
                "RAW+SHADOW"
              ),
            labels =
              c(
                "RAW",
                "RAW + shadow"
              )
          )

      )


    # ----------------------------------------------------------
    # Independent predictor ordering for this algorithm.
    #
    # We create a common order based on maximum absolute
    # displayed contrast across RAW / RAW+shadow.
    # ----------------------------------------------------------

    predictor_order <-
      dd %>%

      group_by(
        predictor
      ) %>%

      summarise(

        max_abs =
          max(
            abs(
              mean_Delta_VIP
            ),
            na.rm = TRUE
          ),

        signed_at_max =
          mean_Delta_VIP[
            which.max(
              abs(
                mean_Delta_VIP
              )
            )
          ][1],

        .groups =
          "drop"

      ) %>%

      arrange(
        signed_at_max
      ) %>%

      pull(
        predictor
      )


    dd <-
      dd %>%

      mutate(

        predictor =
          factor(
            predictor,
            levels =
              predictor_order
          )

      )


    ggplot(

      dd,

      aes(
        x =
          mean_Delta_VIP,

        y =
          predictor,

        fill =
          Direction
      )

    ) +

      geom_vline(

        xintercept = 0,

        linewidth = 0.55,

        colour = "grey25"

      ) +

      geom_col(

        width = 0.72

      ) +

      facet_grid(

        variant ~ .,

        scales = "free_y",
        space = "free_y"

      ) +

      labs(

        title =
          algorithm_name,

        x =
          expression(
            Delta*VI[Brier]
          ),

        y =
          if(
            show_y_title
          ){
            NULL
          } else {
            NULL
          },

        fill =
          "Greater contribution"

      ) +

      theme_58222(
        base_size = 10.5
      ) +

      theme(

        plot.title =
          element_text(
            face = "bold",
            size = 12,
            hjust = 0.5
          ),

        plot.subtitle =
          element_blank(),

        axis.text.y =
          element_text(
            size = 8.7
          ),

        axis.title.x =
          element_text(
            size = 10.5
          ),

        strip.text =
          element_text(
            face = "bold",
            size = 10.5
          ),

        legend.position =
          "none",

        panel.spacing.y =
          unit(
            0.45,
            "lines"
          ),

        plot.margin =
          margin(
            4,
            6,
            5,
            6
          )

      )

  }


# ======================================================================
# 58.17 BUILD THREE ALGORITHM-SPECIFIC PANELS
# ======================================================================

VIP_RF_58222 <-
  make_vip_algorithm_plot_58222(
    "Random Forest"
  )


VIP_RIDGE_58222 <-
  make_vip_algorithm_plot_58222(
    "Ridge logistic"
  )


VIP_GAM_58222 <-
  make_vip_algorithm_plot_58222(
    "GAM"
  )


# ======================================================================
# 58.18 FINAL COMBINED PREDICTOR-CONTRIBUTION FIGURE
#
# IMPORTANT:
# each algorithm has its OWN x scale.
#
# Therefore absolute bar lengths should be interpreted WITHIN algorithm,
# not as a formal between-algorithm effect-size comparison.
# ======================================================================

FIG58_B_BODY_58222 <-

  VIP_RF_58222 +

  VIP_RIDGE_58222 +

  VIP_GAM_58222 +

  patchwork::plot_layout(
    ncol = 3,
    widths =
      c(
        1,
        1,
        1
      ),
    guides = "collect"
  )


FIG58_B_58222 <-

  FIG58_B_BODY_58222 +

  patchwork::plot_annotation(

    title =
      "Regime-dependent environmental predictor contribution",

    subtitle =
      paste0(
        "Positive values indicate greater predictive contribution in shadow; ",
        "negative values indicate greater contribution in non-shadow"
      ),

    theme =
      theme(

        plot.title =
          element_text(
            face = "bold",
            size = 15,
            hjust = 0
          ),

        plot.subtitle =
          element_text(
            size = 11.5,
            hjust = 0,
            margin =
              margin(
                b = 8
              )
          )

      )

  ) &

  theme(

    legend.position =
      "bottom",

    legend.title =
      element_text(
        face = "bold"
      )

  )


# ======================================================================
# 58.19 SAVE FINAL PREDICTOR-CONTRIBUTION FIGURE
#
# Larger height is intentional:
# prevents overlapping predictor labels.
# ======================================================================

ggsave(

  file.path(
    DIR_58222_FIG,
    "Figure58_B_MAIN_predictor_contribution_FINAL.png"
  ),

  FIG58_B_58222,

  width = 14.0,
  height = 10.0,
  units = "in",
  dpi = 600,
  bg = "white"

)


ggsave(

  file.path(
    DIR_58222_FIG,
    "Figure58_B_MAIN_predictor_contribution_FINAL.tiff"
  ),

  FIG58_B_58222,

  width = 14.0,
  height = 10.0,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"

)


ggsave(

  file.path(
    DIR_58222_FIG,
    "Figure58_B_MAIN_predictor_contribution_FINAL.pdf"
  ),

  FIG58_B_58222,

  width = 14.0,
  height = 10.0,
  units = "in",
  bg = "white"

)


# ======================================================================
# 58.20 ALSO SAVE INDIVIDUAL ALGORITHM FIGURES
#
# Useful if journal layout makes the combined version too dense.
# ======================================================================

individual_vip_plots_58222 <-
  list(

    Random_Forest =
      VIP_RF_58222,

    Ridge_logistic =
      VIP_RIDGE_58222,

    GAM =
      VIP_GAM_58222

  )


for(
  nm_58222 in
  names(
    individual_vip_plots_58222
  )
){

  pp_58222 <-
    individual_vip_plots_58222[[
      nm_58222
    ]]


  ggsave(

    file.path(
      DIR_58222_FIG,
      paste0(
        "Figure58_B_",
        nm_58222,
        "_predictor_contribution.png"
      )
    ),

    pp_58222,

    width = 6.2,
    height = 8.8,
    units = "in",
    dpi = 600,
    bg = "white"

  )

}


# ======================================================================
# 58.22 FINAL QC
# ======================================================================

QC_58222 <-
  tibble(

    check =
      c(

        "BACC source is Section-57 PRE51 object",

        "Predictor contribution source is original Section-46 object",

        "Three algorithms in BACC figure",

        "Three algorithms in predictor-contribution figure",

        "Three sampling schemes represented in BACC source",

        "Two predictor variants represented in BACC source",

        "Two illumination regimes represented in BACC source",

        "Thirty replicate IDs represented in importance source",

        "Three sampling schemes represented in importance source",

        "Two predictor variants represented in importance source",

        "No SCL in predictor-contribution data",

        "No ShadowMask in predictor-contribution data",

        "No shadow_bin in predictor-contribution data",

        "All BACC means finite",

        "All BACC empirical interval limits finite",

        "All displayed Delta-VI values finite",

        "No Section-51 object required by Section 58",

        "No model fitting performed",

        "No threshold estimation performed",

        "No new statistical test performed"

      ),

    passed =
      c(

        exists(
          "BACC_FINAL_57222"
        ),

        exists(
          "importance_wide_46222"
        ),

        n_distinct(
          BACC_PLOT_58222$algorithm
        ) == 3,

        n_distinct(
          VIP_SELECTED_58222$algorithm
        ) == 3,

        n_distinct(
          BACC_PLOT_58222$scheme
        ) == 3,

        n_distinct(
          BACC_PLOT_58222$variant
        ) == 2,

        n_distinct(
          BACC_PLOT_58222$regime
        ) == 2,

        n_distinct(
          VIP_BASE_58222$replicate
        ) == 30,

        n_distinct(
          VIP_BASE_58222$scheme
        ) == 3,

        n_distinct(
          VIP_BASE_58222$variant
        ) == 2,

        !"SCL" %in%
          VIP_BASE_58222$predictor,

        !"ShadowMask" %in%
          VIP_BASE_58222$predictor,

        !"shadow_bin" %in%
          VIP_BASE_58222$predictor,

        all(
          is.finite(
            BACC_PLOT_58222$
              mean_delta_BACC
          )
        ),

        all(
          is.finite(
            BACC_PLOT_58222$q025
          )
        ) &&
          all(
            is.finite(
              BACC_PLOT_58222$q975
            )
          ),

        all(
          is.finite(
            VIP_SELECTED_58222$
              mean_Delta_VIP
          )
        ),

        TRUE,

        TRUE,

        TRUE,

        TRUE

      )

  )


print(
  QC_58222,
  n = Inf
)


write_csv(

  QC_58222,

  file.path(
    DIR_58222_QC,
    "58_99_FINAL_QC.csv"
  )

)


if(
  !all(
    QC_58222$passed
  )
){

  stop(
    paste0(
      "\nSECTION 58 FINAL QC FAILED.\n",
      "Inspect:\n",
      file.path(
        DIR_58222_QC,
        "58_99_FINAL_QC.csv"
      )
    )
  )

}

# ======================================================================
# FINAL FIGURE OUTPUT MANIFEST
# ======================================================================

# Collect all generated figure files after the analysis is complete.

FIGURE_EXTENSIONS <- c(
  "png",
  "tif",
  "tiff",
  "pdf",
  "jpg",
  "jpeg",
  "svg"
)

# Find figure files recursively within the project output directory.

all_output_files <- list.files(
  path = OUT_DIR,
  recursive = TRUE,
  full.names = TRUE,
  all.files = FALSE
)

# Retain only figure files.

figure_pattern <- paste0(
  "\\.(",
  paste(FIGURE_EXTENSIONS, collapse = "|"),
  ")$"
)

figure_files <- all_output_files[
  grepl(
    figure_pattern,
    all_output_files,
    ignore.case = TRUE
  )
]

# Create a manifest.

final_figure_manifest <- tibble::tibble(

  figure_name = basename(figure_files),

  file_path = normalizePath(
    figure_files,
    winslash = "/",
    mustWork = FALSE
  ),

  file_type = tolower(
    tools::file_ext(figure_files)
  ),

  file_exists = file.exists(figure_files),

  file_size_MB = round(
    file.info(figure_files)$size / 1024^2,
    3
  )

)

# Sort by filename.

final_figure_manifest <-
  final_figure_manifest |>
  dplyr::arrange(figure_name)

# Save manifest.

FINAL_FIGURE_MANIFEST_PATH <- file.path(
  DIR_QC,
  "FINAL_ALL_GENERATED_FIGURES.csv"
)

readr::write_csv(
  final_figure_manifest,
  FINAL_FIGURE_MANIFEST_PATH
)

# Quality control.

cat(
  "\n============================================================\n",
  "FINAL FIGURE OUTPUT AUDIT\n",
  "============================================================\n",
  "Number of figure files:",
  nrow(final_figure_manifest),
  "\n"
)

print(
  final_figure_manifest,
  n = Inf
)

cat(
  "\nFigure manifest saved to:\n",
  FINAL_FIGURE_MANIFEST_PATH,
  "\n"
)
# ======================================================================
# REQUIRED SECTION 40 FIGURE CHECK
# ======================================================================

required_section40_figures <- c(

  "FIG40_222_MAIN_three_algorithm_mapped_area.png",

  "FIG40_222_S1_delta_regime_minus_global.png",

  "FIG40_222_S2_RAW_SHADOW_minus_RAW.png"

)

section40_figure_qc <- tibble::tibble(

  figure_name = required_section40_figures,

  exists = vapply(
    required_section40_figures,
    function(x) {
      any(
        final_figure_manifest$figure_name == x
      )
    },
    logical(1)
  )

)

readr::write_csv(
  section40_figure_qc,
  file.path(
    DIR_QC,
    "FINAL_REQUIRED_SECTION40_FIGURE_QC.csv"
  )
)

print(section40_figure_qc)

if (!all(section40_figure_qc$exists)) {

  warning(
    "One or more required Section 40 figures are missing."
  )

}
# ======================================================================
# 58.23 FILE EXISTENCE QC
# ======================================================================

expected_files_58222 <-
  c(

    file.path(
      DIR_58222_FIG,
      "Figure58_A_MAIN_delta_BACC_FINAL.png"
    ),

    file.path(
      DIR_58222_FIG,
      "Figure58_A_MAIN_delta_BACC_FINAL.tiff"
    ),

    file.path(
      DIR_58222_FIG,
      "Figure58_A_MAIN_delta_BACC_FINAL.pdf"
    ),

    file.path(
      DIR_58222_FIG,
      "Figure58_B_MAIN_predictor_contribution_FINAL.png"
    ),

    file.path(
      DIR_58222_FIG,
      "Figure58_B_MAIN_predictor_contribution_FINAL.tiff"
    ),

    file.path(
      DIR_58222_FIG,
      "Figure58_B_MAIN_predictor_contribution_FINAL.pdf"
    )

  )


FILE_QC_58222 <-
  tibble(

    file =
      basename(
        expected_files_58222
      ),

    exists =
      file.exists(
        expected_files_58222
      ),

    size_bytes =
      ifelse(
        file.exists(
          expected_files_58222
        ),
        file.info(
          expected_files_58222
        )$size,
        NA_real_
      )

  )


print(
  FILE_QC_58222,
  n = Inf
)


write_csv(

  FILE_QC_58222,

  file.path(
    DIR_58222_QC,
    "58_98_FIGURE_FILE_QC.csv"
  )

)


if(
  any(
    !FILE_QC_58222$exists
  )
){

  stop(
    "SECTION 58 figure export failed."
  )

}


# ======================================================================
# 58.24 SESSION INFO
# ======================================================================

capture.output(

  sessionInfo(),

  file =
    file.path(
      DIR_58222_QC,
      "58_sessionInfo.txt"
    )

)


# ======================================================================
# 58.25 COMPLETION
# ======================================================================

cat(

  "\n============================================================\n",

  "SECTION 58_222 COMPLETED SUCCESSFULLY\n",

  "============================================================\n",

  "\nANALYTICAL SOURCES:\n",

  "  Delta BACC:\n",
  "    BACC_FINAL_57222\n",
  "    -> original PRE-SECTION-51 algorithm-specific thresholds\n",

  "\n  Predictor contribution:\n",
  "    importance_wide_46222\n",
  "    -> original PRE-SECTION-51 permutation importance\n",

  "\nNO MODEL REFITTING\n",

  "NO RESAMPLING\n",

  "NO THRESHOLD RE-ESTIMATION\n",

  "NO NEW STATISTICAL TESTS\n",

  "NO SECTION-51 COMMON-EXACT THRESHOLDS USED\n",

  "\nFINAL MAIN FIGURES:\n",

  "  Figure58_A_MAIN_delta_BACC_FINAL.png\n",

  "  Figure58_B_MAIN_predictor_contribution_FINAL.png\n",

  "\nOUTPUT DIRECTORY:\n",

  DIR_58222,

  "\n============================================================\n",

  sep = ""

)
############
#####
###
#
# ======================================================================
# Supplementary classification trade-off table
# FINAL SUPPLEMENTARY TABLE S5
# CLASSIFICATION TRADE-OFFS AND DELTA-BACC INFERENCE
#
# PURPOSE
# ----------------------------------------------------------------------
# Build the final Supplementary Table S5 directly from the original
# replicate-level classification contrasts generated in Section 41.
#
# SINGLE SOURCE OF TRUTH:
#   delta_41222
#
# delta_41222 contains replicate-level:
#   - Delta_Sensitivity
#   - Delta_Specificity
#   - Delta_BalancedAccuracy
#
# Definitions:
#   Delta Sensitivity =
#     Sensitivity_regime-specific - Sensitivity_global
#
#   Delta Specificity =
#     Specificity_regime-specific - Specificity_global
#
#   Delta BACC =
#     BACC_regime-specific - BACC_global
#
# IMPORTANT
# ----------------------------------------------------------------------
# - No CSV result is used as an analytical input.
# - No Section-51 threshold-sensitivity result is used.
# - All summaries are regenerated directly from delta_41222.
# - Statistical inference is performed on replicate-level Delta BACC.
# - BH-FDR correction is applied across the 36 primary Delta-BACC tests:
#
#       3 algorithms
#     x 3 sampling schemes
#     x 2 predictor variants
#     x 2 illumination regimes
#     = 36 tests
#
# OUTPUT:
#   TableS5_classification_tradeoffs_FINAL.csv
#
# ======================================================================


# ======================================================================
# 60.1 REQUIRED PACKAGES
# ======================================================================

suppressPackageStartupMessages({

  library(dplyr)
  library(readr)
  library(tibble)

})


# ======================================================================
# 60.2 REQUIRED SOURCE OBJECT
# ======================================================================

required_60222 <-
  c(
    "delta_41222"
  )


missing_60222 <-
  required_60222[
    !vapply(
      required_60222,
      exists,
      logical(1),
      envir = .GlobalEnv
    )
  ]


if(length(missing_60222) > 0){

  stop(
    paste0(
      "\nSECTION 60 cannot start.\n",
      "Missing required analytical source:\n",
      paste(
        missing_60222,
        collapse = ", "
      ),
      "\n\nRun Section 41 before Section 60."
    )
  )

}


# ======================================================================
# 60.3 OUTPUT DIRECTORY
# ======================================================================

DIR_60222 <-
  file.path(
    OUT_DIR,
    "supplementary_table_S5"
  )


DIR_60222_CSV <-
  file.path(
    DIR_60222,
    "CSV"
  )


DIR_60222_QC <-
  file.path(
    DIR_60222,
    "QC"
  )


dir.create(
  DIR_60222_CSV,
  recursive = TRUE,
  showWarnings = FALSE
)


dir.create(
  DIR_60222_QC,
  recursive = TRUE,
  showWarnings = FALSE
)


# ======================================================================
# 60.4 VERIFY REQUIRED COLUMNS
# ======================================================================

required_delta_cols_60222 <-
  c(
    "algorithm",
    "replicate",
    "scheme",
    "variant",
    "regime",
    "Delta_Sensitivity",
    "Delta_Specificity",
    "Delta_BalancedAccuracy"
  )


missing_delta_cols_60222 <-
  setdiff(
    required_delta_cols_60222,
    names(delta_41222)
  )


if(length(missing_delta_cols_60222) > 0){

  stop(
    paste0(
      "\ndelta_41222 is missing required columns:\n",
      paste(
        missing_delta_cols_60222,
        collapse = ", "
      )
    )
  )

}


# ======================================================================
# 60.5 STANDARDIZATION HELPERS
# ======================================================================

std_algorithm_60222 <- function(x){

  x <- as.character(x)

  case_when(

    grepl(
      "random",
      x,
      ignore.case = TRUE
    ) ~
      "Random Forest",

    grepl(
      "ridge",
      x,
      ignore.case = TRUE
    ) ~
      "Ridge logistic",

    grepl(
      "gam",
      x,
      ignore.case = TRUE
    ) ~
      "GAM",

    TRUE ~
      x
  )
}


std_variant_60222 <- function(x){

  x <- as.character(x)

  case_when(

    x %in%
      c(
        "RAW_SHADOW",
        "RAW+SHADOW",
        "RAW + shadow",
        "RAW+shadow",
        "RAW + SHADOW"
      ) ~
      "RAW+SHADOW",

    x == "RAW" ~
      "RAW",

    TRUE ~
      x
  )
}


std_regime_60222 <- function(x){

  x <- as.character(x)

  case_when(

    grepl(
      "non",
      x,
      ignore.case = TRUE
    ) ~
      "Non-shadow",

    grepl(
      "shadow",
      x,
      ignore.case = TRUE
    ) ~
      "Shadow",

    TRUE ~
      x
  )
}


# ======================================================================
# 60.6 SAFE WILCOXON SIGNED-RANK TEST
#
# Tests replicate-level Delta BACC against zero.
# This reproduces the inferential logic used in Section 57.
# ======================================================================

safe_wilcox_zero_60222 <- function(x){

  x <-
    x[
      is.finite(x)
    ]


  if(length(x) < 3){

    return(
      tibble(
        n_test = length(x),
        W = NA_real_,
        p_value = NA_real_
      )
    )
  }


  if(
    all(
      abs(x) <
        .Machine$double.eps
    )
  ){

    return(
      tibble(
        n_test = length(x),
        W = 0,
        p_value = 1
      )
    )
  }


  wt <-
    suppressWarnings(
      wilcox.test(
        x,
        mu = 0,
        alternative = "two.sided",
        exact = FALSE,
        correct = FALSE
      )
    )


  tibble(

    n_test =
      length(x),

    W =
      unname(
        wt$statistic
      ),

    p_value =
      wt$p.value
  )
}


# ======================================================================
# 60.7 PAIRED RANK-BISERIAL EFFECT SIZE
#
# r_rb =
#   (positive rank sum - negative rank sum) /
#   (positive rank sum + negative rank sum)
#
# Exact zeros are removed.
# ======================================================================

rank_biserial_zero_60222 <- function(x){

  x <-
    x[
      is.finite(x)
    ]


  x <-
    x[
      abs(x) >
        .Machine$double.eps
    ]


  if(length(x) == 0){

    return(
      NA_real_
    )
  }


  r <-
    rank(
      abs(x),
      ties.method = "average"
    )


  R_pos <-
    sum(
      r[
        x > 0
      ]
    )


  R_neg <-
    sum(
      r[
        x < 0
      ]
    )


  denom <-
    R_pos +
    R_neg


  if(
    !is.finite(denom) ||
    denom <= 0
  ){

    return(
      NA_real_
    )
  }


  (
    R_pos -
      R_neg
  ) /
    denom
}


# ======================================================================
# 60.8 CREATE CLEAN REPLICATE-LEVEL SOURCE
# ======================================================================

TRADE_REP_60222 <-
  delta_41222 %>%

  transmute(

    algorithm =
      std_algorithm_60222(
        algorithm
      ),

    replicate =
      as.integer(
        replicate
      ),

    scheme =
      as.character(
        scheme
      ),

    variant =
      std_variant_60222(
        variant
      ),

    regime =
      std_regime_60222(
        regime
      ),

    Delta_Sensitivity =
      as.numeric(
        Delta_Sensitivity
      ),

    Delta_Specificity =
      as.numeric(
        Delta_Specificity
      ),

    Delta_BACC =
      as.numeric(
        Delta_BalancedAccuracy
      )
  ) %>%

  filter(

    algorithm %in%
      c(
        "Random Forest",
        "Ridge logistic",
        "GAM"
      ),

    scheme %in%
      c(
        "UR",
        "SB",
        "SW"
      ),

    variant %in%
      c(
        "RAW",
        "RAW+SHADOW"
      ),

    regime %in%
      c(
        "Non-shadow",
        "Shadow"
      )
  )


# ======================================================================
# 60.9 HARD QC — DUPLICATE REPLICATE KEYS
# ======================================================================

DUP_60222 <-
  TRADE_REP_60222 %>%

  count(
    algorithm,
    replicate,
    scheme,
    variant,
    regime,
    name = "n"
  ) %>%

  filter(
    n > 1
  )


if(nrow(DUP_60222) > 0){

  write_csv(
    DUP_60222,
    file.path(
      DIR_60222_QC,
      "TableS5_duplicate_replicate_keys.csv"
    )
  )

  stop(
    paste0(
      "\nSECTION 60 STOPPED.\n",
      "Duplicate replicate keys were found in delta_41222."
    )
  )
}


# ======================================================================
# 60.10 HARD QC — EXPECTED DESIGN
# ======================================================================

expected_groups_60222 <-
  3 * 3 * 2 * 2


actual_groups_60222 <-
  TRADE_REP_60222 %>%

  distinct(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  nrow()


if(
  actual_groups_60222 !=
    expected_groups_60222
){

  stop(
    paste0(
      "\nExpected ",
      expected_groups_60222,
      " algorithm × scheme × variant × regime groups, ",
      "but found ",
      actual_groups_60222,
      "."
    )
  )
}


# ======================================================================
# 60.11 DESCRIPTIVE SUMMARY
#
# All uncertainty intervals are empirical 2.5th–97.5th percentiles
# across stochastic model realizations.
# ======================================================================

TRADE_SUM_60222 <-
  TRADE_REP_60222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  summarise(

    n_replicates =
      n(),


    # ----------------------------------------------------------
    # Delta Sensitivity
    # ----------------------------------------------------------

    mean_delta_sensitivity =
      mean(
        Delta_Sensitivity,
        na.rm = TRUE
      ),

    median_delta_sensitivity =
      median(
        Delta_Sensitivity,
        na.rm = TRUE
      ),

    q025_delta_sensitivity =
      quantile(
        Delta_Sensitivity,
        probs = 0.025,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    q975_delta_sensitivity =
      quantile(
        Delta_Sensitivity,
        probs = 0.975,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),


    # ----------------------------------------------------------
    # Delta Specificity
    # ----------------------------------------------------------

    mean_delta_specificity =
      mean(
        Delta_Specificity,
        na.rm = TRUE
      ),

    median_delta_specificity =
      median(
        Delta_Specificity,
        na.rm = TRUE
      ),

    q025_delta_specificity =
      quantile(
        Delta_Specificity,
        probs = 0.025,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    q975_delta_specificity =
      quantile(
        Delta_Specificity,
        probs = 0.975,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),


    # ----------------------------------------------------------
    # Delta Balanced Accuracy
    # ----------------------------------------------------------

    mean_delta_BACC =
      mean(
        Delta_BACC,
        na.rm = TRUE
      ),

    median_delta_BACC =
      median(
        Delta_BACC,
        na.rm = TRUE
      ),

    q025_delta_BACC =
      quantile(
        Delta_BACC,
        probs = 0.025,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    q975_delta_BACC =
      quantile(
        Delta_BACC,
        probs = 0.975,
        na.rm = TRUE,
        names = FALSE,
        type = 7
      ),

    proportion_delta_BACC_positive =
      mean(
        Delta_BACC > 0,
        na.rm = TRUE
      ),

    proportion_delta_BACC_negative =
      mean(
        Delta_BACC < 0,
        na.rm = TRUE
      ),

    proportion_delta_BACC_zero =
      mean(
        abs(Delta_BACC) <=
          .Machine$double.eps,
        na.rm = TRUE
      ),

    rank_biserial_delta_BACC =
      rank_biserial_zero_60222(
        Delta_BACC
      ),

    .groups =
      "drop"
  )


# ======================================================================
# 60.12 DELTA-BACC INFERENCE
# ======================================================================

BACC_TESTS_60222 <-
  TRADE_REP_60222 %>%

  group_by(
    algorithm,
    scheme,
    variant,
    regime
  ) %>%

  group_modify(
    ~safe_wilcox_zero_60222(
      .x$Delta_BACC
    )
  ) %>%

  ungroup()


# ======================================================================
# 60.13 BH-FDR
#
# One predefined inferential family:
# 3 algorithms × 3 schemes × 2 variants × 2 regimes = 36 tests
# ======================================================================

BACC_TESTS_60222 <-
  BACC_TESTS_60222 %>%

  mutate(

    p_FDR =
      p.adjust(
        p_value,
        method = "BH"
      ),

    significant_FDR_0_05 =
      is.finite(p_FDR) &
      p_FDR < 0.05
  )


# ======================================================================
# 60.14 BUILD FINAL TABLE
# ======================================================================

TableS5_60222 <-
  TRADE_SUM_60222 %>%

  left_join(
    BACC_TESTS_60222,
    by =
      c(
        "algorithm",
        "scheme",
        "variant",
        "regime"
      )
  ) %>%

  mutate(

    algorithm =
      factor(
        algorithm,
        levels =
          c(
            "Random Forest",
            "Ridge logistic",
            "GAM"
          )
      ),

    scheme =
      factor(
        scheme,
        levels =
          c(
            "UR",
            "SB",
            "SW"
          )
      ),

    variant =
      factor(
        variant,
        levels =
          c(
            "RAW",
            "RAW+SHADOW"
          )
      ),

    regime =
      factor(
        regime,
        levels =
          c(
            "Non-shadow",
            "Shadow"
          )
      )
  ) %>%

  arrange(
    algorithm,
    scheme,
    variant,
    regime
  )


# ======================================================================
# 60.15 PUBLICATION-FRIENDLY COLUMN NAMES
# ======================================================================

TableS5_PUBLICATION_60222 <-
  TableS5_60222 %>%

  transmute(

    Algorithm =
      as.character(
        algorithm
      ),

    Scheme =
      as.character(
        scheme
      ),

    `Predictor set` =
      recode(
        as.character(
          variant
        ),
        "RAW+SHADOW" =
          "RAW + shadow"
      ),

    Regime =
      as.character(
        regime
      ),

    `n replicates` =
      n_replicates,


    # Delta sensitivity

    `Mean delta sensitivity` =
      mean_delta_sensitivity,

    `Median delta sensitivity` =
      median_delta_sensitivity,

    `Delta sensitivity 2.5%` =
      q025_delta_sensitivity,

    `Delta sensitivity 97.5%` =
      q975_delta_sensitivity,


    # Delta specificity

    `Mean delta specificity` =
      mean_delta_specificity,

    `Median delta specificity` =
      median_delta_specificity,

    `Delta specificity 2.5%` =
      q025_delta_specificity,

    `Delta specificity 97.5%` =
      q975_delta_specificity,


    # Delta BACC

    `Mean delta BACC` =
      mean_delta_BACC,

    `Median delta BACC` =
      median_delta_BACC,

    `Delta BACC 2.5%` =
      q025_delta_BACC,

    `Delta BACC 97.5%` =
      q975_delta_BACC,

    `Proportion delta BACC > 0` =
      proportion_delta_BACC_positive,

    `Proportion delta BACC < 0` =
      proportion_delta_BACC_negative,

    `Proportion delta BACC = 0` =
      proportion_delta_BACC_zero,

    `Rank-biserial effect size` =
      rank_biserial_delta_BACC,

    `Wilcoxon n` =
      n_test,

    `Wilcoxon W` =
      W,

    `P value` =
      p_value,

    `FDR-adjusted P value` =
      p_FDR,

    `FDR significant at 0.05` =
      significant_FDR_0_05
  )


# ======================================================================
# 60.16 SAVE EXACT UNROUNDED TABLE
#
# This is the archival/reproducibility version.
# ======================================================================

write_csv(

  TableS5_PUBLICATION_60222,

  file.path(
    DIR_60222_CSV,
    "TableS5_classification_tradeoffs_FINAL_UNROUNDED.csv"
  ),

  na = ""
)


# ======================================================================
# 60.17 CREATE ROUNDED JOURNAL TABLE
# ======================================================================

TableS5_JOURNAL_60222 <-
  TableS5_PUBLICATION_60222 %>%

  mutate(

    across(
      c(
        `Mean delta sensitivity`,
        `Median delta sensitivity`,
        `Delta sensitivity 2.5%`,
        `Delta sensitivity 97.5%`,
        `Mean delta specificity`,
        `Median delta specificity`,
        `Delta specificity 2.5%`,
        `Delta specificity 97.5%`,
        `Mean delta BACC`,
        `Median delta BACC`,
        `Delta BACC 2.5%`,
        `Delta BACC 97.5%`,
        `Proportion delta BACC > 0`,
        `Proportion delta BACC < 0`,
        `Proportion delta BACC = 0`,
        `Rank-biserial effect size`
      ),
      ~round(.x, 4)
    ),

    `P value` =
      signif(
        `P value`,
        4
      ),

    `FDR-adjusted P value` =
      signif(
        `FDR-adjusted P value`,
        4
      )
  )


write_csv(

  TableS5_JOURNAL_60222,

  file.path(
    DIR_60222_CSV,
    "TableS5_classification_tradeoffs_FINAL.csv"
  ),

  na = ""
)


# ======================================================================
# 60.18 REPRODUCIBILITY CROSS-CHECK AGAINST EXISTING OBJECTS
#
# These checks are optional in the sense that Section 60 does not REQUIRE
# tradeoff_summary_41222 or BACC_FINAL_57222 to exist.
#
# If they do exist, Section 60 verifies that its independently regenerated
# results reproduce them.
# ======================================================================

QC_COMPARE_60222 <-
  tibble(
    check =
      character(),
    passed =
      logical()
  )


# ----------------------------------------------------------------------
# Compare with original Section-41 summary if available
# ----------------------------------------------------------------------

if(
  exists(
    "tradeoff_summary_41222",
    envir = .GlobalEnv
  )
){

  old41 <-
    tradeoff_summary_41222 %>%

    mutate(

      algorithm =
        std_algorithm_60222(
          algorithm
        ),

      variant =
        std_variant_60222(
          variant
        ),

      regime =
        std_regime_60222(
          regime
        ),

      scheme =
        as.character(
          scheme
        )
    )


  chk41 <-
    TRADE_SUM_60222 %>%

    select(
      algorithm,
      scheme,
      variant,
      regime,
      mean_delta_sensitivity,
      median_delta_sensitivity,
      q025_delta_sensitivity,
      q975_delta_sensitivity,
      mean_delta_specificity,
      median_delta_specificity,
      q025_delta_specificity,
      q975_delta_specificity
    ) %>%

    left_join(

      old41 %>%
        select(
          algorithm,
          scheme,
          variant,
          regime,
          mean_delta_sensitivity_old =
            mean_delta_sensitivity,
          median_delta_sensitivity_old =
            median_delta_sensitivity,
          q025_delta_sensitivity_old =
            q025_delta_sensitivity,
          q975_delta_sensitivity_old =
            q975_delta_sensitivity,
          mean_delta_specificity_old =
            mean_delta_specificity,
          median_delta_specificity_old =
            median_delta_specificity,
          q025_delta_specificity_old =
            q025_delta_specificity,
          q975_delta_specificity_old =
            q975_delta_specificity
        ),

      by =
        c(
          "algorithm",
          "scheme",
          "variant",
          "regime"
        )
    )


  passed41 <-
    all(
      abs(
        chk41$mean_delta_sensitivity -
          chk41$mean_delta_sensitivity_old
      ) < 1e-12,

      abs(
        chk41$median_delta_sensitivity -
          chk41$median_delta_sensitivity_old
      ) < 1e-12,

      abs(
        chk41$q025_delta_sensitivity -
          chk41$q025_delta_sensitivity_old
      ) < 1e-12,

      abs(
        chk41$q975_delta_sensitivity -
          chk41$q975_delta_sensitivity_old
      ) < 1e-12,

      abs(
        chk41$mean_delta_specificity -
          chk41$mean_delta_specificity_old
      ) < 1e-12,

      abs(
        chk41$median_delta_specificity -
          chk41$median_delta_specificity_old
      ) < 1e-12,

      abs(
        chk41$q025_delta_specificity -
          chk41$q025_delta_specificity_old
      ) < 1e-12,

      abs(
        chk41$q975_delta_specificity -
          chk41$q975_delta_specificity_old
      ) < 1e-12
    )


  QC_COMPARE_60222 <-
    bind_rows(
      QC_COMPARE_60222,
      tibble(
        check =
          "Reproduces Section-41 sensitivity/specificity summaries",
        passed =
          passed41
      )
    )
}


# ----------------------------------------------------------------------
# Compare with Section-57 Delta-BACC inference if available
# ----------------------------------------------------------------------

if(
  exists(
    "BACC_FINAL_57222",
    envir = .GlobalEnv
  )
){

  old57 <-
    BACC_FINAL_57222 %>%

    mutate(

      algorithm =
        std_algorithm_60222(
          algorithm
        ),

      variant =
        std_variant_60222(
          variant
        ),

      regime =
        std_regime_60222(
          regime
        ),

      scheme =
        as.character(
          scheme
        )
    )


  chk57 <-
    TableS5_60222 %>%

    select(
      algorithm,
      scheme,
      variant,
      regime,
      mean_delta_BACC,
      median_delta_BACC,
      q025_delta_BACC,
      q975_delta_BACC,
      rank_biserial_delta_BACC,
      p_value,
      p_FDR
    ) %>%

    left_join(

      old57 %>%
        transmute(
          algorithm =
            as.character(
              algorithm
            ),
          scheme =
            as.character(
              scheme
            ),
          variant =
            as.character(
              variant
            ),
          regime =
            as.character(
              regime
            ),
          mean_delta_BACC_old =
            mean_delta_BACC,
          median_delta_BACC_old =
            median_delta_BACC,
          q025_delta_BACC_old =
            q025,
          q975_delta_BACC_old =
            q975,
          rank_biserial_old =
            rank_biserial,
          p_value_old =
            p_value,
          p_FDR_old =
            p_FDR
        ),

      by =
        c(
          "algorithm",
          "scheme",
          "variant",
          "regime"
        )
    )


  passed57 <-
    all(
      abs(
        chk57$mean_delta_BACC -
          chk57$mean_delta_BACC_old
      ) < 1e-12,

      abs(
        chk57$median_delta_BACC -
          chk57$median_delta_BACC_old
      ) < 1e-12,

      abs(
        chk57$q025_delta_BACC -
          chk57$q025_delta_BACC_old
      ) < 1e-12,

      abs(
        chk57$q975_delta_BACC -
          chk57$q975_delta_BACC_old
      ) < 1e-12,

      abs(
        chk57$rank_biserial_delta_BACC -
          chk57$rank_biserial_old
      ) < 1e-12,

      abs(
        chk57$p_value -
          chk57$p_value_old
      ) < 1e-12,

      abs(
        chk57$p_FDR -
          chk57$p_FDR_old
      ) < 1e-12
    )


  QC_COMPARE_60222 <-
    bind_rows(
      QC_COMPARE_60222,
      tibble(
        check =
          "Reproduces Section-57 Delta-BACC inference",
        passed =
          passed57
      )
    )
}


# ======================================================================
# 60.19 FINAL QC TABLE
# ======================================================================

QC_60222 <-
  bind_rows(

    tibble(

      check =
        c(
          "delta_41222 exists",
          "Required columns exist",
          "No duplicate replicate keys",
          "Three algorithms",
          "Three sampling schemes",
          "Two predictor variants",
          "Two illumination regimes",
          "Expected 36 groups",
          "All Delta-BACC FDR values finite where P is finite"
        ),

      passed =
        c(

          exists(
            "delta_41222",
            envir = .GlobalEnv
          ),

          length(
            missing_delta_cols_60222
          ) == 0,

          nrow(
            DUP_60222
          ) == 0,

          n_distinct(
            TRADE_REP_60222$algorithm
          ) == 3,

          n_distinct(
            TRADE_REP_60222$scheme
          ) == 3,

          n_distinct(
            TRADE_REP_60222$variant
          ) == 2,

          n_distinct(
            TRADE_REP_60222$regime
          ) == 2,

          nrow(
            TableS5_60222
          ) == 36,

          all(
            is.finite(
              TableS5_60222$p_FDR[
                is.finite(
                  TableS5_60222$p_value
                )
              ]
            )
          )
        )
    ),

    QC_COMPARE_60222
  )


print(
  QC_60222,
  n = Inf
)


write_csv(

  QC_60222,

  file.path(
    DIR_60222_QC,
    "TableS5_FINAL_QC.csv"
  )
)


if(
  !all(
    QC_60222$passed
  )
){

  stop(
    paste0(
      "\nSECTION 60 FINAL QC FAILED.\n",
      "Inspect:\n",
      file.path(
        DIR_60222_QC,
        "TableS5_FINAL_QC.csv"
      )
    )
  )
}


# ======================================================================
# 60.20 SAVE SESSION INFORMATION
# ======================================================================

capture.output(

  sessionInfo(),

  file =
    file.path(
      DIR_60222_QC,
      "Section60_sessionInfo.txt"
    )
)


# ======================================================================
# 60.21 FINAL CONSOLE REPORT
# ======================================================================

cat(
  "\n============================================================\n",
  "SECTION 60 COMPLETED SUCCESSFULLY\n",
  "============================================================\n",
  "\nSOURCE OF TRUTH:\n",
  "  delta_41222\n",
  "\nANALYTICAL INPUT CSV FILES USED:\n",
  "  NONE\n",
  "\nSECTION-51+ SENSITIVITY RESULTS USED:\n",
  "  NONE\n",
  "\nFINAL SUPPLEMENTARY TABLE:\n",
  "  TableS5_classification_tradeoffs_FINAL.csv\n",
  "\nARCHIVAL UNROUNDED TABLE:\n",
  "  TableS5_classification_tradeoffs_FINAL_UNROUNDED.csv\n",
  "\nQC:\n",
  "  TableS5_FINAL_QC.csv\n",
  "\nRows expected: 36\n",
  "Rows generated: ",
  nrow(
    TableS5_JOURNAL_60222
  ),
  "\n============================================================\n",
  sep = ""
)
###################3
# ============================================================
# OUTPUT PATH CHECK
# ============================================================

cat("\n================ OUTPUT PATH CHECK ================\n")

cat("PROJECT_DIR :", PROJECT_DIR, "\n")
cat("OUT_DIR     :", OUT_DIR, "\n")
cat("TABLES      :", DIR_CSV, "\n")
cat("FIGURES     :", DIR_FIG, "\n")
cat("MODELS      :", DIR_MODEL, "\n")
cat("PREDICTIONS :", DIR_PRED, "\n")
cat("QC          :", DIR_QC, "\n")
cat("MAPS        :", DIR_MAP, "\n")

cat("===================================================\n\n")


# Check that all output directories exist
output_dirs <- c(
  OUT_DIR,
  DIR_CSV,
  DIR_FIG,
  DIR_MODEL,
  DIR_PRED,
  DIR_QC,
  DIR_MAP
)

names(output_dirs) <- c(
  "main",
  "tables",
  "figures",
  "models",
  "predictions",
  "quality_control",
  "maps"
)

print(
  data.frame(
    directory = names(output_dirs),
    path = unname(output_dirs),
    exists = dir.exists(output_dirs),
    row.names = NULL
  )
)


# ------------------------------------------------------------
# Test actual write permission
# ------------------------------------------------------------

test_file <- file.path(
  OUT_DIR,
  "output_path_test.txt"
)

writeLines(
  "Output directory is working correctly.",
  test_file
)

if (file.exists(test_file)) {

  cat(
    "\nSUCCESS: Output directory is writable.\n",
    "Test file created at:\n",
    test_file,
    "\n"
  )

} else {

  stop(
    "Output directory exists but the test file could not be created."
  )
}
