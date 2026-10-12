# ======================================================================
# Spatial partition and analytical train/test datasets
#
# This script documents the spatial blocking and train/test partition
# used in the study.
#
# Original spatial-block settings:
#   Random seed    = 42
#   Block size     = 2500 m
#   Number of folds = 6
#   Fold selection = random
#   Iterations     = 200
#   External test  = Fold 2
# ======================================================================


# ============================================================
# 1. Required packages
# ============================================================

required_packages <- c(
  "sf",
  "blockCV",
  "dplyr",
  "ggplot2"
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
  library(blockCV)
  library(dplyr)
  library(ggplot2)
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

stopifnot(
  file.exists(MASTER_PATH)
)


# ============================================================
# 3. Read master dataset
# ============================================================

pts <- sf::st_read(
  MASTER_PATH,
  quiet = TRUE
)

stopifnot(
  "presence" %in% names(pts),
  "shadow_raster" %in% names(pts),
  nrow(pts) == 580
)

cat(
  "\nMaster observations:",
  nrow(pts),
  "\n"
)


# ============================================================
# 4. Generate spatial blocks
# ============================================================

set.seed(42)

sb <- blockCV::spatialBlock(
  speciesData = pts,
  species     = "presence",
  theRange    = 2500,
  k           = 6,
  selection   = "random",
  iteration   = 200,
  showBlocks  = FALSE,
  progress    = TRUE
)

pts$foldID <- sb$foldID


# ============================================================
# 5. Fold composition
# ============================================================

qc_table <- pts |>
  sf::st_drop_geometry() |>
  dplyr::mutate(
    presence_lab = ifelse(
      presence == 1,
      "Presence",
      "Absence"
    ),
    shadow_lab = ifelse(
      shadow_raster == 1,
      "Shadow",
      "Non-shadow"
    )
  ) |>
  dplyr::count(
    foldID,
    presence_lab,
    shadow_lab
  ) |>
  dplyr::arrange(
    foldID,
    presence_lab,
    shadow_lab
  )

cat(
  "\n================ FOLD COMPOSITION ================\n"
)

print(
  qc_table
)


qc_summary <- qc_table |>
  dplyr::group_by(
    foldID
  ) |>
  dplyr::summarise(

    Presence_Shadow =
      sum(
        n[
          presence_lab == "Presence" &
          shadow_lab == "Shadow"
        ]
      ),

    Presence_NonShadow =
      sum(
        n[
          presence_lab == "Presence" &
          shadow_lab == "Non-shadow"
        ]
      ),

    Absence_Shadow =
      sum(
        n[
          presence_lab == "Absence" &
          shadow_lab == "Shadow"
        ]
      ),

    Absence_NonShadow =
      sum(
        n[
          presence_lab == "Absence" &
          shadow_lab == "Non-shadow"
        ]
      ),

    Total = sum(n),

    .groups = "drop"
  )


cat(
  "\n================ FOLD SUMMARY ================\n"
)

print(
  qc_summary
)


write.csv(
  qc_table,
  file.path(
    OUTPUT_DIR,
    "spatial_fold_composition_full.csv"
  ),
  row.names = FALSE
)


write.csv(
  qc_summary,
  file.path(
    OUTPUT_DIR,
    "spatial_fold_composition_summary.csv"
  ),
  row.names = FALSE
)


# ============================================================
# 6. Spatial-fold map
# ============================================================

p <- ggplot2::ggplot(
  pts
) +
  ggplot2::geom_sf(
    ggplot2::aes(
      color = factor(foldID)
    ),
    size = 1.6,
    alpha = 0.85
  ) +
  ggplot2::scale_color_brewer(
    palette = "Set1",
    name = "Fold ID"
  ) +
  ggplot2::theme_bw(
    base_size = 14
  ) +
  ggplot2::labs(
    title = "Spatial distribution of blockCV folds",
    subtitle = "Block size = 2.5 km",
    x = NULL,
    y = NULL
  )

print(
  p
)

ggplot2::ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "spatial_fold_map.png"
  ),
  plot = p,
  width = 8,
  height = 7,
  dpi = 300
)


# ============================================================
# 7. Assign Fold 2 as the external test dataset
# ============================================================

external_fold <- 2

pts$dataset <- ifelse(
  pts$foldID == external_fold,
  "test",
  "train"
)

cat(
  "\nExternal test fold:",
  external_fold,
  "\n"
)


# ============================================================
# 8. Train/test composition
# ============================================================

qc_final <- pts |>
  sf::st_drop_geometry() |>
  dplyr::mutate(
    presence_lab = ifelse(
      presence == 1,
      "Presence",
      "Absence"
    ),
    shadow_lab = ifelse(
      shadow_raster == 1,
      "Shadow",
      "Non-shadow"
    )
  ) |>
  dplyr::count(
    dataset,
    presence_lab,
    shadow_lab
  ) |>
  dplyr::arrange(
    dataset,
    presence_lab,
    shadow_lab
  )

cat(
  "\n================ TRAIN / TEST COMPOSITION ================\n"
)

print(
  qc_final
)

cat(
  "\n================ SAMPLE SIZES ================\n"
)

print(
  table(pts$dataset)
)


# ============================================================
# 9. Create training and external-test datasets
# ============================================================

train_sf <- pts |>
  dplyr::filter(
    dataset == "train"
  )

test_sf <- pts |>
  dplyr::filter(
    dataset == "test"
  )


cat(
  "\nTRAIN size:",
  nrow(train_sf),
  "\n"
)

cat(
  "TEST size:",
  nrow(test_sf),
  "\n"
)


# Historical sample-size QC
if(
  nrow(train_sf) != 486 ||
  nrow(test_sf) != 94
){

  warning(
    paste0(
      "Generated partition sizes differ from the archived ",
      "analytical partition (Train = 486; Test = 94)."
    )
  )
}


# ============================================================
# 10. Save partition outputs
# ============================================================

sf::st_write(
  pts,
  file.path(
    OUTPUT_DIR,
    "master_points_with_generated_fold_assignment.gpkg"
  ),
  delete_dsn = TRUE,
  quiet = TRUE
)


sf::st_write(
  train_sf,
  file.path(
    OUTPUT_DIR,
    "training_points_generated.gpkg"
  ),
  delete_dsn = TRUE,
  quiet = TRUE
)


sf::st_write(
  test_sf,
  file.path(
    OUTPUT_DIR,
    "external_test_points_generated.gpkg"
  ),
  delete_dsn = TRUE,
  quiet = TRUE
)


write.csv(
  qc_final,
  file.path(
    OUTPUT_DIR,
    "train_test_composition.csv"
  ),
  row.names = FALSE
)


cat(
  "\nSpatial-partition workflow completed.\n",
  "\nGenerated training observations: ",
  nrow(train_sf),
  "\nGenerated external-test observations: ",
  nrow(test_sf),
  "\n"
)