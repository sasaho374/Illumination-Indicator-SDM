**Evaluating remotely sensed illumination regime as an ecological indicator in species distribution models**

This repository provides Google Earth Engine (GEE) JavaScript and R scripts used to examine illumination-related information in species distribution modelling (SDM), with *Juniperus polycarpos var. turcomanica* in the Hezarmasjed Mountains, northeastern Iran, as the case study.

The repository is intended to make the analysis code inspectable and to document the inputs needed to run it. **It is not a complete, self-contained reproduction package:** the large environmental predictor raster required by several R scripts is not distributed here.

## Repository contents

```text
Illumination-Indicator-SDM/
├── Scripts/
│   ├── 01_GEE_Sentinel2_preprocessing.js
│   ├── 02_GEE_SCL2_temporal_stability_validation.js
│   ├── 03_Shadow vs Topography_analysis.R
│   ├── 04_Variogram.R
│   ├── 05_blockcv.R
│   └── 06_main analysis2.R
├── data/
│   ├── training_points.gpkg
│   ├── external_test_points.gpkg
│   └── master_points.gpkg
├── .gitignore
└── README.md
```

### Script descriptions

| Script | Purpose |
| --- | --- |
| `01_GEE_Sentinel2_preprocessing.js` | Processes Sentinel-2 Level-2A imagery across seven November–April seasons (2018–2019 through 2024–2025), builds seasonal and hybrid composites, and exports remote-sensing products. |
| `02_GEE_SCL2_temporal_stability_validation.js` | Evaluates temporal properties of the Sentinel-2 SCL-based shadow information. |
| `03_Shadow vs Topography_analysis.R` | Analyses relationships between shadow and terrain variables, with spatial diagnostics and model outputs. |
| `04_Variogram.R` | Examines spatial autocorrelation to support the spatial partitioning workflow. |
| `05_blockcv.R` | Implements spatial blocking / cross-validation preparation. |
| `06_main analysis2.R` | Runs the main SDM comparisons, evaluation, and publication-output generation. |

The scripts represent related workflow components, but **not every input is automatically produced by the preceding numbered script**. In particular, creation of the final 46-band raster stack is not documented as a single end-to-end build step in this repository.

## Data requirements and availability

The repository includes three small GeoPackage files:

- `data/training_points.gpkg` — modelling training observations.
- `data/external_test_points.gpkg` — independent external-test observations.
- `data/master_points.gpkg` — spatial input used by the variogram and block-CV scripts.

The following required raster is **not included**:

- `data/predictor_stack_46bands.tif` — 46-band environmental predictor stack (approximately 2.65 GB).

The R modelling and spatial-analysis scripts require this raster at the exact relative path shown above. **Without it, the full analyses cannot be rerun from the files in this repository alone.** The available GEE scripts document relevant remote-sensing processing, but should not be assumed to reconstruct the complete 46-band raster without additional processing and source data.

Before redistributing occurrence coordinates, users should consider any applicable permissions and conservation sensitivities.

## Software

- [Google Earth Engine](https://earthengine.google.com/) for the `.js` scripts.
- R and an R environment such as RStudio for the `.R` scripts.

R packages referenced by the scripts include:

```r
install.packages(c(
  "sf", "terra", "data.table", "dplyr", "tidyr", "tibble",
  "readr", "purrr", "stringr", "ggplot2", "ranger", "pROC",
  "glmnet", "mgcv", "spdep", "gstat", "blockCV"
))
```

Package installation may require additional system dependencies; this list is not a version-locked software environment.

## Workflow order and preparation of the raster inputs

**Important: complete the required remote-sensing and environmental-data preparation before running the R analyses.** The main R script cannot run without the final 46-band predictor raster.

1. Open `Scripts/01_GEE_Sentinel2_preprocessing.js` in the Google Earth Engine Code Editor. Check the study-area assets, input collections, processing settings, and export destinations before running it.
2. Run the GEE preprocessing script, start the necessary export tasks, and download the resulting remote-sensing raster products once the exports have completed.
3. Use `Scripts/02_GEE_SCL2_temporal_stability_validation.js` to examine the temporal stability of the SCL-based shadow information. This is a validation component, not a substitute for preparing the predictor stack.
4. Prepare the **final 46-band environmental predictor stack** from the required remote-sensing and other environmental layers, ensuring the expected band names, order, projection, resolution, extent, and values match those used by the R scripts.
5. Save the finished stack as `data/predictor_stack_46bands.tif` under the repository root. The raster is **not distributed** in this repository.
6. Confirm that `data/training_points.gpkg` and `data/external_test_points.gpkg` are present. The `data/master_points.gpkg` file is supplied for relevant spatial analyses.
7. Install the required R packages, set the R working directory to the **repository root** (containing `Scripts/` and `data/`), and run the relevant R scripts after checking their path settings.

**Critical limitation:** Running script 01 and downloading its exports is **not confirmed to produce `predictor_stack_46bands.tif` directly**. The repository does not currently provide a verified, complete procedure for assembling all 46 bands from the GEE exports and other environmental sources. Accordingly, the above steps document the intended workflow and required input, **not a fully automated end-to-end reconstruction** of the omitted raster. Do not rename a partial GEE output to `predictor_stack_46bands.tif` unless it has been checked against the required 46-band specification.

For the main analysis:

```r
# Run from the repository root:
source("Scripts/06_main analysis2.R")
```

**Path configuration:** `06_main analysis2.R` uses the current working directory as `PROJECT_DIR`. The scripts `03_Shadow vs Topography_analysis.R`, `04_Variogram.R`, and `05_blockcv.R` contain a machine-specific absolute `PROJECT_DIR` path in their current versions. Before running those scripts on another computer, replace their `PROJECT_DIR <- normalizePath(...)` assignment with:

```r
PROJECT_DIR <- normalizePath(".", winslash = "/", mustWork = TRUE)
```

Then run them from the repository root. The GEE scripts should be opened and configured in the Earth Engine Code Editor; check their study-area assets and export destinations before execution.

### Main SDM design

The main R script compares three sampling schemes:

- **UR:** shadow representation based on landscape prevalence.
- **SB:** shadow representation based on the fitting-pool prevalence.
- **SW:** the same sampled observations as SB, weighted toward landscape shadow prevalence.

Within each scheme, two predictor configurations are evaluated:

- **RAW:** environmental predictors without the binary shadow indicator.
- **RAW+SHADOW:** the same predictors plus `shadow_bin`.

The script implements Random Forest, ridge-regularized logistic regression, and generalized additive models. It uses a fixed random seed (`123`), 30 stochastic replicates, and a 20% calibration split. Calibration observations are separated before resampling, and the independent external test set is reserved for evaluation rather than model fitting or threshold selection.

The script is configured to write outputs under `outputs_publication/`, including tables, figures, predictions, models, and quality-control material. Some optional outputs depend on script settings; for example, raster-ensemble generation is disabled by default (`DO_RASTER_ENSEMBLE <- FALSE`).

## Reproducibility scope

The code and included point datasets are available for inspection and reuse, subject to the repository's stated terms. Full computational reproduction additionally requires the omitted predictor raster, compatible software dependencies, and any external Earth Engine assets used during preprocessing. No claim is made that the GitHub contents alone are sufficient to regenerate every published result.

## Citation

If using these scripts, please cite the associated article once its bibliographic details are available:

*Evaluating remotely sensed illumination regime as an ecological indicator in species distribution models.*

**Article DOI:** To be added after publication.

## Questions

Please open an issue in this GitHub repository for questions about the scripts or input requirements.
