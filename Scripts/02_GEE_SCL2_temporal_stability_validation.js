// ============================================================================
// Temporal recurrence validation of SCL=2
//
// Purpose:
//   Evaluate whether pixels classified as SCL=2 in the final multiseason
//   hybrid are repeatedly classified as SCL=2 across the seven seasonal
//   windows used in the preprocessing workflow.
//
// Reference product:
//   The seven seasonal median composites are reconstructed and combined using
//   the same qualityMosaic('quality') rule as the preprocessing script.
//
// Seasonal comparison:
//   Within each season, the best-quality individual Sentinel-2 observation is
//   selected independently and compared with the hybrid SCL=2 reference mask.
//
// Interpretation:
//   This analysis quantifies recurrence of the SCL=2 classification. It does
//   not directly measure irradiance, temperature, moisture, or microrefugial
//   function.
//
// Output:
//   Results are printed to the Earth Engine Console. No raster or table export
//   is performed by this script.
// ============================================================================
//
// ============================================================================
// 1. Analysis settings
// ============================================================================

var seasons = [
  {
    start: '2018-11-06',
    end:   '2019-04-04',
    label: '2018_2019',
    index: 1
  },
  {
    start: '2019-11-06',
    end:   '2020-04-04',
    label: '2019_2020',
    index: 2
  },
  {
    start: '2020-11-06',
    end:   '2021-04-04',
    label: '2020_2021',
    index: 3
  },
  {
    start: '2021-11-06',
    end:   '2022-04-04',
    label: '2021_2022',
    index: 4
  },
  {
    start: '2022-11-06',
    end:   '2023-04-04',
    label: '2022_2023',
    index: 5
  },
  {
    start: '2023-11-06',
    end:   '2024-04-04',
    label: '2023_2024',
    index: 6
  },
  {
    start: '2024-11-06',
    end:   '2025-04-04',
    label: '2024_2025',
    index: 7
  }
];

var cloudProbThreshold = 30;
var imageCloudPercent  = 70;
var removeShadows      = true;
var bufferMeters       = 60000;

var analysisCRS   = 'EPSG:32640';

// The native SCL band is 20 m.
// Recurrence is evaluated at 20 m to avoid artificial replication of SCL pixels.
var analysisScale = 20;


// ============================================================================
// 2. Study area
// ============================================================================

var studyArea = ee.Geometry.Polygon([
  [59.171411, 37.157133],
  [59.687512, 36.950649],
  [59.544242, 36.689945],
  [58.973477, 36.913303],
  [59.171411, 37.157133]
]);

var expandedArea = studyArea.buffer(bufferMeters);


// ============================================================================
// 3. Spectral and index bands
// ============================================================================

var reflBands = [
  'B2',
  'B3',
  'B4',
  'B5',
  'B6',
  'B7',
  'B8',
  'B8A',
  'B11',
  'B12'
];

var indexBands = [
  'NDVI',
  'GNDVI',
  'NDMI',
  'NDWI',
  'NBR'
];

var outFloatBands = reflBands
  .concat(['SCL'])
  .concat(indexBands);


// ============================================================================
// 4. Collection join settings
// ============================================================================

var joinFilter = ee.Filter.equals({
  leftField:  'system:index',
  rightField: 'system:index'
});

var innerJoin = ee.Join.inner();


// ============================================================================
// 5. Empty-image templates
// ============================================================================

// Empty image used when no valid observation is available for a season.

var emptyFloat = (function () {

  var output = ee.Image.constant(0)
    .rename(outFloatBands[0])
    .toFloat();

  for (var i = 1; i < outFloatBands.length; i++) {

    output = output.addBands(
      ee.Image.constant(0)
        .rename(outFloatBands[i])
        .toFloat()
    );
  }

  return output.updateMask(ee.Image.constant(0));

})();


var emptySCL = ee.Image.constant(0)
  .rename('SCL')
  .toFloat()
  .updateMask(ee.Image.constant(0));


var emptyCov = ee.Image.constant(0)
  .rename('seasonCov')
  .toInt16()
  .updateMask(ee.Image.constant(0));


var emptySource = ee.Image.constant(0)
  .rename('sourceSeason')
  .toInt16()
  .updateMask(ee.Image.constant(0));


var emptyQuality = ee.Image.constant(0)
  .rename('quality')
  .toFloat()
  .updateMask(ee.Image.constant(0));


// ============================================================================
// 6. SCL validity mask
// ============================================================================

function buildValidMask(image) {

  var scl = image.select('SCL');

  // Exclude invalid data, cloud-related classes, and snow/ice.
  var valid = scl.neq(1)
    .and(scl.neq(7))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  // SCL=3 represents cloud shadow and is excluded as in the preprocessing workflow.
  if (removeShadows) {
    valid = valid.and(scl.neq(3));
  }

  return valid;
}


// ============================================================================
// 7. Spectral-index calculation
// ============================================================================

function toFloatWithIndices(image) {

  var reflectance = image
    .select(reflBands)
    .toFloat();

  var scl = image
    .select('SCL')
    .rename('SCL')
    .toFloat();

  var ndvi = image
    .normalizedDifference(['B8', 'B4'])
    .rename('NDVI')
    .toFloat();

  var gndvi = image
    .normalizedDifference(['B8', 'B3'])
    .rename('GNDVI')
    .toFloat();

  var ndmi = image
    .normalizedDifference(['B8', 'B11'])
    .rename('NDMI')
    .toFloat();

  var ndwi = image
    .normalizedDifference(['B3', 'B11'])
    .rename('NDWI')
    .toFloat();

  var nbr = image
    .normalizedDifference(['B8', 'B12'])
    .rename('NBR')
    .toFloat();

  return reflectance
    .addBands(scl)
    .addBands([
      ndvi,
      gndvi,
      ndmi,
      ndwi,
      nbr
    ])
    .select(outFloatBands);
}


// ============================================================================
// 8. Seasonal product construction
// ============================================================================
//
// Two products are constructed for each seasonal window:
//
// A) seasonalMedian:
//    A) the seasonal median composite used in the preprocessing workflow
//
// B) annualBestSCL:
//    B) the independently selected best-quality observation used for validation
// ============================================================================

function buildSeasonProducts(seasonInfo) {

  var startDate = ee.Date(seasonInfo.start);
  var endDate   = ee.Date(seasonInfo.end);

  var yearNumber = ee.Number(
    startDate.get('year')
  );


  // --------------------------------------------------------------------------
  // 8.1 Sentinel-2 surface reflectance
  // --------------------------------------------------------------------------

  var s2SR = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(expandedArea)
    .filterDate(startDate, endDate)
    .filter(
      ee.Filter.lte(
        'CLOUDY_PIXEL_PERCENTAGE',
        imageCloudPercent
      )
    );


  // --------------------------------------------------------------------------
  // 8.2 Sentinel-2 cloud probability
  // --------------------------------------------------------------------------

  var cloudProbabilityCollection =
    ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
      .filterBounds(expandedArea)
      .filterDate(startDate, endDate);


  // --------------------------------------------------------------------------
  // 8.3 Join SR and cloud probability
  // --------------------------------------------------------------------------

  var joined = innerJoin.apply({
    primary: s2SR,
    secondary: cloudProbabilityCollection,
    condition: joinFilter
  });


  var paired = ee.ImageCollection(
    joined.map(function (joinedFeature) {

      joinedFeature = ee.Feature(joinedFeature);

      var srImage = ee.Image(
        joinedFeature.get('primary')
      );

      var cloudImage = ee.Image(
        joinedFeature.get('secondary')
      );

      var probability = cloudImage
        .select('probability')
        .rename('probability');

      return srImage
        .addBands(probability)
        .copyProperties(
          srImage,
          [
            'system:index',
            'system:time_start',
            'CLOUDY_PIXEL_PERCENTAGE'
          ]
        );
    })
  );


  var pairedCount = paired.size();


  // --------------------------------------------------------------------------
  // 8.4 Collection used for seasonal median
  // --------------------------------------------------------------------------

  var processedForMedian = paired.map(function (image) {

    var validSCL = buildValidMask(image);

    var goodPixel = validSCL.and(
      image
        .select('probability')
        .lt(cloudProbThreshold)
    );

    return toFloatWithIndices(
      image.updateMask(goodPixel)
    );
  });


  // --------------------------------------------------------------------------
  // 8.5 Seasonal median image
  // --------------------------------------------------------------------------

  var seasonalMedian = ee.Image(
    ee.Algorithms.If(
      pairedCount.gt(0),
      processedForMedian.median(),
      emptyFloat
    )
  ).select(outFloatBands);


  var seasonalValidMask = seasonalMedian
    .select('B4')
    .mask();


  // --------------------------------------------------------------------------
  // 8.6 Seasonal observation coverage
  // --------------------------------------------------------------------------

  var seasonCov = ee.Image(
    ee.Algorithms.If(

      pairedCount.gt(0),

      processedForMedian
        .select('B4')
        .count()
        .rename('seasonCov')
        .toInt16(),

      emptyCov
    )
  ).updateMask(seasonalValidMask);


  // --------------------------------------------------------------------------
  // 8.7 Source season
  // --------------------------------------------------------------------------

  var sourceSeason = ee.Image.constant(yearNumber)
    .rename('sourceSeason')
    .toInt16()
    .updateMask(seasonalValidMask);


  // --------------------------------------------------------------------------
  // 8.8 Median cloud probability used in original seasonal quality
  // --------------------------------------------------------------------------

  var validCloudProbability = paired.map(function (image) {

    return image
      .select('probability')
      .updateMask(buildValidMask(image));
  });


  var cloudMedian = ee.Image(
    ee.Algorithms.If(

      pairedCount.gt(0),

      validCloudProbability
        .median()
        .rename('cloudMedian'),

      ee.Image.constant(100)
        .rename('cloudMedian')
    )
  );


  // --------------------------------------------------------------------------
  // 8.9 Original quality score for seasonal median
  // --------------------------------------------------------------------------

  var seasonalCloudScore = ee.Image.constant(100)
    .subtract(cloudMedian);

  var seasonalNDVIScore = seasonalMedian
    .select('NDVI')
    .add(1)
    .multiply(100);

  var seasonalEpsilon = ee.Image.constant(
    yearNumber.subtract(2000)
  ).divide(10000);

  var seasonalQuality = seasonalCloudScore
    .multiply(10000)
    .add(seasonalNDVIScore)
    .add(seasonalEpsilon)
    .rename('quality')
    .toFloat()
    .updateMask(seasonalValidMask);


  // --------------------------------------------------------------------------
  // 8.10 Seasonal composite used by original hybrid
  // --------------------------------------------------------------------------

  var originalSeasonImage = seasonalMedian
    .addBands(seasonCov)
    .addBands(sourceSeason)
    .addBands(seasonalQuality)
    .set({
      seasonStart:       seasonInfo.start,
      seasonEnd:         seasonInfo.end,
      seasonLabel:       seasonInfo.label,
      seasonIndex:       seasonInfo.index,
      seasonYear:        yearNumber,
      sceneCount:        pairedCount,
      seasonStartMillis: startDate.millis()
    });


  // --------------------------------------------------------------------------
  // 8.11 Per-image quality for best observation of that season
  // --------------------------------------------------------------------------
  //
  // Observation-quality priority:
  // 1. lower cloud probability
  // 2. higher NDVI
  // 3. more recent acquisition only as a tie-breaker
  // --------------------------------------------------------------------------

  var preparedForBestObservation = paired.map(function (image) {

    var validSCL = buildValidMask(image);

    var cloudProbability = image
      .select('probability');

    var goodPixel = validSCL.and(
      cloudProbability.lt(cloudProbThreshold)
    );

    var ndvi = image
      .normalizedDifference(['B8', 'B4']);

    var cloudScore = ee.Image.constant(100)
      .subtract(cloudProbability)
      .multiply(10000);

    var ndviScore = ndvi
      .add(1)
      .multiply(100);

    var dateTieBreaker = ee.Image.constant(
      ee.Number(image.get('system:time_start'))
    ).divide(1e15);

    var observationQuality = cloudScore
      .add(ndviScore)
      .add(dateTieBreaker)
      .rename('quality');

    return image
      .select('SCL')
      .toFloat()
      .addBands(observationQuality)
      .updateMask(goodPixel)
      .copyProperties(
        image,
        [
          'system:index',
          'system:time_start'
        ]
      );
  });


  // --------------------------------------------------------------------------
  // 8.12 Best-quality SCL observation within the season
  // --------------------------------------------------------------------------

  var annualBestSCL = ee.Image(
    ee.Algorithms.If(

      preparedForBestObservation.size().gt(0),

      preparedForBestObservation
        .qualityMosaic('quality')
        .select('SCL'),

      emptySCL
    )
  )
    .rename('SCL')
    .set({
      seasonStart: seasonInfo.start,
      seasonEnd:   seasonInfo.end,
      seasonLabel: seasonInfo.label,
      seasonIndex: seasonInfo.index,
      sceneCount:  pairedCount
    });


  return {
    originalSeasonImage: originalSeasonImage,
    annualBestSCL: annualBestSCL
  };
}


// ============================================================================
// 9. Build all seven seasonal products
// ============================================================================

var builtProducts = seasons.map(function (seasonInfo) {
  return buildSeasonProducts(seasonInfo);
});


var originalSeasonImages = builtProducts.map(function (product) {
  return product.originalSeasonImage;
});


var annualBestSCLImages = builtProducts.map(function (product) {
  return product.annualBestSCL;
});


var seasonCollection = ee.ImageCollection.fromImages(
  originalSeasonImages
).sort('seasonIndex');


var annualBestSCLCollection = ee.ImageCollection.fromImages(
  annualBestSCLImages
).sort('seasonIndex');


// ============================================================================
// 10. Reconstruct the original hybrid
// ============================================================================
//
// Reconstruct the original hybrid with the same qualityMosaic rule.
// ============================================================================

var hybrid = seasonCollection.qualityMosaic('quality');


// ============================================================================
// 11. Define hybrid SCL=2 reference pixels
// ============================================================================

var hybridSCL = hybrid
  .select('SCL')
  .rename('hybrid_SCL');


var hybridSCL2Mask = hybridSCL
  .eq(2)
  .selfMask()
  .rename('hybrid_SCL2');


// ============================================================================
// 12. Seasonal SCL=2 status
// ============================================================================
//
// 1 = the best-quality observation in that season is SCL=2
// 0 = the best-quality observation is valid but is not SCL=2
// Masked = no valid observation is available for that season
// ============================================================================

var annualSCL2Collection = annualBestSCLCollection.map(function (image) {

  image = ee.Image(image);

  var scl = image.select('SCL');

  return scl
    .eq(2)
    .rename('SCL2')
    .toUint8()
    .updateMask(scl.mask())
    .updateMask(hybridSCL2Mask)
    .copyProperties(
      image,
      [
        'seasonStart',
        'seasonEnd',
        'seasonLabel',
        'seasonIndex',
        'sceneCount'
      ]
    );
});


// ============================================================================
// 13. Number of valid seasons
// ============================================================================

var validSeasonCount = annualSCL2Collection
  .count()
  .rename('valid_season_count')
  .updateMask(hybridSCL2Mask);


// ============================================================================
// 14. Number of seasons classified as SCL=2
// ============================================================================

var recurrenceCount = annualSCL2Collection
  .sum()
  .rename('SCL2_recurrence_count')
  .updateMask(hybridSCL2Mask);


// ============================================================================
// 15. Recurrence frequency among available seasons
// ============================================================================

var recurrenceFrequencyValid = recurrenceCount
  .toFloat()
  .divide(validSeasonCount)
  .rename('SCL2_frequency_valid_seasons')
  .updateMask(validSeasonCount.gt(0));


// ============================================================================
// 16. Complete seven-season coverage
// ============================================================================

var completeSevenSeasonMask = validSeasonCount
  .eq(7)
  .selfMask();


// ============================================================================
// 17. Recurrence frequency across seven seasons
// ============================================================================

var recurrenceCount7 = recurrenceCount
  .updateMask(completeSevenSeasonMask);


var recurrenceFrequency7 = recurrenceCount7
  .divide(7)
  .rename('SCL2_frequency_out_of_7');


// ============================================================================
// 18. Input diagnostics
// ============================================================================

print('======================================================');
print('1. INPUT DIAGNOSTICS');
print('======================================================');

print(
  'Number of seasonal composites:',
  seasonCollection.size()
);

print(
  'Number of best-quality seasonal images:',
  annualBestSCLCollection.size()
);

print(
  'Input scene counts by season:',
  annualBestSCLCollection.aggregate_array('sceneCount')
);

print(
  'Season labels:',
  annualBestSCLCollection.aggregate_array('seasonLabel')
);


// ============================================================================
// 19. Hybrid SCL=2 pixel count
// ============================================================================

var hybridSCL2PixelCount = hybridSCL2Mask
  .reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  })
  .getNumber('hybrid_SCL2');


print('======================================================');
print('2. ORIGINAL HYBRID SCL=2 PIXELS');
print('======================================================');

print(
  'Number of SCL=2 pixels in original hybrid:',
  hybridSCL2PixelCount
);


// ============================================================================
// 20. Valid-season coverage distribution
// ============================================================================

var validSeasonHistogram = validSeasonCount
  .reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  });


print('======================================================');
print('3. TEMPORAL DATA COMPLETENESS');
print('======================================================');

print(
  'Valid-season count histogram:',
  validSeasonHistogram
);


// ============================================================================
// 21. SCL=2 recurrence-count distribution
// ============================================================================
//
// Restrict this summary to pixels with valid observations in all seven seasons.
// ============================================================================

var recurrenceHistogram = recurrenceCount7
  .reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  });


print('======================================================');
print('4. SCL=2 RECURRENCE COUNT');
print('======================================================');

print(
  'Number of pixels by SCL=2 recurrence count, 0–7:',
  recurrenceHistogram
);


// ============================================================================
// 22. Recurrence summary statistics
// ============================================================================

var mainSummary = recurrenceCount7
  .addBands(recurrenceFrequency7)
  .reduceRegion({

    reducer: ee.Reducer.mean()

      .combine({
        reducer2: ee.Reducer.median(),
        sharedInputs: true
      })

      .combine({
        reducer2: ee.Reducer.stdDev(),
        sharedInputs: true
      })

      .combine({
        reducer2: ee.Reducer.minMax(),
        sharedInputs: true
      })

      .combine({
        reducer2: ee.Reducer.percentile([
          25,
          50,
          75,
          90,
          95
        ]),
        sharedInputs: true
      }),

    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  });


print('======================================================');
print('5. MAIN STABILITY STATISTICS');
print('======================================================');

print(
  'Recurrence-count and recurrence-frequency summary:',
  mainSummary
);


// ============================================================================
// 23. Recurrence thresholds
// ============================================================================
//
// The mean of each binary threshold band is the proportion meeting that criterion.
// Multiply each proportion by 100 to obtain a percentage.
// ============================================================================

var persistenceThresholds = ee.Image.cat([

  recurrenceCount7
    .gte(4)
    .rename('at_least_4_of_7'),

  recurrenceCount7
    .gte(5)
    .rename('at_least_5_of_7'),

  recurrenceCount7
    .gte(6)
    .rename('at_least_6_of_7'),

  recurrenceCount7
    .eq(7)
    .rename('all_7_seasons')

]);


var thresholdResults = persistenceThresholds
  .reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  });


print('======================================================');
print('6. PERSISTENCE THRESHOLDS');
print('======================================================');

print(
  'Persistence proportions from 0 to 1:',
  thresholdResults
);

print(
  'Interpretation:',
  'Multiply each proportion by 100 to obtain percentage.'
);


// ============================================================================
// 24. Season-specific recurrence
// ============================================================================
//
// For each season, calculate the proportion of hybrid SCL=2 pixels that
// are also SCL=2 in that season's best-quality observation.
// ============================================================================

var perSeasonResults = ee.FeatureCollection(
  annualSCL2Collection.map(function (image) {

    image = ee.Image(image);

    var result = image.reduceRegion({

      reducer: ee.Reducer.count()
        .combine({
          reducer2: ee.Reducer.sum(),
          sharedInputs: true
        }),

      geometry: studyArea,
      scale: analysisScale,
      crs: analysisCRS,
      maxPixels: 1e13,
      tileScale: 8
    });

    var validPixels = ee.Number(
      result.get('SCL2_count')
    );

    var scl2Pixels = ee.Number(
      result.get('SCL2_sum')
    );

    var recurrenceProportion = ee.Algorithms.If(
      validPixels.gt(0),
      scl2Pixels.divide(validPixels),
      null
    );

    return ee.Feature(null, {
      season: image.get('seasonLabel'),
      input_scene_count: image.get('sceneCount'),
      valid_hybrid_SCL2_pixels: validPixels,
      repeated_SCL2_pixels: scl2Pixels,
      recurrence_proportion: recurrenceProportion,
      recurrence_percentage: ee.Algorithms.If(
        validPixels.gt(0),
        scl2Pixels
          .divide(validPixels)
          .multiply(100),
        null
      )
    });
  })
);


print('======================================================');
print('7. RECURRENCE IN EACH SEASON');
print('======================================================');

print(
  'Per-season SCL=2 recurrence table:',
  perSeasonResults
);


// ============================================================================
// 25. Complete seven-season pixel count
// ============================================================================

var completePixelCount = recurrenceCount7
  .reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  })
  .getNumber('SCL2_recurrence_count');


var completePercentage = completePixelCount
  .divide(hybridSCL2PixelCount)
  .multiply(100);


print('======================================================');
print('8. FINAL DATA COMPLETENESS');
print('======================================================');

print(
  'Hybrid SCL=2 pixels with all seven valid seasons:',
  completePixelCount
);

print(
  'Percentage of hybrid SCL=2 pixels with all seven seasons:',
  completePercentage
);


// ============================================================================
// 26. Reporting summary
// ============================================================================

var meanRecurrenceCount = recurrenceCount7
  .reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  })
  .getNumber('SCL2_recurrence_count');


var medianRecurrenceCount = recurrenceCount7
  .reduceRegion({
    reducer: ee.Reducer.median(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  })
  .getNumber('SCL2_recurrence_count');


var meanRecurrenceFrequency = recurrenceFrequency7
  .reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  })
  .getNumber('SCL2_frequency_out_of_7');


var medianRecurrenceFrequency = recurrenceFrequency7
  .reduceRegion({
    reducer: ee.Reducer.median(),
    geometry: studyArea,
    scale: analysisScale,
    crs: analysisCRS,
    maxPixels: 1e13,
    tileScale: 8
  })
  .getNumber('SCL2_frequency_out_of_7');


print('======================================================');
print('9. FINAL REPORTING VALUES');
print('======================================================');

print(
  'Mean number of seasons with SCL=2, out of 7:',
  meanRecurrenceCount
);

print(
  'Median number of seasons with SCL=2, out of 7:',
  medianRecurrenceCount
);

print(
  'Mean recurrence frequency, 0–1:',
  meanRecurrenceFrequency
);

print(
  'Mean recurrence percentage:',
  meanRecurrenceFrequency.multiply(100)
);

print(
  'Median recurrence frequency, 0–1:',
  medianRecurrenceFrequency
);

print(
  'Median recurrence percentage:',
  medianRecurrenceFrequency.multiply(100)
);


// ============================================================================
// 27. Console interpretation guide
// ============================================================================

print('======================================================');
print('10. INTERPRETATION');
print('======================================================');

print(
  'Reference pixels:',
  'Pixels classified as SCL=2 in the original qualityMosaic hybrid.'
);

print(
  'Annual comparison:',
  'Best-quality individual Sentinel-2 observation selected independently in each season.'
);

print(
  'Primary result:',
  'Mean and median number of seasons in which hybrid SCL=2 pixels were also SCL=2.'
);

print(
  'At least 4 of 7:',
  'SCL=2 was repeated in the majority of the seven seasons.'
);

print(
  'At least 5 of 7:',
  'High temporal recurrence, frequency at least 0.714.'
);

print(
  'At least 6 of 7:',
  'Very high temporal recurrence, frequency at least 0.857.'
);

print(
  'All 7 seasons:',
  'Complete observed recurrence.'
);

print(
  'No raster or table was exported.',
  'All results are available in the Console.'
);
