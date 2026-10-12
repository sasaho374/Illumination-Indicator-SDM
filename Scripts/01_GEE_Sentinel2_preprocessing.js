/******************************************************************************
Sentinel-2 seasonal preprocessing and hybrid composite construction

Study area:
  Hezarmasjed Mountains, northeastern Iran
  UTM Zone 40N (EPSG:32640)

Temporal coverage:
  Seven November-April seasons from 2018-2019 through 2024-2025

Purpose:
  Construct the multiseason Sentinel-2 composite used as the remote-sensing
  basis of the study.

Main processing steps:
  1. Define seven November-April seasonal windows.
  2. Filter Sentinel-2 Level-2A and Cloud Probability collections.
  3. Exclude invalid, cloudy, snow/ice, and cloud-shadow SCL classes.
  4. Derive NDVI, GNDVI, NDMI, NDWI, and NBR.
  5. Build a median composite for each season.
  6. Calculate a seasonal quality score.
  7. Combine seasonal composites with qualityMosaic('quality').
  8. Add SRTM-derived elevation and terrain variables.
  9. Export the final hybrid and the seasonal products.

Notes:
  - SCL=3 represents cloud shadow and is excluded when removeShadows = true.
  - The SCL band is retained in the seasonal and final products.
  - Only explanatory comments have been edited for the public version;
    executable code is unchanged.
******************************************************************************/

/***************************************************************
Hybrid seasonal composite (explicit seasons, quality-prioritized)
Priority: fewer clouds/snow/shadow  → then higher NDVI
Tie-breaker: newer season wins (tiny epsilon)
***************************************************************/

// -----------------------------------------------------------------------------
// 1. Analysis settings
// -----------------------------------------------------------------------------
var seasons = [
  {start: '2018-11-06', end: '2019-04-04'},
  {start: '2019-11-06', end: '2020-04-04'},
  {start: '2020-11-06', end: '2021-04-04'},
  {start: '2021-11-06', end: '2022-04-04'},
  {start: '2022-11-06', end: '2023-04-04'},
  {start: '2023-11-06', end: '2024-04-04'},
  {start: '2024-11-06', end: '2025-04-04'} // سال جدید اضافه شد
];

var cloudProbThreshold = 30;    // Cloud Probability %
var imageCloudPercent = 70;     // CLOUDY_PIXEL_PERCENTAGE (metadata)
var removeShadows = true;      // true => remove SCL==3
var bufferMeters = 60000;

var exportCRS   = 'EPSG:32640';
var exportScale = 10;
var exportFolder = 'GEE_Exports';

// ---------------- STUDY AREA ----------------
var studyArea = ee.Geometry.Polygon([
  [59.171411, 37.157133],
  [59.687512, 36.950649],
  [59.544242, 36.689945],
  [58.973477, 36.913303],
  [59.171411, 37.157133]
]);
// Buffer the study area to ensure complete scene coverage near the boundary.
var expandedArea = studyArea.buffer(bufferMeters);

// -----------------------------------------------------------------------------
// 2. Output bands
// -----------------------------------------------------------------------------
var reflBands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];   // reflectance
var sclBand   = 'SCL';
var indexBands = ['NDVI','GNDVI','NDMI','NDWI','NBR'];
var outFloatBands = reflBands.concat([sclBand]).concat(indexBands);       // 16 float bands
var controlBands  = ['seasonCov','sourceSeason'];                         // Int16
var demBands      = ['DEM','Slope','Aspect','TPI','TRI','TWI'];

// -----------------------------------------------------------------------------
// 3. Helper objects and functions
// -----------------------------------------------------------------------------
var joinFilter = ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'});
var innerJoin  = ee.Join.inner();

// Empty templates used when a seasonal window contains no valid data.
var emptyFloat = (function(){
  var img = ee.Image.constant(0).rename([outFloatBands[0]]).toFloat();
  for (var i=1; i<outFloatBands.length; i++){
    img = img.addBands(ee.Image.constant(0).rename([outFloatBands[i]]).toFloat());
  }
  return img.reproject({crs: exportCRS, scale: exportScale})
            .updateMask(ee.Image.constant(0));
})();
var emptySeasonCov = ee.Image.constant(0).toInt16().rename('seasonCov')
                      .reproject({crs: exportCRS, scale: exportScale})
                      .updateMask(ee.Image.constant(0));
var emptySource = ee.Image.constant(0).toInt16().rename('sourceSeason')
                   .reproject({crs: exportCRS, scale: exportScale})
                   .updateMask(ee.Image.constant(0));

// Build the SCL-based validity mask.
function buildValidMask(img){
  var scl = img.select('SCL');
  // Exclude invalid data, cloud-related classes, and snow/ice.
  var good = scl.neq(1)
    .and(scl.neq(7))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));
  if (removeShadows) good = good.and(scl.neq(3)); // سایه اختیاری
  return good;
}

// Convert reflectance bands to float and derive spectral indices.
function toFloatWithIndices(img){
  var base = img.select(reflBands).toFloat();
  var sclf = img.select('SCL').toFloat();
  var ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI').toFloat();
  var gndvi= img.normalizedDifference(['B8','B3']).rename('GNDVI').toFloat();
  var ndmi = img.normalizedDifference(['B8','B11']).rename('NDMI').toFloat();
  var ndwi = img.normalizedDifference(['B3','B11']).rename('NDWI').toFloat();
  var nbr  = img.normalizedDifference(['B8','B12']).rename('NBR').toFloat();
  return base.addBands(sclf.rename(sclBand)).addBands([ndvi,gndvi,ndmi,ndwi,nbr]).select(outFloatBands);
}

// Derive elevation and terrain variables from SRTM.
function makeDemBands(){
  var dem = ee.Image('USGS/SRTMGL1_003')
    .resample('bilinear')
    .reproject({crs: exportCRS, scale: exportScale})
    .rename('DEM');
  var slope = ee.Terrain.slope(dem).rename('Slope');
  var aspect= ee.Terrain.aspect(dem).rename('Aspect');
  var kernel= ee.Kernel.circle({radius:3, units:'pixels'});
  var tri   = dem.reduceNeighborhood({reducer: ee.Reducer.stdDev(), kernel: kernel}).rename('TRI');
  var meanElev = dem.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kernel});
  var tpi   = dem.subtract(meanElev).rename('TPI');
  var radians = slope.multiply(Math.PI).divide(180);
  var tanSlope = radians.tan();
  var twi   = ee.Image.constant(100).log().divide(tanSlope.add(0.001)).rename('TWI');
  return dem.addBands([slope, aspect, tpi, tri, twi]);
}
var demImage = makeDemBands();

// -----------------------------------------------------------------------------
// 4. Seasonal composite construction
// -----------------------------------------------------------------------------
var seasonImages = seasons.map(function(s){
  s = ee.Dictionary(s);
  var sd = ee.Date(s.get('start'));
  var ed = ee.Date(s.get('end'));
  var yearNum = ee.Number(sd.get('year'));

  // S2_SR
  var s2_sr = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(expandedArea)
    .filterDate(sd, ed)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', imageCloudPercent));

  // Cloud Probability
  var cpCol = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(expandedArea)
    .filterDate(sd, ed);

  // Join surface reflectance and cloud-probability images by system:index.
  var joined = innerJoin.apply({
    primary: s2_sr,
    secondary: cpCol,
    condition: joinFilter
  });

  var paired = ee.ImageCollection(
    joined.map(function(f){
      f = ee.Feature(f);
      var p = ee.Image(f.get('primary'));
      var q = ee.Image(f.get('secondary'));
      var prob = ee.Image(ee.Algorithms.If(q, ee.Image(q).select('probability'),
                        ee.Image.constant(100).rename('probability')));
      return p.addBands(prob);
    })
  );

  // Apply the SCL validity mask when estimating median cloud probability.
  var forCloudMed = paired.map(function(im){
    var valid = buildValidMask(im);
    return im.updateMask(valid);
  });

  // If no valid paired image is available, use cloudMed = 100.
  var cloudMed = ee.Image(ee.Algorithms.If(
    forCloudMed.size().gt(0),
    forCloudMed.select('probability').median().rename('cloudMed'),
    ee.Image.constant(100).rename('cloudMed')
  ));

  // Apply the cloud-probability threshold and derive output bands.
  var processed = paired.map(function(im){
    var valid = buildValidMask(im);
    var good  = valid.and(im.select('probability').lt(cloudProbThreshold));
    var masked = im.updateMask(good);
    return toFloatWithIndices(masked);
  });

  var count = processed.size();

  // Create the seasonal median, or the empty template if no valid data are available.
  var seasonMedian = ee.Image(ee.Algorithms.If(count.gt(0), processed.median(), emptyFloat))
                      .select(outFloatBands);

  // Define the seasonal validity mask from the median B4 band.
  var validMask = seasonMedian.select('B4').mask();

  // seasonCov records the number of valid B4 observations.
  var cov = ee.Image(ee.Algorithms.If(count.gt(0),
    processed.select('B4').count()
      .reproject({crs: exportCRS, scale: exportScale})
      .toInt16().rename('seasonCov'),
    emptySeasonCov
  )).updateMask(validMask);

  // sourceSeason records the starting year of the seasonal window.
  var src = ee.Image.constant(yearNum).toInt16().rename('sourceSeason')
             .reproject({crs: exportCRS, scale: exportScale})
             .updateMask(validMask);

  // Seasonal quality prioritizes lower cloud probability, then higher NDVI; year breaks ties.
  var cloudScore = ee.Image.constant(100).subtract(cloudMed).rename('cloudScore'); // بزرگ‌تر بهتر
  var ndviScore  = seasonMedian.select('NDVI').add(1).multiply(100).rename('ndviScore');
  var eps        = ee.Image.constant(yearNum.subtract(2000)).divide(10000).rename('eps');

  var quality = cloudScore.multiply(10000).add(ndviScore).add(eps).rename('quality')
                 .updateMask(validMask); // خیلی مهم: جایی که فصل بی‌داده است، quality هم ماسک شود

  var out = seasonMedian.addBands([cov, src, quality])
    .set({
      seasonStart: s.get('start'),
      seasonEnd:   s.get('end'),
      sceneCount:  count,
      seasonStartMillis: sd.millis(),
      seasonYear: yearNum
    });

  // Report the percentage of valid pixels in the seasonal composite.
  var validPct = ee.Number(validMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: exportScale,
    maxPixels: 1e13
  }).get('B4')).multiply(100);
  print('Valid % [' + s.get('start') + ' - ' + s.get('end') + ']:', validPct);

  return out;
});

// Assemble the seven seasonal composites.
var seasonCollection = ee.ImageCollection(seasonImages);

print('Season count:', seasonCollection.size());
print('Scene counts per season:', seasonCollection.aggregate_array('sceneCount'));

// -----------------------------------------------------------------------------
// 5. Multiseason hybrid composite
// -----------------------------------------------------------------------------
var hybrid = seasonCollection.qualityMosaic('quality');

// Add elevation and terrain variables.
var final = hybrid.addBands(makeDemBands());

// Select the final band order.
var finalBandList = outFloatBands.concat(controlBands).concat(demBands);
var finalSelected = final.select(finalBandList);

// Clip and mask the final product to the study area.
var finalMasked = finalSelected.clip(studyArea)
  .updateMask(ee.Image.constant(1).clip(studyArea));

// -----------------------------------------------------------------------------
// 6. Processing diagnostics
// -----------------------------------------------------------------------------
var pctValidFinal = ee.Number(finalMasked.select('B4').mask().reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: studyArea,
  scale: exportScale,
  maxPixels: 1e13
}).get('B4')).multiply(100);
print('FINAL valid % (B4) in studyArea:', pctValidFinal);

var ssMinMax = finalMasked.select('sourceSeason').reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea, scale: exportScale, maxPixels: 1e13
});
print('SourceSeason min/max in hybrid:', ssMinMax);

var ssHist = finalMasked.select('sourceSeason').reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: studyArea, scale: exportScale, maxPixels: 1e13
});
print('SourceSeason histogram (pixel counts per year):', ssHist);

// Sum seasonal observation coverage across all seven seasons.
var coverageSum = seasonCollection.select('seasonCov').sum().rename('coverage_sum');

// -----------------------------------------------------------------------------
// 7. Map display
// -----------------------------------------------------------------------------
Map.centerObject(studyArea, 11);
Map.addLayer(finalMasked, {bands:['B4','B3','B2'], min:0, max:3000}, 'Final RGB (hybrid, masked)');
Map.addLayer(finalMasked.select('sourceSeason'), {
  min: Number(seasons[0].start.slice(0,4)),
  max: Number(seasons[seasons.length-1].start.slice(0,4)),
  palette: ['ffffcc','c2e699','78c679','31a354','006837']
}, 'SourceSeason (final)');
Map.addLayer(coverageSum.clip(studyArea), {min:0, max:seasons.length*50}, 'Coverage sum');

// -----------------------------------------------------------------------------
// 8. Exports
// -----------------------------------------------------------------------------
// Export the final hybrid composite with terrain and control bands.
Export.image.toDrive({
  image: finalMasked.toFloat(),
  description: 'hybrid_final_quality_Nov6_Apr4_with_DEM',
  folder: exportFolder,
  fileNamePrefix: 'hybrid_final_quality_Nov6_Apr4_with_DEM',
  region: studyArea,
  scale: exportScale,
  crs: exportCRS,
  maxPixels: 1e13
});

// Export each seasonal composite with control and quality bands.
// Use the seasons array directly to select each seasonal image.
var exportSeasonBands = outFloatBands.concat(controlBands).concat(['quality']);
for (var i = 0; i < seasons.length; i++) {
  var s = seasons[i];
  var img = seasonCollection
              .filter(ee.Filter.eq('seasonStart', s.start))
              .first();
  // Cast the selected seasonal object as an image before export.
  img = ee.Image(img);
  Export.image.toDrive({
    image: img.select(exportSeasonBands).toFloat().clip(studyArea),
    description: 'season_' + s.start.replace(/-/g,'') + '_to_' + s.end.replace(/-/g,'') + '_quality',
    folder: exportFolder,
    fileNamePrefix: 'season_' + s.start.replace(/-/g,'') + '_to_' + s.end.replace(/-/g,'') + '_quality',
    region: studyArea,
    scale: exportScale,
    crs: exportCRS,
    maxPixels: 1e13
  });
}
