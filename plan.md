# Agri Orthomosaic Engine

## Vision

Build a specialized orthomosaic generation engine optimized exclusively for agricultural drone imagery.

Unlike generic photogrammetry platforms such as Metashape, ODM, Pix4D, or RealityCapture, this engine will focus only on agricultural workflows and make assumptions that significantly reduce processing complexity.

The goal is not to create a general-purpose photogrammetry application but rather a high-performance backend service capable of producing accurate RGB and multispectral orthomosaics for downstream agricultural analytics.

---

# Long-Term Goals

The engine should:

* Generate RGB orthomosaics
* Generate multispectral orthomosaics
* Use a single geometry reconstruction pipeline
* Guarantee RGB ↔ multispectral pixel alignment
* Support vegetation index generation
* Support yield prediction workflows
* Support future cloud deployment
* Support future distributed processing

---

# Scope

Supported:

* Agricultural fields
* RGB imagery
* Multispectral imagery
* Grid flight missions
* Nadir imagery
* RTK GPS workflows
* Reflectance calibrated multispectral sensors

Not Supported:

* Buildings
* Mines
* Forest reconstruction
* Cultural heritage
* 3D mesh generation
* Textured model generation
* Volumetric reconstruction

---

# Core Philosophy

Instead of supporting every photogrammetry use case, optimize aggressively for:

* Fixed drone platforms
* Fixed sensors
* Known overlap
* Known altitude
* Known mission patterns
* Mostly flat terrain

This allows significant performance optimizations compared to generic photogrammetry software.

---


# Expected Processing Pipeline

Mission Upload
↓
Image Validation
↓
Image Grouping
↓
Feature Detection
↓
Feature Matching
↓
Camera Alignment
↓
Bundle Adjustment
↓
Depth Map Generation
↓
DEM Generation
↓
RGB Orthomosaic Generation
↓
Multispectral Orthomosaic Generation
↓
Output Export

---

# RGB + Multispectral Strategy

Only one geometry should be computed.

Workflow:

RGB Images
↓
Alignment
↓
Depth Maps
↓
DEM

Reuse DEM for:

RGB Orthomosaic
and
Multispectral Orthomosaic

Benefits:

* Faster processing
* Perfect pixel alignment
* More accurate NDVI/NDRE
* Lower compute cost

---

# Repository Structure

src/

ingestion/
alignment/
matching/
geometry/
depth_maps/
dem/
orthomosaic/
export/
utils/

tests/
benchmarks/
docs/

---

# Development Roadmap

## Phase 1

Research

Goals:

* Understand photogrammetry pipeline
* Study ODM architecture
* Study OpenMVG
* Study OpenMVS
* Study GDAL

Deliverables:

* Architecture document
* Performance targets

---

## Phase 2

Pipeline Prototype

Goals:

* Input mission images
* Group RGB and multispectral captures
* Generate camera alignment
* Generate DEM
* Generate orthomosaic

Deliverables:

* First working pipeline

---

## Phase 3

Agricultural Optimization

Goals:

* GPS constrained image matching
* Reduced search space
* Fixed camera calibration profiles
* RGB-only geometry reconstruction
* Tile-based processing

Deliverables:

* Faster than baseline implementation

---

## Phase 4

Integration

Goals:

* Integrate with Agri Platform
* API communication
* Mission processing endpoint

Deliverables:

* End-to-end workflow

Upload
↓
Orthomosaic
↓
NDVI
↓
NDRE
↓
Yield Prediction

---

# Candidate Technologies

Feature Detection:

* OpenCV

Feature Matching:

* OpenCV FLANN
* OpenMVG

Bundle Adjustment:

* Ceres Solver

Depth Maps:

* OpenMVS

Raster Processing:

* GDAL

Geospatial Operations:

* GDAL
* Rasterio

Machine Learning:

* PyTorch

Backend:

* FastAPI (future)

---

# Performance Goals

Target 1:
Generate RGB + Multispectral Orthomosaics from a single geometry pipeline.

Target 2:
Reduce processing time by leveraging agricultural assumptions.

Target 3:
Support future GPU acceleration.

Target 4:
Support cloud and on-prem deployment.

Target 5:
Serve as the core geospatial engine for the Agri Platform.

---

# Success Criteria

Input:

RGB.jpg
Blue.tif
Green.tif
Red.tif
NIR.tif

Output:

rgb_orthomosaic.tif
multispectral_orthomosaic.tif
dem.tif

All outputs must be georeferenced and spatially aligned.

The engine should become the foundation for all downstream analytics including:

* NDVI
* NDRE
* Vegetation health assessment
* Yield prediction
* Future disease detection models


Your Engine (target)

Feature Extraction : COLMAP SiftGPU       → matches Metashape quality
Camera Mode        : SINGLE               → matches Metashape behavior  
GPS Filtering      : your custom code     → faster than both Metashape and ODM
Matching           : COLMAP GPU FLANN     → matches Metashape quality
Verification       : COLMAP RANSAC        → matches Metashape quality


ODM (CPU SIFT)        : ~45-60 minutes
COLMAP (GPU SiftGPU)  : ~4-6 minutes
Metashape (GPU)       : ~3-5 minutes



  Feature Extraction
Metashape
  Algorithm  : proprietary SIFT variant
  GPU        : yes, CUDA
  Camera     : shared calibration across mission (same as COLMAP SINGLE mode)
  Descriptor : 128-dim, RootSIFT normalized
  Extra      : adaptive feature count based on image content
  Speed      : fastest, highly optimized proprietary code

COLMAP
  Algorithm  : SiftGPU + RootSIFT + domain size pooling
  GPU        : yes, CUDA
  Camera     : SINGLE mode = shared calibration (matches Metashape behavior)
  Descriptor : 128-dim, RootSIFT normalized
  Extra      : domain size pooling for repetitive textures
  Speed      : very fast, 10-20x over CPU

ODM
  Algorithm  : VLFeat SIFT (CPU only)
  GPU        : no
  Camera     : estimates per image by default
  Descriptor : 128-dim standard SIFT
  Extra      : nothing special
  Speed      : slowest of the three

Feature Matching
Metashape
  Method     : proprietary ANN + guided matching
  GPU        : yes
  Pairs      : adaptive, image similarity based
  Verification: RANSAC
  Speed      : fastest

COLMAP
  Method     : FLANN on GPU + guided matching + RANSAC
  GPU        : yes
  Pairs      : exhaustive / sequential / vocab tree / custom
  Verification: RANSAC built in
  Speed      : very fast

ODM
  Method     : FLANN CPU
  GPU        : no
  Pairs      : bow (bag of words) based filtering
  Verification: RANSAC
  Speed      : slow

Quality difference in plain terms
Metashape
  → Best quality, proprietary optimizations, paid software
  → Gold standard for professionals

COLMAP
  → 95% of Metashape quality
  → Free, open source
  → What academic researchers use to benchmark against Metashape
  → Difference is negligible for agricultural flat fields

ODM
  → 80-85% of Metashape quality
  → CPU only feature extraction
  → Much slower
  → Good enough for basic orthomosaics, not optimized

What this means for your engine
Your Engine (target)

Feature Extraction : COLMAP SiftGPU       → matches Metashape quality
Camera Mode        : SINGLE               → matches Metashape behavior  
GPS Filtering      : your custom code     → faster than both Metashape and ODM
Matching           : COLMAP GPU FLANN     → matches Metashape quality
Verification       : COLMAP RANSAC        → matches Metashape quality







STAGE BY STAGE  -  OPTIONS AND BEST CHOICE

1. Ingestion
What it does : parse filenames, group by capture ID, read GPS from EXIF, validate completeness
Input        : raw image folders
Output       : structured capture list with GPS
Options      : custom code only  (no library needed)
Our choice   : custom  (already built)

2. Feature Extraction
What it does : find distinctive keypoints in each image and compute a descriptor for each
Input        : RGB images
Output       : keypoints + 128-dim descriptors per image
Options      : SIFT (OpenCV)  |  SIFT (VLFeat)  |  COLMAP SiftGPU  |  SuperPoint  |  ALIKED  |  DISK  |  DeDoDe
Our choice   : ALIKED  (GPU, learned, best on repetitive crop textures, memory controlled)

3. Feature Matching
What it does : match descriptors between overlapping image pairs, filter false matches
Input        : descriptors + GPS-filtered image pairs
Output       : verified tie points between pairs
Options      : FLANN + RANSAC  |  SuperGlue  |  LightGlue  |  LoFTR
Our choice   : LightGlue  (GPU, transformer-based, built-in geometry verification, fast on easy pairs)

4. SfM Mapping
What it does : compute exact 3D position and orientation of every camera from matches
Input        : verified matches (.db)
Output       : camera poses + sparse 3D point cloud
Options      : COLMAP  |  OpenMVG  |  Theia  |  glomap
Our choice   : COLMAP incremental mapper  (most robust, best documented, GPS prior support)

5. Georeferencing
What it does : align the reconstructed model to real-world GPS coordinates
Input        : sparse model + GPS from EXIF
Output       : model in UTM / WGS84 coordinate system
Options      : COLMAP model_aligner  |  custom GPS alignment  |  GCP-based (ground control points)
Our choice   : COLMAP model_aligner with RTK GPS priors  (RTK GPS is accurate enough, no GCPs needed)

6. Depth Maps
What it does : estimate per-pixel depth for each image using neighboring views
Input        : camera poses + RGB images
Output       : per-image depth map
Options      : OpenMVS  |  COLMAP stereo  |  custom SGM
Our choice   : OpenMVS  (faster than COLMAP stereo, GPU accelerated, well maintained)

7. DSM Generation
What it does : fuse all depth maps into a single elevation grid over the field
Input        : depth maps
Output       : dsm.tif
Options      : OpenMVS fusion  |  PDAL  |  custom rasterization with numpy + scipy
Our choice   : OpenMVS fusion for dense cloud then GDAL rasterize to GeoTIFF

8. Orthorectification
What it does : project each image onto the DSM ground plane to remove perspective distortion
Input        : DSM + images + camera poses
Output       : flat georeferenced image tiles
Options      : GDAL warp  |  custom ray casting  |  OpenDroneMap ortho module
Our choice   : GDAL warp with RPC model  (standard, fast, battle tested)

9. Mosaicking
What it does : stitch all orthorectified tiles into one seamless orthomosaic
Input        : orthorectified RGB tiles + all band tiles (reuse same geometry)
Output       : rgb_orthomosaic.tif  +  multispectral_orthomosaic.tif
Options      : GDAL merge  |  custom blending  |  OpenCV seamless clone
Our choice   : GDAL merge with feathering blending  (georeferenced output, standard format)

10. Indices
What it does : compute vegetation indices from multispectral bands
Input        : multispectral_orthomosaic.tif
Output       : ndvi.tif  ndre.tif
Options      : numpy  |  rasterio  |  custom
Formula NDVI : (NIR - RED) / (NIR + RED)
Formula NDRE : (NIR - REG) / (NIR + REG)
Our choice   : numpy + rasterio  (trivial, no library needed beyond these)


KEY OPTIMIZATIONS IN OUR ENGINE

1.  GPS neighbor filtering         900x900 pairs reduced to ~7200 pairs before matching
2.  Single camera mode             fixed drone = shared calibration = skip per-image estimation
3.  RGB-only geometry              compute DSM once, reuse same geometry for all 5 bands
4.  No mesh generation             skip dense point cloud export entirely
5.  Flat terrain assumption        simpler DSM, no complex surface reconstruction needed
6.  ALIKED not SIFT               handles repetitive crop texture better than classical SIFT
