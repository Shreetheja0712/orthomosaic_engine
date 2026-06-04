# Agri Orthomosaic Engine

A specialized orthomosaic generation engine optimized for agricultural drone imagery.

## Overview

This engine is designed to focus exclusively on agricultural workflows, making assumptions that reduce processing complexity while maintaining high accuracy for RGB and multispectral orthomosaics.

## Features

- **Ingestion**: Automated grouping and validation of RGB and multispectral imagery.
- **EXIF Processing**: Extraction of GPS and sensor metadata.
- **Multispectral Alignment**: (In development) Alignment of multiple spectral bands.
- **Agriculture-focused**: Optimized for nadir flight missions and RTK GPS workflows.

## Installation

```bash
pip install -r requirements.txt
```

PyCOLMAP is installed on supported Python versions through `requirements.txt`.
On Python 3.14, use the COLMAP CLI fallback by installing the `colmap` binary
separately and making sure it is available on `PATH`.

## Testing

```bash
python -m pytest
```

## Project Structure

- `src/ingestion/`: Handles image loading, EXIF reading, and capture grouping.
- `src/features/`: Handles RGB feature extraction and GPS-filtered matching.
- `src/processing/`: (Planned) Core geometry and reconstruction pipeline.
- `tests/`: Unit and integration tests.
