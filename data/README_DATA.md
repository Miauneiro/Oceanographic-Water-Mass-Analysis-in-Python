# Example CTD Data

## Overview

This directory contains three sample CTD (Conductivity, Temperature, Depth) profiles from the North Atlantic Ocean for testing and demonstration purposes. These profiles are from a June 2004 oceanographic transect offshore Portugal.

## Files

| File | Location | Date | Depth Range | Description |
|------|----------|------|-------------|-------------|
| `example_profile_001.nc` | 44.68°N, 18.21°W | 2004-06-27 | 0-4919 m | Deep profile reaching NEADW |
| `example_profile_002.nc` | 44.08°N, 17.42°W | 2004-06-28 | 0-3819 m | Mid-depth profile |
| `example_profile_003.nc` | 43.48°N, 16.64°W | 2004-06-28 | 0-4236 m | Southern transect station |

## Data Source

**Source:** [World Ocean Database (WOD)](https://www.ncei.noaa.gov/products/world-ocean-database)

**Cruise:** FR014872 (French research vessel)

**Region:** North Atlantic, offshore Portugal (~40-45°N, 15-20°W)

**Instrument:** CTD (Conductivity-Temperature-Depth profiler)

**Quality Control:** WOD standard QC procedures applied

**License:** Public domain - freely available for scientific, educational, and commercial use

## Geographic Context

These three profiles form a north-south transect in the eastern North Atlantic, capturing the vertical structure of major water masses in this region:

```
44.68°N ● example_profile_001.nc (4919 m depth)
        |
        | ~70 km
        |
44.08°N ● example_profile_002.nc (3819 m depth)
        |
        | ~70 km  
        |
43.48°N ● example_profile_003.nc (4236 m depth)
```

## NetCDF Structure

Each file contains the following variables:

### Dimensions
- `z`: Depth/pressure levels (3820-4920 points depending on profile)

### Coordinates
- `lat`: Latitude (decimal degrees North)
- `lon`: Longitude (decimal degrees East/West)
- `time`: Time of measurement (days since 1770-01-01, WOD convention)

### Data Variables
- `Pressure`: Seawater pressure (dbar, approximately equals depth in meters)
- `Temperature`: In situ temperature (degrees Celsius)
- `Salinity`: Practical Salinity (PSU)

### Quality Control Variables
- Various QC flags indicating data quality and processing steps

### Global Attributes
- Cruise identifier, country, station number, and metadata

## Expected Water Masses

Based on the location (eastern North Atlantic) and depth range, these profiles contain:

| Water Mass | Abbreviation | Typical Depth | T-S Characteristics |
|------------|--------------|---------------|---------------------|
| **Eastern North Atlantic Central Water** | ENACW | 0-1000 m | 12-19°C, 35.5-36.0 PSU |
| **Mediterranean Water** | MW | 800-1500 m | 11-13°C, 36.0-36.5 PSU |
| **Labrador Sea Water** | LSW | 1500-2500 m | 3-4°C, ~35.0 PSU |
| **North East Atlantic Deep Water** | NEADW | >2500 m | 2-3°C, ~35.0 PSU |

The profiles show classic eastern North Atlantic stratification with warm surface waters, a Mediterranean Water tongue at mid-depth, and cold deep waters originating from the Labrador Sea and Nordic Seas.

## Temperature and Salinity Ranges

| Profile | Temperature (°C) | Salinity (PSU) | Max Depth (m) |
|---------|------------------|----------------|---------------|
| 001 | 2.57 - 17.93 | 34.906 - 35.926 | 4919 |
| 002 | 2.58 - 18.60 | 34.913 - 35.935 | 3819 |
| 003 | 2.54 - 18.68 | 34.907 - 35.942 | 4236 |

## Usage Examples

### Quick Start - Python

```python
from netCDF4 import Dataset
import numpy as np

# Read a single profile
nc = Dataset('data/example_profile_001.nc', 'r')
pressure = nc.variables['Pressure'][:]
temperature = nc.variables['Temperature'][:]
salinity = nc.variables['Salinity'][:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
nc.close()

print(f"Profile location: {lat:.2f}°N, {lon:.2f}°W")
print(f"Depth range: {pressure.min():.0f} - {pressure.max():.0f} m")
print(f"Temperature range: {temperature.min():.2f} - {temperature.max():.2f}°C")
```

### With Streamlit App

1. Launch the app:
   ```bash
   streamlit run app.py
   ```

2. In the sidebar:
   - Click "Upload NetCDF Files"
   - Navigate to `data/` directory
   - Select all three `example_profile_*.nc` files
   - Click "Open"

3. The app will automatically:
   - Read and interpolate all three profiles
   - Display geographic map showing transect
   - Generate T-S diagrams with water mass mixing
   - Show RGB mixing visualization
   - Create vertical cross-sections

### With Jupyter Notebook

1. Open `Oceanography Water Mass Evaluation.ipynb`

2. Update the file pattern in the data loading cell:
   ```python
   pattern = 'data/example_profile_*.nc'
   ncf = sorted(glob.glob(pattern))
   ```

3. Run all cells to see complete analysis

### With Python Module

```python
from oceanography import read_profiles, calculate_density, plot_ts_diagram
import numpy as np
import glob

# Load all example profiles
files = sorted(glob.glob('data/example_profile_*.nc'))
Znew = np.linspace(10, 4000, 400)

# Process profiles
la, lo, Ti, Te, Se = read_profiles(files, Znew)
Rho = calculate_density(Te, Se, Znew, lo, la)

# Define water masses
water_masses = {
    'ENACW16': {'temp': [16.0], 'sal': [36.15]},
    'MW': {'temp': [13.0], 'sal': [36.50]},
    'NEADWL': {'temp': [3.0], 'sal': [34.95]}
}

# Generate T-S diagram with first profile
from netCDF4 import Dataset
nc = Dataset(files[0], 'r')
T = nc.variables['Temperature'][:]
S = nc.variables['Salinity'][:]
P = nc.variables['Pressure'][:]
nc.close()

fig, mixing = plot_ts_diagram(T, S, P, water_masses)
```

## Scientific Context

### Transect Location
This transect crosses the eastern North Atlantic subtropical gyre, capturing water masses that:
- Originate from Mediterranean outflow (MW)
- Mix with North Atlantic Central Water (ENACW)
- Overlay deep waters from the Nordic Seas and Labrador Sea

### Expected Results
When analyzed with the OMP method:
- **Surface layers (0-500m):** Dominated by ENACW (~70-90%)
- **Mid-depth (800-1500m):** Mediterranean Water influence (~30-50%)
- **Deep layers (>2500m):** NEADW dominance (~60-80%)

The RGB mixing diagram should show:
- Reddish colors (ENACW) in upper ocean
- Greenish colors (MW) at intermediate depths
- Bluish colors (NEADWL) in deep waters

## Data Quality

All profiles have been quality controlled by NOAA/NCEI for:
- ✓ Spike detection
- ✓ Gradient checks  
- ✓ Range validation
- ✓ Density inversions
- ✓ Duplicate elimination

Data quality is generally excellent with <1% flagged or missing values.

## Citation

If using this data in publications, please cite:

```
Boyer, T.P., et al., 2018. World Ocean Database 2018. 
A.V. Mishonov, Technical Editor, NOAA Atlas NESDIS 87.
```

## License and Usage

**License:** Public Domain (U.S. Government Work)

This data is freely available for:
- ✓ Scientific research
- ✓ Educational purposes
- ✓ Commercial applications
- ✓ Portfolio demonstrations
- ✓ Software testing

No attribution required, but citation appreciated for publications.

## Additional Resources

- [WOD Homepage](https://www.ncei.noaa.gov/products/world-ocean-database)
- [WOD User Manual](https://www.ncei.noaa.gov/data/oceans/wod/DOCUMENTATION/)
- [TEOS-10 Website](http://www.teos-10.org/)
- [GSW Python Documentation](https://teos-10.github.io/GSW-Python/)

## File Sizes

| File | Size |
|------|------|
| example_profile_001.nc | 231 KB |
| example_profile_002.nc | 182 KB |
| example_profile_003.nc | 200 KB |
| **Total** | **613 KB** |

Small file sizes ensure fast download and processing, suitable for testing and demonstrations.

## Contact

For questions about this dataset:
- **Repository:** [Oceanographic Water Mass Analysis](https://github.com/Miauneiro/Oceanographic-Water-Mass-Analysis-in-Python)
- **Data Source:** NOAA National Centers for Environmental Information (NCEI)
- **WOD Contact:** https://www.ncei.noaa.gov/products/world-ocean-database

---

**Last Updated:** November 2024  
**Data Collection Date:** June 2004  
**Region:** Eastern North Atlantic (40-45°N, 15-20°W)
