# Oceanographic Water Mass Analysis

**[🔴 Live Demo](https://oceanographicwma.streamlit.app)**

This project performs quantitative water mass mixing analysis on oceanographic CTD (Conductivity, Temperature, Depth) data. Using the Optimum Multiparameter (OMP) method and TEOS-10 thermodynamic calculations, it decomposes observed water properties into contributions from end-member water masses.

The analysis visualizes water mass distributions through Temperature-Salinity diagrams with an innovative RGB color mixing approach, where color intensity directly represents proportional contributions from three core water masses (ENACW16, MW, NEADWL).

## The Analysis

The core analysis processes CTD profiles through the following workflow:

1. **Data Processing:** Reads NetCDF files containing pressure, temperature, and salinity measurements from multiple CTD casts.
2. **Interpolation:** Normalizes all profiles to a common depth grid (10-4000m) for cross-comparison.
3. **Density Calculation:** Computes in-situ density using the TEOS-10 Gibbs SeaWater (GSW) international standard equation of state.
4. **Water Mass Identification:** Plots observed T-S properties against characteristic end-member water masses (configurable - supports any number of water masses).
5. **OMP Mixing Analysis:** For three selected water masses, solves a linear system to calculate mixing fractions for points within the mixing triangle.
6. **RGB Visualization (3-Component):** As a special case visualization, maps three water mass mixing ratios to color channels for intuitive interpretation.
7. **Compositional Statistics:** Generates ternary diagrams showing average water mass contributions for three-component analysis.

**Note:** While this example focuses on three North Atlantic water masses (ENACW16, MW, NEADWL), the analysis supports **any number of water masses**. The RGB mixing visualization is specifically designed for three-component systems, but the standard T-S diagram can display unlimited water masses.

## Results

### Water Mass Mixing Analysis

The default configuration includes three primary North Atlantic water masses for demonstration:

| Water Mass | Type | Characteristic T-S |
| :--- | :--- | :---: |
| **ENACW16** | Eastern North Atlantic Central Water | 16.0°C, 36.15 PSU |
| **MW** | Mediterranean Water | 13.0°C, 36.50 PSU |
| **NEADWL** | North East Atlantic Deep Water Lower | 3.0°C, 34.95 PSU |

**Additional water masses available** (see `Pontos_MA.txt`):
- ENACW12, SAIW1, SAIW2 (Sub-Arctic Intermediate Water)
- SPMW7, SPMW8, IrSPMW (Sub-Polar Mode Waters)
- LSW (Labrador Sea Water)
- ISOW, DSOW (Overflow Waters)
- NEADWU (North East Atlantic Deep Water Upper)

Users can define custom water masses for other ocean basins (Pacific, Indian, Southern Ocean) or focus on different three-component systems for RGB analysis.

For each CTD observation point, the OMP method solves:
- m₁·T₁ + m₂·T₂ + m₃·T₃ = T_observed
- m₁·S₁ + m₂·S₂ + m₃·S₃ = S_observed  
- m₁ + m₂ + m₃ = 1

Where m₁, m₂, m₃ are mixing fractions (0 ≤ mᵢ ≤ 1) representing the proportional contribution of each water mass.

### Output Visualizations

The analysis generates a comprehensive suite of oceanographic plots:

#### 1. Geographic Context
![CTD Profile Locations](images/Loc%20Perfis%20CTD.png)

#### 2. Vertical Structure
![Temperature Section](images/Vert%20Section%20Temp.png)

#### 3. RGB Water Mass Mixing (Innovation)
![RGB T-S Diagram](images/Diagram%20TS%20Isocpicnas%20Mistura%20RGB.png)

The RGB visualization is a novel approach where each data point's color directly represents its mixing composition. This transforms quantitative analysis into an intuitive, colorimetric interpretation accessible to non-specialists.

#### 4. Compositional Statistics
![Ternary Diagram](images/Ternary%20Diagram.png)

## Project Structure

```
Oceanographic-Water-Mass-Analysis-in-Python/
├── README.md
├── requirements.txt
├── app.py                                    # Streamlit web application
├── oceanography.py                           # Core analysis module
├── Pontos_MA.txt                            # Water mass definitions
├── Oceanography Water Mass Evaluation.ipynb  # Jupyter analysis notebook
├── DEPLOYMENT.md                            # Deployment guide
├── data/                                     # Example CTD data
│   ├── example_profile_001.nc
│   ├── example_profile_002.nc
│   └── README_DATA.md                       # Data documentation
└── images/                                   # Example outputs
    ├── Loc Perfis CTD.png
    ├── Vert Section Temp.png
    ├── Diagram TS Isocpicnas Mistura RGB.png
    └── Ternary Diagram.png
```

## Technology Used

* **Python**
* **NumPy:** For numerical calculations and array operations
* **GSW (TEOS-10):** For accurate seawater thermodynamic calculations
* **Matplotlib:** For publication-quality plotting
* **Cartopy:** For geographic mapping and projections
* **netCDF4:** For scientific data file handling
* **Streamlit:** For interactive web application deployment

## Business Applications

This analysis has practical applications in:

- **Offshore Energy:** Subsurface current prediction for underwater infrastructure routing
- **Fisheries Management:** Water mass boundary identification for habitat mapping
- **Climate Monitoring:** Ocean circulation quantification and change detection
- **Environmental Consulting:** Baseline oceanographic studies for impact assessments

## How to Use

### Option 1: Web Application (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Miauneiro/Oceanographic-Water-Mass-Analysis-in-Python.git
   cd Oceanographic-Water-Mass-Analysis-in-Python
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application:**
   ```bash
   streamlit run app.py
   ```
   
   Open your browser to `http://localhost:8501`. Click **"📊 Load Example Analysis"** for instant results with included sample data, or upload your own NetCDF files.

**💡 Tip:** Select your application focus (Offshore Energy, Fisheries, Climate Services, etc.) to see business-relevant interpretations tailored to your field.

### Option 2: Jupyter Notebook

1. **Install dependencies** (same as above)

2. **Launch Jupyter:**
   ```bash
   jupyter notebook "Oceanography Water Mass Evaluation.ipynb"
   ```

3. **Update file paths** in the notebook to point to your NetCDF data files

### Option 3: Python Module

```python
from oceanography import read_profiles, calculate_density, plot_ts_diagram
import numpy as np

# Define depth grid
Znew = np.linspace(10, 4000, 400)

# Process CTD files
la, lo, Ti, Te, Se = read_profiles(ncf_files, Znew)
Rho = calculate_density(Te, Se, Znew, lo, la)

# Generate T-S diagram with mixing analysis
fig, mixing_percentages = plot_ts_diagram(T, S, P, water_masses)
```

## Data Format

NetCDF files should contain the following variables:
- `lat`: Latitude (degrees_north)
- `lon`: Longitude (degrees_east)  
- `time`: Time of measurement
- `Pressure`: Seawater pressure (dbar)
- `Temperature`: In situ temperature (°C)
- `Salinity`: Practical Salinity (PSU)

**Example data source:** [World Ocean Database](https://www.ncei.noaa.gov/products/world-ocean-database)

### Sample Data Included

This repository includes example CTD profiles from the North Atlantic for testing and demonstration:

```
data/
├── example_profile_001.nc    # Single CTD cast
├── example_profile_002.nc    # Additional profiles
└── README_DATA.md            # Data source and metadata
```

These files can be used to:
- Test the application functionality
- Learn the expected NetCDF format
- Explore water mass mixing in the North Atlantic
- Verify installation and dependencies

## Scientific Background

The Optimum Multiparameter (OMP) analysis is a standard technique in physical oceanography for water mass characterization. This implementation follows methods used in:
- WOCE (World Ocean Circulation Experiment) data analysis
- CLIVAR (Climate Variability and Predictability) studies
- ARGO float data interpretation
- Ocean reanalysis validation

The TEOS-10 equation of state replaces the older EOS-80 standard and provides more accurate thermodynamic calculations across the full range of ocean conditions.

## Contributing

Contributions welcome for:
- Additional water mass end-member databases for other ocean basins
- Extended OMP formulations with oxygen or nutrient tracers
- Performance optimizations for large datasets
- Additional visualization options

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- World Ocean Database for oceanographic data
- TEOS-10 GSW library for accurate seawater calculations
- Streamlit for the web application framework
- Physical oceanography community for water mass classification standards
