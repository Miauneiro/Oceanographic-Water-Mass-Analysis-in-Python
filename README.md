# Oceanographic Water Mass Analysis

🌊 **Interactive web application for analyzing oceanographic CTD data and water mass mixing**

This project provides both a Jupyter notebook and an interactive Streamlit web application for analyzing oceanographic CTD (Conductivity, Temperature, Depth) profiles from the World Ocean Database. It processes raw `.nc` (NetCDF) files to calculate derived properties, visualize data on maps and vertical sections, and perform detailed water mass mixing analysis using T-S diagrams.

---

## 🚀 Quick Start - Web Application

The easiest way to use this project is through the interactive Streamlit web application:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and upload your NetCDF files!

---

## Key Features

* **Interactive Web Interface:** Easy-to-use Streamlit application with drag-and-drop file uploads
* **Data Processing:** Reads and interpolates data from multiple NetCDF files
* **Scientific Calculations:** Uses the `gsw` (TEOS-10 Gibbs SeaWater) library to accurately calculate density (`rho`) and accumulated distance
* **Water Mass Mixing:** Performs Optimum Multipoint (OMP) analysis to calculate the mixing percentages of three core water masses (ENACW16, MW, and NEADWL)
* **Advanced Visualization:** Generates a suite of professional plots to analyze the data
* **RGB Mixing Visualization:** Unique color-coded visualization showing proportional water mass contributions

---

## Visualizations

This script automatically generates several high-quality plots. Here are examples of the key outputs:

### 1. Profile Map
Shows the geographic location of the CTD casts and the ship's track using `cartopy`.

![Profile Map](images/Loc%20Perfis%20CTD.png)

### 2. Vertical Sections
Visualizes the changes in Temperature, Salinity, and Density with depth along the ship's track.

![Temperature Section](images/Vert%20Section%20Temp.png)

### 3. T-S Diagram (with RGB Mixing)
This is the core analysis of the project. This plot shows the Temperature-Salinity properties of the water. Each data point is colored based on its mixing ratio between the three primary water masses (Red = ENACW16, Green = MW, Blue = NEADWL).

![RGB T-S Diagram](images/Diagram%20TS%20Isocpicnas%20Mistura%20RGB.png)

### 4. Ternary Diagram
This plot shows the *average* contribution of the three water masses for all data points that fall within the mixing triangle.

![Ternary Diagram](images/Ternary%20Diagram.png)

---

## Technologies Used

* **Python 3**
* **Streamlit:** Interactive web application framework
* **NumPy:** For all numerical calculations
* **Matplotlib:** For all plotting
* **netCDF4:** For reading scientific data files
* **gsw (TEOS-10):** For accurate oceanographic thermodynamic calculations
* **Cartopy:** For creating high-quality maps

---

## 📋 How to Use

### Option 1: Web Application (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Miauneiro/Oceanographic-Water-Mass-Analysis-in-Python.git
   cd Oceanographic-Water-Mass-Analysis-in-Python
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Upload your data:**
   * Open your browser to `http://localhost:8501`
   * Upload your `.nc` (NetCDF) CTD files using the sidebar
   * Optionally customize water mass definitions
   * View interactive visualizations and analysis results

### Option 2: Jupyter Notebook

1. **Clone the repository and install dependencies** (same as above)

2. **Prepare your data:**
   * Place your `.nc` data files in the project folder
   * Create a `Pontos_MA.txt` file with water mass definitions (optional)

3. **Open and run the notebook:**
   ```bash
   jupyter notebook "Oceanography Water Mass Evaluation.ipynb"
   ```

4. **Update file paths:**
   * Modify the `pattern` variable to match your NetCDF file names
   * Run all cells to generate visualizations

### Option 3: Python Module

You can also import and use the functions directly:

```python
from oceanography import ler_perfis, calcular_densidade, plotar_TS
import numpy as np

# Define depth levels
Znew = np.linspace(10, 4000, 400)

# Read profiles
la, lo, Ti, Te, Se = ler_perfis(ncf_files, Znew)

# Calculate density
Rho = calcular_densidade(Te, Se, Znew, lo, la)

# Create visualizations
# ... (see oceanography.py for available functions)
```

---

## 📁 Project Structure

```
Oceanographic-Water-Mass-Analysis-in-Python/
├── app.py                                    # Streamlit web application
├── oceanography.py                           # Core analysis module
├── Oceanography Water Mass Evaluation.ipynb  # Jupyter notebook version
├── Pontos_MA.txt                            # Water mass definitions
├── requirements.txt                          # Python dependencies
├── README.md                                # This file
└── images/                                  # Example visualization outputs
    ├── Loc Perfis CTD.png
    ├── Vert Section Temp.png
    ├── Diagram TS Isocpicnas Mistura RGB.png
    └── Ternary Diagram.png
```

---

## 💡 Understanding the Analysis

### Water Mass Mixing (OMP Method)

The Optimum Multipoint (OMP) analysis solves a system of linear equations to determine the mixing proportions of different water masses:

- **m₁·T₁ + m₂·T₂ + m₃·T₃ = T_observed**
- **m₁·S₁ + m₂·S₂ + m₃·S₃ = S_observed**
- **m₁ + m₂ + m₃ = 1**

Where:
- m₁, m₂, m₃ are the mixing fractions (percentages)
- T₁, S₁ etc. are the characteristic temperature and salinity of each water mass
- T_observed, S_observed are the measured values at each point

### RGB Color Visualization

The unique RGB visualization maps the three primary water masses to color channels:
- **Red** = ENACW16 (Eastern North Atlantic Central Water)
- **Green** = MW (Mediterranean Water)
- **Blue** = NEADWL (North East Atlantic Deep Water Lower)

Points are colored proportionally to their mixing fractions, making it easy to visually identify water mass dominance and transitions.

---

## 📝 Data Format

### NetCDF Files

Your NetCDF files should contain the following variables:
- `lat`: Latitude
- `lon`: Longitude
- `time`: Time of measurement
- `Pressure`: Pressure (dbar)
- `Temperature`: In situ temperature (°C)
- `Salinity`: Salinity (PSU)

### Water Mass Definitions

The `Pontos_MA.txt` file (or text input in the web app) should follow this format:

```
# Comments start with #
WATER_MASS_NAME TEMPERATURE SALINITY
ENACW16 16.0 36.15
MW 13.0 36.50
NEADWL 3.0 34.95
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- World Ocean Database for providing oceanographic data
- TEOS-10 GSW library for accurate seawater calculations
- Streamlit for the excellent web framework
- The oceanography community for water mass classifications

---

**Made with 🌊 for oceanographers and marine scientists**
