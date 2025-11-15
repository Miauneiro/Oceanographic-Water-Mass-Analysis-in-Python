#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oceanographic Water Mass Analysis - Streamlit Application
Interactive web interface for CTD data analysis and water mass mixing evaluation
"""

import streamlit as st
import numpy as np
import gsw
from netCDF4 import Dataset
import tempfile
import os
from oceanography import (
    Znew, MA_ESTILO, ler_perfis, calcular_densidade, parse_massas_agua_text,
    plotar_perfis, plotar_mapa, plotar_secao, plotar_ternary,
    plotar_TS, plotar_TS_mistura_RGB
)

# Page configuration
st.set_page_config(
    page_title="Oceanographic Water Mass Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def default_water_masses_text():
    """Return default water mass definitions for North Atlantic."""
    return """# Water Mass Definitions (Name Temperature Salinity)
# Format: NAME TEMP(°C) SALINITY(PSU)
ENACW16 16.0 36.15
MW 13.0 36.50
NEADWL 3.0 34.95
"""


def main():
    # Header
    st.markdown('<h1 class="main-header">Oceanographic Water Mass Analysis</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Interactive CTD Analysis Platform</b><br>
    Process oceanographic CTD profiles and perform quantitative water mass mixing analysis 
    using Temperature-Salinity diagrams and the Optimum Multiparameter (OMP) method.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("Data Input")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload NetCDF Files (.nc)",
            type=['nc'],
            accept_multiple_files=True,
            help="Upload one or more CTD profile NetCDF files from World Ocean Database or similar sources"
        )

        st.markdown("---")

        # Water mass definitions
        st.header("Water Mass Configuration")

        use_default = st.checkbox("Use default North Atlantic water masses", value=True)

        if use_default:
            water_mass_text = default_water_masses_text()
            st.info("Using default: ENACW16, MW, NEADWL")
        else:
            water_mass_text = st.text_area(
                "Define custom water masses (Name Temp Sal):",
                value=default_water_masses_text(),
                height=200,
                help="Format: NAME TEMPERATURE(°C) SALINITY(PSU) - one per line, minimum 3 required for mixing analysis"
            )

        st.markdown("---")

        # Analysis options
        st.header("Visualization Options")
        show_profiles = st.checkbox("Temperature profiles", value=True)
        show_map = st.checkbox("Geographic map", value=True)
        show_sections = st.checkbox("Vertical sections", value=True)
        show_ts_diagram = st.checkbox("T-S diagram", value=True)
        show_ts_rgb = st.checkbox("RGB mixing visualization", value=True)
        show_ternary = st.checkbox("Ternary composition diagram", value=True)

    # Main content area
    if not uploaded_files:
        st.warning("Please upload NetCDF files using the sidebar to begin analysis")

        # Documentation and instructions
        st.markdown('<h2 class="section-header">Application Overview</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Data Upload
            - Upload one or more CTD NetCDF files
            - Required variables: Temperature, Salinity, Pressure, Latitude, Longitude
            - Compatible with World Ocean Database format

            ### Water Mass Configuration
            - Use predefined North Atlantic water masses
            - Or define custom end-members for other ocean basins
            - Minimum 3 water masses required for mixing triangle analysis
            """)

        with col2:
            st.markdown("""
            ### Analysis Outputs
            - Interactive geographic maps with profile locations
            - Vertical cross-sections of T, S, and density
            - T-S diagrams with isopycnal contours
            - Quantitative water mass mixing analysis via OMP method
            - RGB color visualization of proportional mixing

            ### Scientific Method
            - TEOS-10 equation of state for density calculations
            - Linear system solving for mixing ratio decomposition
            - Ternary diagrams for compositional statistics
            """)

        st.markdown('<h2 class="section-header">Key Features</h2>', unsafe_allow_html=True)

        features = [
            "**Scientific Rigor**: TEOS-10 Gibbs SeaWater library for accurate thermodynamic calculations",
            "**Batch Processing**: Handles multiple CTD casts with automatic interpolation to common depth grid",
            "**Geospatial Visualization**: Publication-quality maps using Cartopy",
            "**Quantitative Analysis**: OMP method for water mass mixing decomposition",
            "**Advanced Visualization**: RGB color mixing shows proportional contributions intuitively",
            "**Flexible Configuration**: Customizable water mass definitions for different ocean regions"
        ]

        for feature in features:
            st.markdown(feature)

        return

    # Data processing workflow
    try:
        # Save uploaded files to temporary directory
        temp_files = []
        temp_dir = tempfile.mkdtemp()

        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_path)

        # Sort files alphabetically
        ncf = sorted(temp_files)

        st.success(f"Successfully loaded {len(ncf)} NetCDF file(s)")

        # Add info message for single profile
        if len(ncf) == 1:
            st.info("Note: You've uploaded a single CTD profile. For best results with vertical sections and track visualization, upload multiple profiles from a transect.")

        # Read and interpolate CTD profiles
        with st.spinner("Reading CTD profiles and interpolating to common depth grid..."):
            la, lo, Ti, Te, Se = ler_perfis(ncf, Znew)

        # Calculate in-situ density using TEOS-10
        with st.spinner("Calculating in-situ density using GSW (TEOS-10)..."):
            Rho = calcular_densidade(Te, Se, Znew, lo, la)

        # Calculate accumulated distance along track
        if len(lo) > 1:
            d = gsw.geostrophy.distance(lo, la) * 1e-3  # Convert to km
            sd = np.zeros(len(la))
            sd[1:] = np.cumsum(d)
        else:
            # Single profile case
            sd = np.array([0.0])
        X, Y = np.meshgrid(sd, Znew, indexing='ij')

        # Parse water mass definitions
        grupo = parse_massas_agua_text(water_mass_text)

        if not grupo:
            st.warning("No valid water masses defined. Please check configuration.")

        # Display data summary
        st.markdown('<h2 class="section-header">Data Summary</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CTD Profiles", len(ncf))

        with col2:
            st.metric("Depth Levels", len(Znew))

        with col3:
            st.metric("Maximum Depth", f"{Znew[-1]:.0f} m")

        with col4:
            st.metric("Water Masses", len(grupo))

        # Display configured water masses
        if grupo:
            with st.expander("View Water Mass Properties"):
                for nome, dados in grupo.items():
                    estilo = MA_ESTILO.get(nome, {'cor': 'black'})
                    st.markdown(f"**{nome}** - Temperature: {dados['temp'][0]:.2f}°C, Salinity: {dados['sal'][0]:.3f} PSU")

        # Generate visualizations
        st.markdown('<h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)

        # Temperature profiles
        if show_profiles:
            st.markdown("### Temperature Profiles by Depth")
            with st.spinner("Generating temperature profiles..."):
                fig = plotar_perfis(Te, Znew)
                st.pyplot(fig)

        # Geographic map
        if show_map:
            st.markdown("### CTD Profile Locations")
            with st.spinner("Generating geographic map..."):
                try:
                    fig = plotar_mapa(lo, la, Ti)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating map: {str(e)}")
                    st.info("Cartopy may require additional system dependencies for map rendering")

        # Vertical sections
        if show_sections:
            st.markdown("### Vertical Cross-Sections")

            tab1, tab2, tab3 = st.tabs(["Temperature", "Salinity", "Density"])

            with tab1:
                with st.spinner("Generating temperature section..."):
                    fig = plotar_secao(X, Y, Te, "Vertical Section - Temperature", "Temperature (°C)")
                    st.pyplot(fig)

            with tab2:
                with st.spinner("Generating salinity section..."):
                    fig = plotar_secao(X, Y, Se, "Vertical Section - Salinity", "Salinity (PSU)")
                    st.pyplot(fig)

            with tab3:
                with st.spinner("Generating density section..."):
                    fig = plotar_secao(X, Y, Rho, "Vertical Section - Density", "Density (kg/m³)")
                    st.pyplot(fig)

        # T-S Diagram Analysis
        if show_ts_diagram or show_ts_rgb or show_ternary:
            st.markdown("### Temperature-Salinity Diagram Analysis")

            # Read first profile for detailed T-S analysis
            nc = Dataset(ncf[0], 'r')
            T_perfil = nc.variables['Temperature'][:]
            S_perfil = nc.variables['Salinity'][:]
            P_perfil = nc.variables['Pressure'][:]
            nc.close()

            if show_ts_diagram:
                st.markdown("#### Standard T-S Diagram with Water Masses")
                with st.spinner("Generating T-S diagram with isopycnals..."):
                    fig, percentagens = plotar_TS(T_perfil, S_perfil, P_perfil, grupo)
                    st.pyplot(fig)

                    # Generate ternary diagram if mixing percentages calculated
                    if show_ternary and percentagens is not None:
                        st.markdown("#### Ternary Composition Diagram")
                        with st.spinner("Generating ternary diagram..."):
                            fig = plotar_ternary(percentagens)
                            st.pyplot(fig)

            if show_ts_rgb:
                st.markdown("#### RGB Water Mass Mixing Visualization")
                st.info("Color channels represent proportional mixing: Red = ENACW16, Green = MW, Blue = NEADWL")
                with st.spinner("Generating RGB mixing diagram..."):
                    fig = plotar_TS_mistura_RGB(T_perfil, S_perfil, P_perfil, grupo)
                    st.pyplot(fig)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <b>Oceanographic Water Mass Analysis</b><br>
        Scientific computing with Python | GSW (TEOS-10) | Streamlit
        </div>
        """, unsafe_allow_html=True)

        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
