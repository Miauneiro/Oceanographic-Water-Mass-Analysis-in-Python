#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oceanographic Water Mass Analysis - Streamlit App
Interactive web application for analyzing CTD data and water mass mixing
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

# Custom CSS
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
    """Return default water mass definitions."""
    return """# Water Mass Definitions (Name Temperature Salinity)
# Format: NAME TEMP(°C) SALINITY(PSU)
ENACW16 16.0 36.15
MW 13.0 36.50
NEADWL 3.0 34.95
"""


def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Oceanographic Water Mass Analysis</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b> This application analyzes oceanographic CTD (Conductivity, Temperature, Depth) profiles
    and performs water mass mixing analysis using T-S diagrams and the Optimum Multipoint (OMP) method.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Input")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload NetCDF files (.nc)",
            type=['nc'],
            accept_multiple_files=True,
            help="Upload one or more CTD profile NetCDF files"
        )

        st.markdown("---")

        # Water mass definitions
        st.header("💧 Water Mass Definitions")

        use_default = st.checkbox("Use default water masses", value=True)

        if use_default:
            water_mass_text = default_water_masses_text()
            st.info("Using default: ENACW16, MW, NEADWL")
        else:
            water_mass_text = st.text_area(
                "Define water masses (Name Temp Sal):",
                value=default_water_masses_text(),
                height=200,
                help="Format: NAME TEMPERATURE SALINITY (one per line)"
            )

        st.markdown("---")

        # Analysis options
        st.header("⚙️ Options")
        show_profiles = st.checkbox("Show temperature profiles", value=True)
        show_map = st.checkbox("Show map", value=True)
        show_sections = st.checkbox("Show vertical sections", value=True)
        show_ts_diagram = st.checkbox("Show T-S diagram", value=True)
        show_ts_rgb = st.checkbox("Show RGB mixing diagram", value=True)
        show_ternary = st.checkbox("Show ternary diagram", value=True)

    # Main content
    if not uploaded_files:
        st.warning("⬅️ Please upload NetCDF files using the sidebar to begin analysis")

        # Show example/instructions
        st.markdown('<h2 class="section-header">📖 How to Use</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 1️⃣ Upload Data
            - Upload one or more CTD NetCDF files
            - Files should contain: Temperature, Salinity, Pressure, Latitude, Longitude

            ### 2️⃣ Configure Water Masses
            - Use default water masses or define your own
            - Format: NAME TEMPERATURE SALINITY
            - At least 3 water masses needed for mixing analysis
            """)

        with col2:
            st.markdown("""
            ### 3️⃣ View Results
            - Interactive maps and vertical sections
            - T-S diagrams with isopycnals
            - Water mass mixing analysis (OMP method)
            - RGB visualization of mixing ratios

            ### 4️⃣ Interpret Results
            - Red = ENACW16, Green = MW, Blue = NEADWL
            - Ternary diagram shows average composition
            """)

        st.markdown('<h2 class="section-header">🎨 Features</h2>', unsafe_allow_html=True)

        features = [
            "📊 **Data Processing:** Reads and interpolates multiple NetCDF files",
            "🔬 **Scientific Calculations:** Uses GSW (TEOS-10) for accurate density calculations",
            "🌡️ **Water Mass Mixing:** OMP analysis for mixing percentages",
            "🎨 **Advanced Visualization:** Professional plots with RGB color mixing",
            "🗺️ **Geographic Mapping:** Profile locations with Cartopy",
            "📈 **Vertical Sections:** Temperature, Salinity, and Density distributions"
        ]

        for feature in features:
            st.markdown(feature)

        return

    # Process uploaded files
    try:
        # Save uploaded files temporarily
        temp_files = []
        temp_dir = tempfile.mkdtemp()

        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_path)

        # Sort files
        ncf = sorted(temp_files)

        st.success(f"✅ Loaded {len(ncf)} NetCDF file(s)")

        # Read profiles
        with st.spinner("Reading CTD profiles..."):
            la, lo, Ti, Te, Se = ler_perfis(ncf, Znew)

        # Calculate density
        with st.spinner("Calculating density..."):
            Rho = calcular_densidade(Te, Se, Znew, lo, la)

        # Calculate accumulated distance
        d = gsw.geostrophy.distance(lo, la) * 1e-3
        sd = np.zeros(len(la))
        sd[1:] = np.cumsum(d)
        X, Y = np.meshgrid(sd, Znew, indexing='ij')

        # Parse water masses
        grupo = parse_massas_agua_text(water_mass_text)

        if not grupo:
            st.warning("⚠️ No valid water masses defined")

        # Display data summary
        st.markdown('<h2 class="section-header">📊 Data Summary</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Number of Profiles", len(ncf))

        with col2:
            st.metric("Depth Levels", len(Znew))

        with col3:
            st.metric("Max Depth", f"{Znew[-1]:.0f} m")

        with col4:
            st.metric("Water Masses", len(grupo))

        # Display water masses
        if grupo:
            with st.expander("🔍 View Water Mass Properties"):
                for nome, dados in grupo.items():
                    estilo = MA_ESTILO.get(nome, {'cor': 'black'})
                    st.markdown(f"**{nome}** - T: {dados['temp'][0]:.2f}°C, S: {dados['sal'][0]:.3f} PSU")

        # Visualizations
        st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)

        # Temperature profiles
        if show_profiles:
            st.markdown("### 🌡️ Temperature Profiles")
            with st.spinner("Generating temperature profiles..."):
                fig = plotar_perfis(Te, Znew)
                st.pyplot(fig)

        # Map
        if show_map:
            st.markdown("### 🗺️ CTD Profile Locations")
            with st.spinner("Generating map..."):
                try:
                    fig = plotar_mapa(lo, la, Ti)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating map: {str(e)}")

        # Vertical sections
        if show_sections:
            st.markdown("### 📊 Vertical Sections")

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

        # T-S Diagrams
        if show_ts_diagram or show_ts_rgb or show_ternary:
            st.markdown("### 🎨 T-S Diagram Analysis")

            # Read first profile for T-S diagram
            nc = Dataset(ncf[0], 'r')
            T_perfil = nc.variables['Temperature'][:]
            S_perfil = nc.variables['Salinity'][:]
            P_perfil = nc.variables['Pressure'][:]
            nc.close()

            if show_ts_diagram:
                st.markdown("#### Standard T-S Diagram")
                with st.spinner("Generating T-S diagram..."):
                    fig, percentagens = plotar_TS(T_perfil, S_perfil, P_perfil, grupo)
                    st.pyplot(fig)

                    # Show ternary if percentages calculated and option enabled
                    if show_ternary and percentagens is not None:
                        st.markdown("#### Ternary Diagram - Average Composition")
                        with st.spinner("Generating ternary diagram..."):
                            fig = plotar_ternary(percentagens)
                            st.pyplot(fig)

            if show_ts_rgb:
                st.markdown("#### RGB Mixing Visualization")
                st.info("Colors represent mixing ratios: Red=ENACW16, Green=MW, Blue=NEADWL")
                with st.spinner("Generating RGB mixing diagram..."):
                    fig = plotar_TS_mistura_RGB(T_perfil, S_perfil, P_perfil, grupo)
                    st.pyplot(fig)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <b>Oceanographic Water Mass Analysis</b> | Powered by Python, GSW, and Streamlit
        </div>
        """, unsafe_allow_html=True)

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
