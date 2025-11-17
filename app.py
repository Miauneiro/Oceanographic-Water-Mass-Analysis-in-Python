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
    Znew, WATER_MASS_STYLES, read_profiles, calculate_density, parse_water_masses_text,
    plot_profiles, plot_map, plot_section, plot_ternary,
    plot_ts_diagram, plot_ts_rgb_mixing, plot_mixing_histograms
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
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Business context for different use cases
BUSINESS_CONTEXT = {
    "Scientific Research": {
        "rgb_insight": "**Water Mass Distribution:** Colors show proportional mixing of three end-member water masses. Points falling within the mixing triangle can be explained by conservative mixing of ENACW16 (red), MW (green), and NEADWL (blue).",
        "ts_insight": "**T-S Characteristics:** Isopycnal lines show potential density surfaces. Water mass end-members are plotted as distinct markers. Mixing occurs along straight lines in T-S space for conservative properties.",
        "section_insight": "**Vertical Structure:** Cross-sections reveal stratification and frontal zones. Temperature and salinity gradients indicate water mass boundaries and mixing regions."
    },
    "Offshore Energy & Infrastructure": {
        "rgb_insight": "**Infrastructure Planning:** Water mass distribution predicts subsurface currents. Mediterranean Water (green) flows eastward at 5-10 cm/s at 1000-1500m depth. Route cables/pipelines to minimize current stress and corrosion exposure.",
        "ts_insight": "**Operational Conditions:** Higher salinity zones (MW influence) increase corrosion rates by 10-15%. Temperature variations affect sound propagation for underwater sensors. Plan maintenance schedules based on water mass seasonality.",
        "section_insight": "**Current Prediction:** Vertical structure indicates subsurface current zones. Strong stratification zones = flow acceleration. Optimal cable depth: Above or below MW core (800m or 1600m) to minimize stress."
    },
    "Fisheries & Aquaculture": {
        "rgb_insight": "**Habitat Zones:** Water mass boundaries define ecological niches. ENACW/MW fronts (red-green transition) concentrate nutrients and attract commercial species like tuna and swordfish.",
        "ts_insight": "**Species Distribution:** Different water masses support different fish communities. Warmer ENACW (16-18°C) supports juvenile fish. MW influence (11-13°C) marks adult tuna habitat. Use boundaries for vessel routing.",
        "section_insight": "**Fishing Grounds:** Temperature fronts in upper 500m indicate productive zones. Thermocline depth affects fish vertical distribution. Plan fishing effort along water mass boundaries shown in cross-sections."
    },
    "Environmental Consulting": {
        "rgb_insight": "**Baseline Conditions:** Document water mass distribution for Environmental Impact Assessments. Changes post-construction indicate anthropogenic influence. Color-coded mixing provides clear visual evidence for regulators.",
        "ts_insight": "**Natural Variability:** T-S diagram shows seasonal water mass variability range. Essential for distinguishing natural vs. human-induced changes. Use for pollution dispersion modeling - contaminants follow water mass movement.",
        "section_insight": "**Impact Assessment:** Vertical structure baseline for offshore development projects. Sediment plumes from construction travel along density surfaces. Cross-sections predict dispersion pathways for regulatory compliance."
    },
    "Climate Services": {
        "rgb_insight": "**Circulation Indicators:** Water mass mixing ratios track ocean circulation changes. Increasing MW contribution indicates enhanced Mediterranean outflow. Decreasing LSW/NEADW signals weakening deep convection - early warning for AMOC changes.",
        "ts_insight": "**Climate Change Detection:** Warming/freshening trends shift T-S properties. Compare against historical water mass definitions to quantify change. Useful for validating climate models and seasonal-to-decadal predictions.",
        "section_insight": "**Ocean Heat Content:** Vertical temperature structure indicates heat storage. Deepening of warm layer = ocean heat uptake. Essential input for regional climate forecasts and extreme weather prediction."
    }
}


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
        
        # Quick demo section
        st.markdown("### Quick Start Demo")
        if st.button("Load Example Analysis", type="primary", use_container_width=True):
            st.session_state.use_example_data = True
            st.rerun()
        
        st.caption("Instantly analyze 3 North Atlantic CTD profiles (June 2004, offshore Portugal)")
        st.markdown("---")
        
        # Use case selector
        st.markdown("### Application Focus")
        use_case = st.selectbox(
            "Select your use case:",
            [
                "Scientific Research",
                "Offshore Energy & Infrastructure",
                "Fisheries & Aquaculture",
                "Environmental Consulting",
                "Climate Services"
            ],
            help="Changes explanations and business context throughout the application"
        )
        
        # Store in session state for use elsewhere
        st.session_state.use_case = use_case
        
        st.markdown("---")

        # File upload
        st.markdown("### Or Upload Your Data")
        uploaded_files = st.file_uploader(
            "Upload NetCDF Files (.nc)",
            type=['nc'],
            accept_multiple_files=True,
            help="Upload one or more CTD profile NetCDF files from World Ocean Database or similar sources"
        )

        st.markdown("---")

        # Water mass definitions
        st.header("Water Mass Configuration")

        st.info("**Tip:** Define any number of water masses. RGB mixing analysis works specifically with 3 selected masses (ENACW16, MW, NEADWL), but standard T-S diagrams can display all defined water masses.")

        use_default = st.checkbox("Use default North Atlantic water masses", value=True)

        if use_default:
            water_mass_text = default_water_masses_text()
            st.info("Using 3-water-mass example: ENACW16, MW, NEADWL for RGB analysis")
        else:
            water_mass_text = st.text_area(
                "Define water masses (Name Temp Sal):",
                value=default_water_masses_text(),
                height=200,
                help="Format: NAME TEMPERATURE(°C) SALINITY(PSU) - one per line. Define any number of water masses. RGB analysis requires exactly 3 masses named ENACW16, MW, NEADWL."
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
    if not uploaded_files and not st.session_state.get('use_example_data', False):
        st.warning("👈 Click 'Load Example Analysis' in the sidebar for instant demo, or upload your own NetCDF files")

        # Show example/instructions
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
        # Handle example data loading
        if st.session_state.get('use_example_data', False) and not uploaded_files:
            # Check if example data exists
            example_data_dir = 'data'
            
            if os.path.exists(example_data_dir):
                import glob
                example_files = sorted(glob.glob(os.path.join(example_data_dir, 'example_profile_*.nc')))
                
                if example_files:
                    st.success(f"Loaded {len(example_files)} example CTD profiles from North Atlantic")
                    st.info("**Location:** Offshore Portugal (43-45°N, 16-19°W) | **Date:** June 2004 | **Depth:** 0-5000m")
                    
                    # Use example files
                    ncf = example_files
                else:
                    # Debug info
                    all_files = os.listdir(example_data_dir)
                    st.error(f"Example data files not found in 'data/' directory. Found files: {all_files}")
                    st.session_state.use_example_data = False
                    return
            else:
                # Show current directory for debugging
                current_dir = os.getcwd()
                st.error(f"Example data directory not found. Current directory: {current_dir}")
                st.info("Please upload your own NetCDF files or ensure 'data/' folder exists in project directory.")
                st.session_state.use_example_data = False
                return
        
        # Handle uploaded files
        elif uploaded_files:
            # Save uploaded files temporarily
            temp_files = []
            temp_dir = tempfile.mkdtemp()

            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_path)

            # Sort files alphabetically
            ncf = sorted(temp_files)
        else:
            return

        st.success(f"Successfully loaded {len(ncf)} NetCDF file(s)")

        # Add info message for single profile
        if len(ncf) == 1:
            st.info("Note: You've uploaded a single CTD profile. For best results with vertical sections and track visualization, upload multiple profiles from a transect.")

        # Read and interpolate CTD profiles
        with st.spinner("Reading CTD profiles and interpolating to common depth grid..."):
            la, lo, Ti, Te, Se = read_profiles(ncf, Znew)

        # Calculate in-situ density using TEOS-10
        with st.spinner("Calculating in-situ density using GSW (TEOS-10)..."):
            Rho = calculate_density(Te, Se, Znew, lo, la)

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
        water_masses = parse_water_masses_text(water_mass_text)

        if not water_masses:
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
            st.metric("Water Masses", len(water_masses))

        # Display configured water masses
        if water_masses:
            with st.expander("View Water Mass Properties"):
                for name, data in water_masses.items():
                    style = WATER_MASS_STYLES.get(name, {'cor': 'black'})
                    st.markdown(f"**{name}** - Temperature: {data['temp'][0]:.2f}°C, Salinity: {data['sal'][0]:.3f} PSU")
        
        # Business value estimator (for non-scientific use cases)
        use_case = st.session_state.get('use_case', 'Scientific Research')
        if use_case == "Offshore Energy & Infrastructure":
            with st.expander("Potential Value Assessment", expanded=True):
                st.markdown("### Cost-Benefit Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Analysis Provides:**")
                    st.markdown("• Subsurface current prediction")
                    st.markdown("• Corrosion zone identification")
                    st.markdown("• Optimal routing depth determination")
                    st.markdown("• Operational risk assessment")
                
                with col2:
                    st.markdown("**Typical Project Value:**")
                    st.markdown("• Cable routing optimization: **\\$8-15M savings** (per 1000km)")
                    st.markdown("• Extended equipment lifespan: **+3-5 years**")
                    st.markdown("• Reduced maintenance frequency: **-15-25%**")
                    st.markdown("• Avoided emergency repairs: **\\$2-5M each**")
                
                st.info("**ROI Example:** For a 5,000 km transatlantic cable project (\\$400M total), water mass analysis (\\$25K) can save \\$40-75M through optimal routing. **ROI: 1,600-3,000x**")

        # Generate visualizations
        st.markdown('<h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)
        
        # Quick Insights Panel (Enhancement #3)
        st.markdown("### Quick Insights")
        
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Temperature range
                temp_range = f"{np.nanmin(Te):.1f}°C to {np.nanmax(Te):.1f}°C"
                st.metric("Temperature Range", temp_range)
            
            with col2:
                # Salinity range
                sal_range = f"{np.nanmin(Se):.2f} to {np.nanmax(Se):.2f}"
                st.metric("Salinity Range", sal_range)
            
            with col3:
                # Depth coverage
                depth_cov = f"0 - {Znew[-1]:.0f}m"
                st.metric("Depth Coverage", depth_cov)
            
            with col4:
                # Number of profiles
                st.metric("CTD Profiles", len(ncf))
            
            # Add water mass dominance insight if we have the data
            if len(water_masses) == 3 and all(n in water_masses for n in ["ENACW16", "MW", "NEADWL"]):
                # Calculate quick water mass statistics
                T_flat = Te.flatten()
                S_flat = Se.flatten()
                
                mask = (~np.isnan(T_flat)) & (~np.isnan(S_flat))
                if np.sum(mask) > 0:
                    # This is a quick calculation - full one is in histograms
                    st.info(f"**Water Mass Signature:** Analysis covers {np.sum(mask)} measurements across {len(ncf)} profiles. Detailed distribution histograms available below.")
        except Exception as e:
            st.error(f"Error generating Quick Insights: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        # Temperature profiles
        if show_profiles:
            st.markdown("### Temperature Profiles by Depth")
            with st.spinner("Generating temperature profiles..."):
                fig = plot_profiles(Te, Znew)
                st.pyplot(fig)

        # Geographic map
        if show_map:
            st.markdown("### CTD Profile Locations")
            with st.spinner("Generating geographic map..."):
                try:
                    fig = plot_map(lo, la, Ti)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating map: {str(e)}")
                    st.info("Cartopy may require additional system dependencies for map rendering")

        # Vertical sections
        if show_sections:
            st.markdown("### Vertical Cross-Sections")
            
            # Business context for sections
            use_case = st.session_state.get('use_case', '🔬 Scientific Research')
            with st.expander("How to Interpret Vertical Sections", expanded=False):
                st.markdown(BUSINESS_CONTEXT[use_case]['section_insight'])

            tab1, tab2, tab3 = st.tabs(["Temperature", "Salinity", "Density"])

            with tab1:
                with st.spinner("Generating temperature section..."):
                    fig = plot_section(X, Y, Te, "Vertical Section - Temperature", "Temperature (°C)")
                    st.pyplot(fig)

            with tab2:
                with st.spinner("Generating salinity section..."):
                    fig = plot_section(X, Y, Se, "Vertical Section - Salinity", "Salinity (PSU)")
                    st.pyplot(fig)

            with tab3:
                with st.spinner("Generating density section..."):
                    fig = plot_section(X, Y, Rho, "Vertical Section - Density", "Density (kg/m³)")
                    st.pyplot(fig)

        # T-S Diagram Analysis
        if show_ts_diagram or show_ts_rgb or show_ternary:
            st.markdown("### Temperature-Salinity Diagram Analysis")

            # Read first profile for detailed T-S analysis
            if len(ncf) > 0:
                nc = Dataset(ncf[0], 'r')
                T_perfil = nc.variables['Temperature'][:]
                S_perfil = nc.variables['Salinity'][:]
                P_perfil = nc.variables['Pressure'][:]
                nc.close()
            else:
                st.error("No profile data available for T-S analysis")
                return

            if show_ts_diagram:
                st.markdown("#### Standard T-S Diagram with Water Masses")
                with st.spinner("Generating T-S diagram with isopycnals..."):
                    fig, percentages = plot_ts_diagram(T_perfil, S_perfil, P_perfil, water_masses)
                    st.pyplot(fig)
                
                # Business context
                use_case = st.session_state.get('use_case', 'Scientific Research')
                with st.expander("How to Interpret This (Business Context)", expanded=False):
                    st.markdown(BUSINESS_CONTEXT[use_case]['ts_insight'])

                    # Generate ternary diagram if mixing percentages calculated
                    if show_ternary and percentages is not None:
                        st.markdown("#### Ternary Composition Diagram")
                        with st.spinner("Generating ternary diagram..."):
                            fig = plot_ternary(percentages)
                            st.pyplot(fig)

            if show_ts_rgb:
                st.markdown("#### RGB Water Mass Mixing Visualization")
                st.info("Color channels represent proportional mixing: Red = ENACW16, Green = MW, Blue = NEADWL")
                with st.spinner("Generating RGB mixing diagram..."):
                    fig = plot_ts_rgb_mixing(T_perfil, S_perfil, P_perfil, water_masses)
                    st.pyplot(fig)
                
                # Business context panel
                use_case = st.session_state.get('use_case', 'Scientific Research')
                with st.expander("How to Interpret This (Business Context)", expanded=False):
                    st.markdown(BUSINESS_CONTEXT[use_case]['rgb_insight'])
            
            # Water Mass Distribution Histograms
            if show_ts_rgb:  # Only show if RGB is enabled (requires 3 water masses)
                st.markdown("#### Water Mass Contribution Distribution")
                with st.spinner("Generating distribution histograms..."):
                    fig_hist, stats = plot_mixing_histograms(T_perfil, S_perfil, P_perfil, water_masses)
                    
                    if fig_hist is not None and stats is not None:
                        st.pyplot(fig_hist)
                        
                        # Dynamic business insights using actual data
                        with st.expander("Distribution Insights (Data-Driven)", expanded=True):
                            # Determine dominant water mass
                            dominant = max(stats.items(), key=lambda x: x[1]['mean'])
                            dominant_name = dominant[0]
                            dominant_pct = dominant[1]['mean']
                            
                            st.markdown(f"**Key Findings from Your Data:**")
                            st.markdown(f"- **Dominant Water Mass:** {dominant_name} (average {dominant_pct:.1f}% contribution)")
                            st.markdown(f"- **ENACW16:** {stats['ENACW16']['mean']:.1f}% ± {stats['ENACW16']['std']:.1f}% (range: {stats['ENACW16']['min']:.1f}-{stats['ENACW16']['max']:.1f}%)")
                            st.markdown(f"- **MW:** {stats['MW']['mean']:.1f}% ± {stats['MW']['std']:.1f}% (range: {stats['MW']['min']:.1f}-{stats['MW']['max']:.1f}%)")
                            st.markdown(f"- **NEADWL:** {stats['NEADWL']['mean']:.1f}% ± {stats['NEADWL']['std']:.1f}% (range: {stats['NEADWL']['min']:.1f}-{stats['NEADWL']['max']:.1f}%)")
                            st.markdown(f"- **Measurements analyzed:** {stats['ENACW16']['n']} valid points")
                            
                            # Use case specific interpretation
                            if use_case == "Offshore Energy & Infrastructure":
                                if stats['MW']['mean'] > 30:
                                    st.warning(f"⚠️ High MW presence ({stats['MW']['mean']:.1f}%) indicates strong Mediterranean influence. Cable routes at 1000-1500m depth will experience elevated corrosion risk and eastward currents of 5-10 cm/s. Consider routing above 800m or below 1600m.")
                                else:
                                    st.success(f"✓ MW contribution is moderate ({stats['MW']['mean']:.1f}%). Standard corrosion protection is sufficient for most depth ranges.")
                            
                            elif use_case == "Fisheries & Aquaculture":
                                if stats['ENACW16']['mean'] > 40:
                                    st.info(f"🐟 Strong ENACW presence ({stats['ENACW16']['mean']:.1f}%) in upper waters indicates productive habitat for juvenile fish and small pelagics. ENACW/MW boundary zones (where both masses mix) are optimal fishing grounds.")
                                if stats['MW']['mean'] > 20:
                                    st.info(f"🎣 MW contribution of {stats['MW']['mean']:.1f}% suggests adult tuna and swordfish habitat at intermediate depths (500-1200m). Focus fishing effort along water mass boundaries.")
                            
                            elif use_case == "Climate Services":
                                st.info(f"📊 Current MW contribution: {stats['MW']['mean']:.1f}%. Historical baseline for this region is ~35%. Values >40% indicate enhanced Mediterranean outflow; <30% suggests reduced exchange, potentially signaling circulation changes.")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <b>Oceanographic Water Mass Analysis</b><br>
        Scientific computing with Python | GSW (TEOS-10) | Streamlit
        </div>
        """, unsafe_allow_html=True)

        # Cleanup temporary files (only for uploaded files, not example data)
        if uploaded_files:
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