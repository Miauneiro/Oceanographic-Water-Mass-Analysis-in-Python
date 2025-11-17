#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oceanographic Water Mass Analysis Module
Provides functions for processing CTD data and analyzing water mass mixing
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import gsw
import glob
import os
from collections import defaultdict
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl import ticker as cticker


# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

# Default depth interpolation
Znew = np.linspace(10, 4000, 400)

# Water mass styles (with RGB colors for mixing visualization)
WATER_MASS_STYLES = {
    'ENACW16': {'cor': 'blue',       'marker': 'o', 'size': 120, 'rgb': (1.0, 0.0, 0.0)},
    'ENACW12': {'cor': 'darkorange', 'marker': 'o', 'size': 120, 'rgb': (1.0, 0.5, 0.0)},
    'MW':      {'cor': 'red',        'marker': 's', 'size': 120, 'rgb': (0.0, 1.0, 0.0)},
    'SAIW1':   {'cor': 'cyan',       'marker': '^', 'size': 120, 'rgb': (0.0, 1.0, 1.0)},
    'SAIW2':   {'cor': 'deepskyblue','marker': '^', 'size': 120, 'rgb': (0.0, 0.7, 1.0)},
    'SPMW8':   {'cor': 'lime',       'marker': 'D', 'size': 100, 'rgb': (0.5, 1.0, 0.0)},
    'SPMW7':   {'cor': 'limegreen',  'marker': 'D', 'size': 100, 'rgb': (0.3, 0.8, 0.0)},
    'IrSPMW':  {'cor': 'darkgreen',  'marker': 'D', 'size': 100, 'rgb': (0.0, 0.5, 0.0)},
    'LSW':     {'cor': 'magenta',    'marker': 'v', 'size': 120, 'rgb': (1.0, 0.0, 1.0)},
    'ISOW':    {'cor': 'navy',       'marker': 's', 'size': 120, 'rgb': (0.0, 0.0, 0.5)},
    'DSOW':    {'cor': 'indigo',     'marker': 's', 'size': 120, 'rgb': (0.3, 0.0, 0.5)},
    'NEADWU':  {'cor': 'teal',       'marker': 'X', 'size': 150, 'rgb': (0.0, 0.5, 0.5)},
    'NEADWL':  {'cor': 'chocolate',  'marker': 'X', 'size': 150, 'rgb': (0.0, 0.0, 1.0)}
}


# ============================================================================
# DATA READING AND CALCULATION FUNCTIONS
# ============================================================================

def read_profiles(ncf, Znew):
    """
    Read temperature and salinity profiles from NetCDF files.

    Parameters
    ----------
    ncf : list
        List of NetCDF file paths
    Znew : array
        Depth levels for interpolation

    Returns
    -------
    la : array
        Latitudes
    lo : array
        Longitudes
    Ti : array
        Times
    Te : array
        Interpolated temperatures
    Se : array
        Interpolated salinities
    """
    n = len(ncf)
    la = np.zeros(n)
    lo = np.zeros(n)
    Ti = np.zeros(n)
    Te = np.zeros([n, len(Znew)])
    Se = np.zeros([n, len(Znew)])

    for k, fi in enumerate(ncf):
        nc = Dataset(fi, 'r')
        la[k] = nc.variables['lat'][:]
        lo[k] = nc.variables['lon'][:]
        Ti[k] = nc.variables['time'][:]
        P = nc.variables['Pressure'][:]
        T = nc.variables['Temperature'][:]
        S = nc.variables['Salinity'][:]

        Te[k, :] = np.interp(Znew, P, T, left=np.nan, right=np.nan)
        Se[k, :] = np.interp(Znew, P, S, left=np.nan, right=np.nan)
        nc.close()

    return la, lo, Ti, Te, Se


def calculate_density(Te, Se, Znew, lo, la):
    """
    Calculate in-situ density using GSW.

    Parameters
    ----------
    Te : array
        Temperature array
    Se : array
        Salinity array
    Znew : array
        Depth levels
    lo : array
        Longitudes
    la : array
        Latitudes

    Returns
    -------
    Rho : array
        Density array
    """
    Rho = np.zeros_like(Te)
    for i in range(Te.shape[0]):
        SA = gsw.SA_from_SP(Se[i, :], Znew, lo[i], la[i])
        CT = gsw.CT_from_t(SA, Te[i, :], Znew)
        Rho[i, :] = gsw.rho(SA, CT, Znew)
    return Rho


def read_water_masses(filepath="Pontos_MA.txt"):
    """
    Read characteristic water mass points from a text file.

    Parameters
    ----------
    filepath : str
        Path to water mass file

    Returns
    -------
    water_masses : defaultdict
        Dictionary of water masses with their T/S properties
    """
    water_masses = defaultdict(lambda: {'sal': [], 'temp': []})

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    try:
                        name = parts[0]
                        temp = float(parts[1])
                        sal = float(parts[2])
                        water_masses[name]['sal'].append(sal)
                        water_masses[name]['temp'].append(temp)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"⚠️ File {filepath} not found. Water masses will not be plotted.")

    return water_masses


def parse_water_masses_text(text_content):
    """
    Parse water mass data from text content.

    Parameters
    ----------
    text_content : str
        Text content with water mass data

    Returns
    -------
    water_masses : defaultdict
        Dictionary of water masses with their T/S properties
    """
    water_masses = defaultdict(lambda: {'sal': [], 'temp': []})

    for line in text_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 3:
            try:
                name = parts[0]
                temp = float(parts[1])
                sal = float(parts[2])
                water_masses[name]['sal'].append(sal)
                water_masses[name]['temp'].append(temp)
            except ValueError:
                continue

    return water_masses


def calculate_mixing_percentages(T_obs, S_obs, T1, S1, T2, S2, T3, S3):
    """
    Calculate mixing percentages of three water masses using OMP analysis.

    System of equations:
    m1*T1 + m2*T2 + m3*T3 = T_obs
    m1*S1 + m2*S2 + m3*S3 = S_obs
    m1 + m2 + m3 = 1

    Parameters
    ----------
    T_obs, S_obs : float
        Observed temperature and salinity
    T1, S1, T2, S2, T3, S3 : float
        Temperature and salinity of three end-member water masses

    Returns
    -------
    m : array or None
        Mixing fractions (m1, m2, m3) or None if point outside triangle
    """
    # System matrix
    A = np.array([
        [T1, T2, T3],
        [S1, S2, S3],
        [1,  1,  1]
    ])

    b = np.array([T_obs, S_obs, 1])

    try:
        # Solve linear system
        m = np.linalg.solve(A, b)

        # Check if solution is physical (0 <= mi <= 1)
        if np.all(m >= -0.01) and np.all(m <= 1.01):  # Numerical tolerance
            m = np.clip(m, 0, 1)  # Correct small numerical errors
            return m
        else:
            return None  # Point outside triangle
    except np.linalg.LinAlgError:
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_profiles(Te, Znew):
    """Plot vertical temperature profiles."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(Te.shape[0]):
        ax.plot(Te[k, :], -Znew, label=f'Profile {k}', linewidth=1.5)

    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title('Interpolated Profiles - In Situ Temperature', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=6, fontsize='small', frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_map(lo, la, Ti):
    """Plot map with CTD profile locations."""
    # Check for empty or invalid data
    if len(lo) == 0 or len(la) == 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, 'No location data available', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("CTD Profile Locations", fontsize=12, fontweight='bold')
        return fig
    
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    ax.plot(lo, la, 'b-', linewidth=2, label='Track', transform=ccrs.PlateCarree())
    ax.scatter(lo, la, c='red', s=100, marker='*',
               edgecolors='black', linewidths=1, transform=ccrs.PlateCarree(), zorder=5)

    for i in range(len(la)):
        ax.text(lo[i] + 0.05, la[i] + 0.05, str(i), fontsize=10,
                fontweight='bold', transform=ccrs.PlateCarree())

    ax.set_extent([min(lo)-1, max(lo)+1, min(la)-1, max(la)+1], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    ax.set_title("CTD Profile Locations", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_section(X, Y, campo, titulo, label, cmap='viridis'):
    """Plot vertical section of any field."""
    # Check if we have enough data for a 2D contour
    if campo.shape[0] < 2 or campo.shape[1] < 2:
        # Not enough profiles for a section - create a simple plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Vertical sections require at least 2 CTD profiles.\nPlease upload multiple profiles from a transect.', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Accumulated distance (km)', fontsize=11)
        ax.set_ylabel('Depth (m)', fontsize=11)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # Handle empty or all-NaN data
    valid_data = campo[~np.isnan(campo)]
    
    if len(valid_data) == 0:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No valid data available for this section', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlabel('Accumulated distance (km)', fontsize=11)
        ax.set_ylabel('Depth (m)', fontsize=11)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    clevs = np.linspace(np.nanmin(campo), np.nanmax(campo), 20)

    fig, ax = plt.subplots(figsize=(10, 6))
    cp = ax.contourf(X, -Y, campo, clevs, cmap=cmap)
    plt.colorbar(cp, label=label, ax=ax)
    ax.set_xlabel('Accumulated distance (km)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ternary(percentages, names=["ENACW16", "MW", "NEADWL"]):
    """Plot ternary diagram showing average mixing percentages."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Coordinates of equilateral triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    # Draw triangle
    triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)

    # Labels at vertices
    offset = 0.08
    ax.text(vertices[0, 0], vertices[0, 1] - offset, names[0],
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=WATER_MASS_STYLES[names[0]]['cor'])
    ax.text(vertices[1, 0], vertices[1, 1] - offset, names[1],
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=WATER_MASS_STYLES[names[1]]['cor'])
    ax.text(vertices[2, 0], vertices[2, 1] + offset, names[2],
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=WATER_MASS_STYLES[names[2]]['cor'])

    # Convert percentages (m1, m2, m3) to Cartesian coordinates
    m1, m2, m3 = percentages
    x = m2 * vertices[1, 0] + m3 * vertices[2, 0]
    y = m3 * vertices[2, 1]

    # Plot average point
    ax.plot(x, y, 'ko', markersize=15, markeredgewidth=2, markerfacecolor='yellow',
            zorder=10, label='Average Composition')

    # Add grid lines
    for i in np.arange(0.2, 1.0, 0.2):
        # Lines parallel to base
        x_line = [i * vertices[1, 0], i * vertices[1, 0] + (1-i) * vertices[2, 0]]
        y_line = [0, (1-i) * vertices[2, 1]]
        ax.plot(x_line, y_line, 'gray', linewidth=0.5, alpha=0.5)

        # Lines parallel to left side
        x_line = [(1-i) * vertices[0, 0] + i * vertices[2, 0],
                  (1-i) * vertices[1, 0] + i * vertices[2, 0]]
        y_line = [i * vertices[2, 1], i * vertices[2, 1]]
        ax.plot(x_line, y_line, 'gray', linewidth=0.5, alpha=0.5)

        # Lines parallel to right side
        x_line = [i * vertices[0, 0] + (1-i) * vertices[1, 0],
                  i * vertices[0, 0] + (1-i) * vertices[2, 0]]
        y_line = [0, (1-i) * vertices[2, 1]]
        ax.plot(x_line, y_line, 'gray', linewidth=0.5, alpha=0.5)

    # Text with percentages
    info_text = f'{names[0]}: {m1*100:.1f}%\n'
    info_text += f'{names[1]}: {m2*100:.1f}%\n'
    info_text += f'{names[2]}: {m3*100:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.5, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace', fontweight='bold')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ternary Diagram - Average Water Mass Composition',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def plot_ts_rgb_mixing(T, S, P, water_masses):
    """
    Plot T-S diagram where CTD points are colored by RGB mixing
    of three water masses. Works with any 3 water masses.
    """
    # Filter valid data
    mask = (~np.isnan(T)) & (~np.isnan(S)) & (T > 0) & (S > 0)
    T_valid, S_valid, P_valid = T[mask], S[mask], P[mask]

    # Grid for isopycnals
    tvec = np.arange(0, 20, 0.5)
    svec = np.arange(34, 37, 0.1)
    Tm, Sm = np.meshgrid(tvec, svec)
    dens = gsw.sigma0(Sm, Tm)

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 8))

    # Isopycnals
    CS = ax.contour(Sm, Tm, dens, np.arange(20, 30, 0.2), colors='gray',
                    linewidths=0.8, alpha=0.6)
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.1f")

    # Check if we have exactly 3 water masses for RGB mixing
    if len(water_masses) == 3:
        # Get the three water masses (in whatever order they are)
        wm_names = list(water_masses.keys())
        wm1_name, wm2_name, wm3_name = wm_names[0], wm_names[1], wm_names[2]
        
        # Get coordinates
        T1, S1 = water_masses[wm1_name]['temp'][0], water_masses[wm1_name]['sal'][0]
        T2, S2 = water_masses[wm2_name]['temp'][0], water_masses[wm2_name]['sal'][0]
        T3, S3 = water_masses[wm3_name]['temp'][0], water_masses[wm3_name]['sal'][0]
        
        # Assign RGB colors (Red, Green, Blue)
        # Check if we have the standard ones, otherwise use generic assignment
        if wm1_name in WATER_MASS_STYLES and 'rgb' in WATER_MASS_STYLES[wm1_name]:
            rgb1 = WATER_MASS_STYLES[wm1_name]['rgb']
        else:
            rgb1 = (1.0, 0.0, 0.0)  # Red for first water mass
        
        if wm2_name in WATER_MASS_STYLES and 'rgb' in WATER_MASS_STYLES[wm2_name]:
            rgb2 = WATER_MASS_STYLES[wm2_name]['rgb']
        else:
            rgb2 = (0.0, 1.0, 0.0)  # Green for second water mass
        
        if wm3_name in WATER_MASS_STYLES and 'rgb' in WATER_MASS_STYLES[wm3_name]:
            rgb3 = WATER_MASS_STYLES[wm3_name]['rgb']
        else:
            rgb3 = (0.0, 0.0, 1.0)  # Blue for third water mass
        T3, S3 = water_masses["NEADWL"]['temp'][0], water_masses["NEADWL"]['sal'][0]

        rgb1 = np.array(WATER_MASS_STYLES["ENACW16"]['rgb'])  # Red
        rgb2 = np.array(WATER_MASS_STYLES["MW"]['rgb'])       # Green
        rgb3 = np.array(WATER_MASS_STYLES["NEADWL"]['rgb'])   # Blue

        # Calculate RGB colors for each CTD point
        rgb_colors = []
        points_inside = 0
        percentages_acum = np.zeros(3)

        for i in range(len(T_valid)):
            result = calculate_mixing_percentages(
                T_valid[i], S_valid[i], T1, S1, T2, S2, T3, S3
            )

            if result is not None:
                points_inside += 1
                percentages_acum += result
                # RGB color = weighted mixture of vertex colors
                mixed_color = result[0] * rgb1 + result[1] * rgb2 + result[2] * rgb3
                rgb_colors.append(mixed_color)
            else:
                # Points outside triangle = gray
                rgb_colors.append([0.7, 0.7, 0.7])

        rgb_colors = np.array(rgb_colors)

        # Plot CTD points with mixed colors
        ax.scatter(S_valid, T_valid, c=rgb_colors, s=20,
                  alpha=0.8, edgecolors='black', linewidths=0.2)

        # Draw triangle
        x = [S1, S2, S3, S1]
        y = [T1, T2, T3, T1]
        ax.plot(x, y, 'k--', linewidth=1.5, alpha=0.8, zorder=8, label='Mixing Triangle')

        # Plot vertices with their pure RGB colors
        ax.scatter([S1], [T1], marker='o', s=200, color=rgb1,
                  edgecolors='black', linewidths=2, zorder=10, label=f'{wm1_name} (R)')
        ax.scatter([S2], [T2], marker='s', s=200, color=rgb2,
                  edgecolors='black', linewidths=2, zorder=10, label=f'{wm2_name} (G)')
        ax.scatter([S3], [T3], marker='X', s=250, color=rgb3,
                  edgecolors='black', linewidths=2, zorder=10, label=f'{wm3_name} (B)')

        # Information
        if points_inside > 0:
            percentages_med = percentages_acum / points_inside
            info_text = f'RGB Water Mass Mixing\n'
            info_text += f'\nPoints in triangle:\n{points_inside}/{len(T_valid)} '
            info_text += f'({100*points_inside/len(T_valid):.1f}%)\n'
            info_text += f'\nAverage mixing:\n'
            info_text += f'{wm1_name}: {percentages_med[0]*100:.1f}%\n'
            info_text += f'{wm2_name}: {percentages_med[1]*100:.1f}%\n'
            info_text += f'{wm3_name}: {percentages_med[2]*100:.1f}%\n'
            info_text += f'\nColor = m₁·R + m₂·G + m₃·B'
        else:
            info_text = 'No points inside mixing triangle'

        props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

    ax.set_xlabel('Salinity (PSU)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title('T-S Diagram - RGB Water Mass Mixing\n'
                '(Color = proportional contribution of each mass)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_ts_diagram(T, S, P, water_masses, triangulo=True):
    """Plot T-S diagram with isopycnals and water masses."""
    # Filter valid data
    mask = (~np.isnan(T)) & (~np.isnan(S)) & (T > 0) & (S > 0)
    T_valid, S_valid, P_valid = T[mask], S[mask], P[mask]

    # Grid for isopycnals
    tvec = np.arange(0, 20, 0.5)
    svec = np.arange(34, 37, 0.1)
    Tm, Sm = np.meshgrid(tvec, svec)
    dens = gsw.sigma0(Sm, Tm)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Isopycnals
    CS = ax.contour(Sm, Tm, dens, np.arange(20, 30, 0.2), colors='gray',
                    linewidths=0.8, alpha=0.6)
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.1f")

    # CTD data
    sc = ax.scatter(S_valid, T_valid, c=P_valid, cmap='viridis', s=15,
                    alpha=0.7, edgecolors='none', label='CTD Data')
    cbar = plt.colorbar(sc, label='Pressure (dbar)', pad=0.02, ax=ax)

    # Water masses
    percentages_medias = None

    for name, data in water_masses.items():
        style = WATER_MASS_STYLES.get(name, {'cor': 'black', 'marker': 'o', 'size': 100})
        ax.scatter(data['sal'], data['temp'],
                   marker=style['marker'],
                   color=style['cor'],
                   s=style['size'],
                   edgecolors='black',
                   linewidths=1.5,
                   label=name,
                   zorder=10)

    # Mixing triangle (ENACW16 - MW - NEADWL)
    if triangulo and all(n in water_masses for n in ["ENACW16", "MW", "NEADWL"]):
        pontos = ["ENACW16", "MW", "NEADWL"]
        x = [water_masses[p]['sal'][0] for p in pontos]
        y = [water_masses[p]['temp'][0] for p in pontos]
        x.append(x[0])  # Close triangle
        y.append(y[0])
        ax.plot(x, y, 'k--', linewidth=1.2, alpha=0.7, zorder=8)

        # Calculate mixing percentages
        T1, S1 = water_masses["ENACW16"]['temp'][0], water_masses["ENACW16"]['sal'][0]
        T2, S2 = water_masses["MW"]['temp'][0], water_masses["MW"]['sal'][0]
        T3, S3 = water_masses["NEADWL"]['temp'][0], water_masses["NEADWL"]['sal'][0]

        points_inside = 0
        percentages_medias = np.zeros(3)

        for i in range(len(T_valid)):
            result = calculate_mixing_percentages(
                T_valid[i], S_valid[i], T1, S1, T2, S2, T3, S3
            )

            if result is not None:
                points_inside += 1
                percentages_medias += result

        # Statistics
        if points_inside > 0:
            percentages_medias /= points_inside
            info_text = f'Mixing Analysis\n'
            info_text += f'\nPoints in triangle:\n{points_inside}/{len(T_valid)} '
            info_text += f'({100*points_inside/len(T_valid):.1f}%)\n'
            info_text += f'\nAverage contributions:\n'
            info_text += f'ENACW16: {percentages_medias[0]*100:.1f}%\n'
            info_text += f'MW: {percentages_medias[1]*100:.1f}%\n'
            info_text += f'NEADWL: {percentages_medias[2]*100:.1f}%'

            props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                        edgecolor='black', linewidth=1.5)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')

    ax.set_xlabel('Salinity (PSU)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title('T-S Diagram with Isopycnals and Water Masses',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', ncol=2, framealpha=0.95)
    plt.tight_layout()

    return fig, percentages_medias


def plot_mixing_histograms(T, S, P, water_masses):
    """
    Plot histograms showing distribution of water mass contribution percentages.
    Works specifically with 3 water masses: ENACW16, MW, NEADWL.
    
    Returns:
        fig: matplotlib figure
        stats: dictionary with statistics for each water mass
    """
    # Filter valid data
    mask = (~np.isnan(T)) & (~np.isnan(S)) & (T > 0) & (S > 0)
    T_valid, S_valid = T[mask], S[mask]
    
    # Check if we have the required 3 water masses
    if not all(n in water_masses for n in ["ENACW16", "MW", "NEADWL"]):
        return None, None
    
    T1, S1 = water_masses["ENACW16"]['temp'][0], water_masses["ENACW16"]['sal'][0]
    T2, S2 = water_masses["MW"]['temp'][0], water_masses["MW"]['sal'][0]
    T3, S3 = water_masses["NEADWL"]['temp'][0], water_masses["NEADWL"]['sal'][0]
    
    # Calculate percentages for all points
    perc_enacw16 = []
    perc_mw = []
    perc_neadwl = []
    
    for i in range(len(T_valid)):
        # Using OMP method to calculate mixing fractions
        # System: T = f1*T1 + f2*T2 + f3*T3, S = f1*S1 + f2*S2 + f3*S3, f1+f2+f3=1
        A = np.array([[T1, T2, T3], 
                      [S1, S2, S3], 
                      [1, 1, 1]])
        b = np.array([T_valid[i], S_valid[i], 1])
        
        try:
            fractions = np.linalg.solve(A, b)
            if np.all(fractions >= -0.05) and np.all(fractions <= 1.05):  # Allow small numerical errors
                perc_enacw16.append(fractions[0] * 100)
                perc_mw.append(fractions[1] * 100)
                perc_neadwl.append(fractions[2] * 100)
        except:
            continue
    
    if len(perc_enacw16) == 0:
        return None, None
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    bins = np.arange(0, 105, 5)  # 5% bins
    
    # Histogram ENACW16
    axes[0].hist(perc_enacw16, bins=bins, color=WATER_MASS_STYLES['ENACW16']['rgb'], 
                 alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[0].axvline(np.mean(perc_enacw16), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(perc_enacw16):.1f}%')
    axes[0].axvline(np.median(perc_enacw16), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(perc_enacw16):.1f}%')
    axes[0].set_xlabel('Contribution Percentage (%)', fontsize=11)
    axes[0].set_ylabel('Number of Measurements', fontsize=11)
    axes[0].set_title('Distribution - ENACW16 (Eastern North Atlantic Central Water)', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 100)
    
    # Statistics text box
    textstr = f'n = {len(perc_enacw16)}\n'
    textstr += f'Min: {np.min(perc_enacw16):.1f}%\n'
    textstr += f'Max: {np.max(perc_enacw16):.1f}%\n'
    textstr += f'Std Dev: {np.std(perc_enacw16):.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Histogram MW
    axes[1].hist(perc_mw, bins=bins, color=WATER_MASS_STYLES['MW']['rgb'], 
                 alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[1].axvline(np.mean(perc_mw), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(perc_mw):.1f}%')
    axes[1].axvline(np.median(perc_mw), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(perc_mw):.1f}%')
    axes[1].set_xlabel('Contribution Percentage (%)', fontsize=11)
    axes[1].set_ylabel('Number of Measurements', fontsize=11)
    axes[1].set_title('Distribution - MW (Mediterranean Water)', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 100)
    
    textstr = f'n = {len(perc_mw)}\n'
    textstr += f'Min: {np.min(perc_mw):.1f}%\n'
    textstr += f'Max: {np.max(perc_mw):.1f}%\n'
    textstr += f'Std Dev: {np.std(perc_mw):.1f}%'
    axes[1].text(0.02, 0.98, textstr, transform=axes[1].transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Histogram NEADWL
    axes[2].hist(perc_neadwl, bins=bins, color=WATER_MASS_STYLES['NEADWL']['rgb'], 
                 alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[2].axvline(np.mean(perc_neadwl), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(perc_neadwl):.1f}%')
    axes[2].axvline(np.median(perc_neadwl), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(perc_neadwl):.1f}%')
    axes[2].set_xlabel('Contribution Percentage (%)', fontsize=11)
    axes[2].set_ylabel('Number of Measurements', fontsize=11)
    axes[2].set_title('Distribution - NEADWL (North East Atlantic Deep Water - Lower)', 
                     fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 100)
    
    textstr = f'n = {len(perc_neadwl)}\n'
    textstr += f'Min: {np.min(perc_neadwl):.1f}%\n'
    textstr += f'Max: {np.max(perc_neadwl):.1f}%\n'
    textstr += f'Std Dev: {np.std(perc_neadwl):.1f}%'
    axes[2].text(0.02, 0.98, textstr, transform=axes[2].transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.suptitle('Water Mass Contribution Distribution', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Return statistics dictionary
    stats = {
        'ENACW16': {
            'mean': np.mean(perc_enacw16),
            'median': np.median(perc_enacw16),
            'std': np.std(perc_enacw16),
            'min': np.min(perc_enacw16),
            'max': np.max(perc_enacw16),
            'n': len(perc_enacw16)
        },
        'MW': {
            'mean': np.mean(perc_mw),
            'median': np.median(perc_mw),
            'std': np.std(perc_mw),
            'min': np.min(perc_mw),
            'max': np.max(perc_mw),
            'n': len(perc_mw)
        },
        'NEADWL': {
            'mean': np.mean(perc_neadwl),
            'median': np.median(perc_neadwl),
            'std': np.std(perc_neadwl),
            'min': np.min(perc_neadwl),
            'max': np.max(perc_neadwl),
            'n': len(perc_neadwl)
        }
    }
    
    return fig, stats