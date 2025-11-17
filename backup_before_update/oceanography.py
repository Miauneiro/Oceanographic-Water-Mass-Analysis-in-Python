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
MA_ESTILO = {
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

def ler_perfis(ncf, Znew):
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


def calcular_densidade(Te, Se, Znew, lo, la):
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


def ler_massas_agua(ficheiro="Pontos_MA.txt"):
    """
    Read characteristic water mass points from a text file.

    Parameters
    ----------
    ficheiro : str
        Path to water mass file

    Returns
    -------
    grupo : defaultdict
        Dictionary of water masses with their T/S properties
    """
    grupo = defaultdict(lambda: {'sal': [], 'temp': []})

    try:
        with open(ficheiro, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    try:
                        nome = parts[0]
                        temp = float(parts[1])
                        sal = float(parts[2])
                        grupo[nome]['sal'].append(sal)
                        grupo[nome]['temp'].append(temp)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"⚠️ File {ficheiro} not found. Water masses will not be plotted.")

    return grupo


def parse_massas_agua_text(text_content):
    """
    Parse water mass data from text content.

    Parameters
    ----------
    text_content : str
        Text content with water mass data

    Returns
    -------
    grupo : defaultdict
        Dictionary of water masses with their T/S properties
    """
    grupo = defaultdict(lambda: {'sal': [], 'temp': []})

    for line in text_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 3:
            try:
                nome = parts[0]
                temp = float(parts[1])
                sal = float(parts[2])
                grupo[nome]['sal'].append(sal)
                grupo[nome]['temp'].append(temp)
            except ValueError:
                continue

    return grupo


def calcular_percentagens_mistura(T_obs, S_obs, T1, S1, T2, S2, T3, S3):
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

def plotar_perfis(Te, Znew):
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


def plotar_mapa(lo, la, Ti):
    """Plot map with CTD profile locations."""
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


def plotar_secao(X, Y, campo, titulo, label, cmap='viridis'):
    """Plot vertical section of any field."""
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


def plotar_ternary(percentagens, nomes=["ENACW16", "MW", "NEADWL"]):
    """Plot ternary diagram showing average mixing percentages."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Coordinates of equilateral triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    # Draw triangle
    triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)

    # Labels at vertices
    offset = 0.08
    ax.text(vertices[0, 0], vertices[0, 1] - offset, nomes[0],
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=MA_ESTILO[nomes[0]]['cor'])
    ax.text(vertices[1, 0], vertices[1, 1] - offset, nomes[1],
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=MA_ESTILO[nomes[1]]['cor'])
    ax.text(vertices[2, 0], vertices[2, 1] + offset, nomes[2],
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=MA_ESTILO[nomes[2]]['cor'])

    # Convert percentages (m1, m2, m3) to Cartesian coordinates
    m1, m2, m3 = percentagens
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
    textstr = f'{nomes[0]}: {m1*100:.1f}%\n'
    textstr += f'{nomes[1]}: {m2*100:.1f}%\n'
    textstr += f'{nomes[2]}: {m3*100:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=11,
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


def plotar_TS_mistura_RGB(T, S, P, grupo):
    """
    Plot T-S diagram where CTD points are colored by RGB mixing
    of three water masses (ENACW16, MW, NEADWL).
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

    # Check if triangle exists
    if all(n in grupo for n in ["ENACW16", "MW", "NEADWL"]):
        # Get coordinates and RGB colors of vertices
        T1, S1 = grupo["ENACW16"]['temp'][0], grupo["ENACW16"]['sal'][0]
        T2, S2 = grupo["MW"]['temp'][0], grupo["MW"]['sal'][0]
        T3, S3 = grupo["NEADWL"]['temp'][0], grupo["NEADWL"]['sal'][0]

        rgb1 = np.array(MA_ESTILO["ENACW16"]['rgb'])  # Red
        rgb2 = np.array(MA_ESTILO["MW"]['rgb'])       # Green
        rgb3 = np.array(MA_ESTILO["NEADWL"]['rgb'])   # Blue

        # Calculate RGB colors for each CTD point
        cores_rgb = []
        pontos_dentro = 0
        percentagens_acum = np.zeros(3)

        for i in range(len(T_valid)):
            resultado = calcular_percentagens_mistura(
                T_valid[i], S_valid[i], T1, S1, T2, S2, T3, S3
            )

            if resultado is not None:
                pontos_dentro += 1
                percentagens_acum += resultado
                # RGB color = weighted mixture of vertex colors
                cor_misturada = resultado[0] * rgb1 + resultado[1] * rgb2 + resultado[2] * rgb3
                cores_rgb.append(cor_misturada)
            else:
                # Points outside triangle = gray
                cores_rgb.append([0.7, 0.7, 0.7])

        cores_rgb = np.array(cores_rgb)

        # Plot CTD points with mixed colors
        ax.scatter(S_valid, T_valid, c=cores_rgb, s=20,
                  alpha=0.8, edgecolors='black', linewidths=0.2)

        # Draw triangle
        x = [S1, S2, S3, S1]
        y = [T1, T2, T3, T1]
        ax.plot(x, y, 'k--', linewidth=1.5, alpha=0.8, zorder=8, label='Mixing Triangle')

        # Plot vertices with their pure RGB colors
        ax.scatter([S1], [T1], marker='o', s=200, color=rgb1,
                  edgecolors='black', linewidths=2, zorder=10, label='ENACW16 (R)')
        ax.scatter([S2], [T2], marker='s', s=200, color=rgb2,
                  edgecolors='black', linewidths=2, zorder=10, label='MW (G)')
        ax.scatter([S3], [T3], marker='X', s=250, color=rgb3,
                  edgecolors='black', linewidths=2, zorder=10, label='NEADWL (B)')

        # Information
        if pontos_dentro > 0:
            percentagens_med = percentagens_acum / pontos_dentro
            textstr = f'RGB Water Mass Mixing\n'
            textstr += f'\nPoints in triangle:\n{pontos_dentro}/{len(T_valid)} '
            textstr += f'({100*pontos_dentro/len(T_valid):.1f}%)\n'
            textstr += f'\nAverage mixing:\n'
            textstr += f'ENACW16: {percentagens_med[0]*100:.1f}%\n'
            textstr += f'MW: {percentagens_med[1]*100:.1f}%\n'
            textstr += f'NEADWL: {percentagens_med[2]*100:.1f}%\n'
            textstr += f'\nColor = m₁·R + m₂·G + m₃·B'
        else:
            textstr = 'No points inside mixing triangle'

        props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
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


def plotar_TS(T, S, P, grupo, triangulo=True):
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
    percentagens_medias = None

    for nome, dados in grupo.items():
        estilo = MA_ESTILO.get(nome, {'cor': 'black', 'marker': 'o', 'size': 100})
        ax.scatter(dados['sal'], dados['temp'],
                   marker=estilo['marker'],
                   color=estilo['cor'],
                   s=estilo['size'],
                   edgecolors='black',
                   linewidths=1.5,
                   label=nome,
                   zorder=10)

    # Mixing triangle (ENACW16 - MW - NEADWL)
    if triangulo and all(n in grupo for n in ["ENACW16", "MW", "NEADWL"]):
        pontos = ["ENACW16", "MW", "NEADWL"]
        x = [grupo[p]['sal'][0] for p in pontos]
        y = [grupo[p]['temp'][0] for p in pontos]
        x.append(x[0])  # Close triangle
        y.append(y[0])
        ax.plot(x, y, 'k--', linewidth=1.2, alpha=0.7, zorder=8)

        # Calculate mixing percentages
        T1, S1 = grupo["ENACW16"]['temp'][0], grupo["ENACW16"]['sal'][0]
        T2, S2 = grupo["MW"]['temp'][0], grupo["MW"]['sal'][0]
        T3, S3 = grupo["NEADWL"]['temp'][0], grupo["NEADWL"]['sal'][0]

        pontos_dentro = 0
        percentagens_medias = np.zeros(3)

        for i in range(len(T_valid)):
            resultado = calcular_percentagens_mistura(
                T_valid[i], S_valid[i], T1, S1, T2, S2, T3, S3
            )

            if resultado is not None:
                pontos_dentro += 1
                percentagens_medias += resultado

        # Statistics
        if pontos_dentro > 0:
            percentagens_medias /= pontos_dentro
            textstr = f'Mixing Analysis\n'
            textstr += f'\nPoints in triangle:\n{pontos_dentro}/{len(T_valid)} '
            textstr += f'({100*pontos_dentro/len(T_valid):.1f}%)\n'
            textstr += f'\nAverage contributions:\n'
            textstr += f'ENACW16: {percentagens_medias[0]*100:.1f}%\n'
            textstr += f'MW: {percentagens_medias[1]*100:.1f}%\n'
            textstr += f'NEADWL: {percentagens_medias[2]*100:.1f}%'

            props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                        edgecolor='black', linewidth=1.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')

    ax.set_xlabel('Salinity (PSU)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title('T-S Diagram with Isopycnals and Water Masses',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', ncol=2, framealpha=0.95)
    plt.tight_layout()

    return fig, percentagens_medias
