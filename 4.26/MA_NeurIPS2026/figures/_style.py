"""
Shared E-style (seaborn ggplot pastel) for all NeurIPS 2026 subfigures.
Keep all subfigure scripts visually consistent.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl


def apply():
    """Apply E ggplot pastel style + figure-friendly rcParams."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        plt.style.use('ggplot')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica'],
        'font.size': 8.5,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'figure.facecolor': 'white',
        'legend.fontsize': 7.5,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#DDDDDD',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


# Regime palette (colorblind-aware pastel)
C_CONC = '#5A8FBE'
C_FS   = '#E58E73'
C_DISP = '#7BB07B'
C_ANOM = '#9E9E9E'
REG = {"CONC": C_CONC, "FS": C_FS, "DISP": C_DISP, "ANOM": C_ANOM}
MARK = {"CONC": "o", "FS": "s", "DISP": "^", "ANOM": "X"}
LBL = {
    "CONC": "Concentrated (n=9)",
    "FS":   "Few-Source (n=7)",
    "DISP": "Dispersed (n=8)",
    "ANOM": "Anomaly (n=2)",
}

# Panel-internal accents (per-model card)
P_BLUE   = '#A4C8E1'
P_TEAL   = '#88C7BC'
P_CORAL  = '#F4B7B7'
P_AMBER  = '#E8C9A4'
P_PURPLE = '#C9B7E8'
P_DARK   = '#5A7A9C'
P_RED    = '#D4756B'
