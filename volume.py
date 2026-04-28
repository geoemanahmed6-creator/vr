import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import skew, probplot

st.set_page_config(page_title="Volumetric Risk Analysis", layout="wide")
st.title("🛢️ Volumetric Risk Analysis - Monte Carlo Simulation")
st.markdown("### Professional Reservoir Uncertainty Assessment")

with st.sidebar:
    st.header("⚙️ Simulation Settings")
    iterations = st.number_input("Iterations", min_value=1000, max_value=100000, value=50000, step=1000)
    rock_volume_m3 = st.number_input("Gross Rock Volume (m³)", value=80576000.0)
    st.markdown("### Parameter Distributions (Min / Med / Max)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.subheader("NTG")
    ntg_min = st.number_input("Min", 0.17, key="ntg_min")
    ntg_med = st.number_input("Med", 0.30, key="ntg_med")
    ntg_max = st.number_input("Max", 0.42, key="ntg_max")

with col2:
    st.subheader("Porosity")
    por_min = st.number_input("Min", 0.09, key="por_min")
    por_med = st.number_input("Med", 0.12, key="por_med")
    por_max = st.number_input("Max", 0.18, key="por_max")

with col3:
    st.subheader("Water Saturation")
    sw_min = st.number_input("Min", 0.30, key="sw_min")
    sw_med = st.number_input("Med", 0.40, key="sw_med")
    sw_max = st.number_input("Max", 0.48, key="sw_max")

with col4:
    st.subheader("Recovery Factor")
    rf_min = st.number_input("Min", 0.16, key="rf_min")
    rf_med = st.number_input("Med", 0.18, key="rf_med")
    rf_max = st.number_input("Max", 0.22, key="rf_max")

with col5:
    st.subheader("Boi")
    boi_min = st.number_input("Min", 1.15, key="boi_min")
    boi_med = st.number_input("Med", 1.20, key="boi_med")
    boi_max = st.number_input("Max", 1.28, key="boi_max")

if st.button("🚀 Run Simulation"):
    with st.spinner("Running Monte Carlo..."):
        rock_volume = rock_volume_m3 * 0.0008107132
        np.random.seed(42)

        ntg = np.random.triangular(ntg_min, ntg_med, ntg_max, iterations)
        porosity = np.random.triangular(por_min, por_med, por_max, iterations)
        sw = np.random.triangular(sw_min, sw_med, sw_max, iterations)
        rf = np.random.triangular(rf_min, rf_med, rf_max, iterations)
        boi = np.random.triangular(boi_min, boi_med, boi_max, iterations)

        ooip = (7758 * rock_volume * ntg * porosity * (1 - sw)) / boi
        recoverable_oil = ooip * rf
        rec_mm = recoverable_oil / 1_000_000

        rec_p90 = np.percentile(rec_mm, 10)
        rec_p50 = np.percentile(rec_mm, 50)
        rec_p10 = np.percentile(rec_mm, 90)
        rec_mean = np.mean(rec_mm)
        rec_std = np.std(rec_mm)
        rec_cv = rec_std / rec_mean
        rec_skew = skew(rec_mm)
        var_95 = np.percentile(rec_mm, 5)

        st.subheader("Results (MMSTB)")
        a, b, c, d = st.columns(4)
        a.metric("P90 (Conservative)", f"{rec_p90:.2f}")
        b.metric("P50 (Most Likely)", f"{rec_p50:.2f}")
        c.metric("P10 (Optimistic)", f"{rec_p10:.2f}")
        d.metric("Mean", f"{rec_mean:.2f}")
        e, f, g, h = st.columns(4)
        e.metric("Std Dev", f"{rec_std:.2f}")
        f.metric("CV", f"{rec_cv:.3f}")
        g.metric("Skewness", f"{rec_skew:.3f}")
        h.metric("VaR 95%", f"{var_95:.2f}")

        df = pd.DataFrame({
            'NTG': ntg, 'Porosity': porosity, 'Sw': sw, 'RF': rf, 'Boi': boi, 'Recoverable': rec_mm
        })
        corr = df.corr(method='spearman')['Recoverable'].drop('Recoverable')
        corr_sorted = corr.sort_values(key=abs)

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Monte Carlo Volumetric Analysis", fontweight='bold')
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
        formatter = ticker.FuncFormatter(lambda x, p: f"{x:.1f}")

        sns.histplot(rec_mm, bins=80, kde=True, color='#2ab7ca', ax=ax1)
        ax1.axvline(rec_p90, color='red', linestyle='--', label=f'P90: {rec_p90:.1f}')
        ax1.axvline(rec_p50, color='green', linestyle='-', label=f'P50: {rec_p50:.1f}')
        ax1.axvline(rec_p10, color='blue', linestyle='--', label=f'P10: {rec_p10:.1f}')
        ax1.legend()
        ax1.set_title('1. Distribution + KDE')

        sns.ecdfplot(rec_mm, color='#673ab7', ax=ax2)
        ax2.set_title('2. Cumulative (Less Than)')

        sns.ecdfplot(rec_mm, complementary=True, color='#ff9800', ax=ax3)
        ax3.axhline(0.9, color='red', linestyle=':')
        ax3.axvline(rec_p90, color='red', linestyle='--')
        ax3.axhline(0.5, color='green', linestyle=':')
        ax3.axvline(rec_p50, color='green', linestyle='-')
        ax3.axhline(0.1, color='blue', linestyle=':')
        ax3.axvline(rec_p10, color='blue', linestyle='--')
        ax3.set_title('3. Exceedance (Greater Than)')

        corr_matrix = df.drop('Recoverable', axis=1).corr(method='spearman')
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax4)
        ax4.set_title('4. Spearman Heatmap')

        colors = ['red' if x<0 else 'green' for x in corr_sorted.values]
        ax5.barh(corr_sorted.index, corr_sorted.values, color=colors)
        ax5.axvline(0, color='black')
        ax5.set_title('5. Tornado Chart')
        for i, (_, val) in enumerate(corr_sorted.items()):
            ax5.text(val + 0.02, i, f"{val:.2f}", va='center')

        probplot(rec_mm, dist='norm', plot=ax6)
        ax6.set_title('6. Q-Q Plot')
        ax6.get_lines()[0].set_marker('o')
        ax6.get_lines()[0].set_markersize(2)
        ax6.get_lines()[1].set_color('red')

        plt.tight_layout()
        st.pyplot(fig)

        csv = df[['Recoverable']].head(1000).to_csv(index=False)
        st.download_button("📥 Download sample results", csv, "results.csv", "text/csv")
