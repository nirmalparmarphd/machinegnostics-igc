import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Tuple

from machinegnostics.magcal import DataConversion
from machinegnostics.magcal import GnosticsCharacteristics

st.set_page_config(page_title="Ideal Gnostic Cycle Explorer [Machine Gnostics]", layout="wide")

# --- Helpers ---
def _to_fininf(values: np.ndarray, data_form: str, DL: float, DU: float) -> Tuple[np.ndarray, float, float]:
    """Convert raw values to infinite-interval domain without ELDF."""
    LB, UB = DataConversion.get_zi_bounds(data_form, DL, DU)
    if data_form == 'a':
        z_fin = DataConversion._convert_az(values, DL, DU)
    else:
        z_fin = DataConversion._convert_mz(values, DL, DU)
    z_inf = DataConversion._convert_fininf(z_fin, LB, UB)
    # Ensure numpy array output for consistent downstream handling
    return np.asarray(z_inf, dtype=float), LB, UB

def compute_characteristics(z_obs: np.ndarray,
                            z0: float,
                            S: float,
                            data_form: str,
                            DL: float,
                            DU: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute fi, hi, fj, hj directly from observed Z and Z0 using DataConversion and GnosticsCharacteristics."""
    z_inf_obs, LB, UB = _to_fininf(np.asarray(z_obs, dtype=float), data_form, DL, DU)
    z_inf_z0, _, _ = _to_fininf(np.asarray([z0], dtype=float), data_form, DL, DU)
    z0_inf_scalar = float(z_inf_z0.reshape(-1)[0])
    R = z_inf_obs.reshape(-1, 1) / (z0_inf_scalar + np.finfo(float).eps)
    gc = GnosticsCharacteristics(R=R, verbose=False)
    q, q1 = gc._get_q_q1(S=S)
    fi = gc._fi(q=q, q1=q1).reshape(-1)
    hi = gc._hi(q=q, q1=q1).reshape(-1)
    fj = gc._fj(q=q, q1=q1).reshape(-1)
    hj = gc._hj(q=q, q1=q1).reshape(-1)
    return fi, hi, fj, hj

# Additional helpers for highlighting samples and tracing movement
def select_nearest_indices(values: np.ndarray, target: float, k: int) -> np.ndarray:
    diffs = np.abs(np.asarray(values, dtype=float) - float(target))
    return np.argsort(diffs)[:int(k)]

def _extract_row_stat(arr: np.ndarray, idx: int) -> float:
    a = np.asarray(arr)
    if a.ndim == 1:
        return float(a[idx])
    elif a.ndim == 2:
        return float(np.nanmean(a[idx, :]))
    else:
        return float(np.nanmean(a))

def fi_hi_scatter_for_indices(z_obs: np.ndarray, indices: np.ndarray, z0: float, S: float, data_form: str, DL: float, DU: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fi, hi, _, _ = compute_characteristics(z_obs, z0=z0, S=S, data_form=data_form, DL=DL, DU=DU)
    xs = [_extract_row_stat(fi, int(i)) for i in indices]
    ys = [_extract_row_stat(hi, int(i)) for i in indices]
    zs = [float(z_obs[int(i)]) for i in indices]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)

def path_for_index(z_obs: np.ndarray, idx: int, z0: float, S_vals: np.ndarray, data_form: str, DL: float, DU: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_i_path: list[float] = []
    h_i_path: list[float] = []
    f_j_path: list[float] = []
    h_j_path: list[float] = []
    val = float(z_obs[int(idx)])
    for s in S_vals:
        fi, hi, fj, hj = compute_characteristics(np.asarray([val]), z0=z0, S=float(s), data_form=data_form, DL=DL, DU=DU)
        f_i_path.append(float(fi[0]))
        h_i_path.append(float(hi[0]))
        f_j_path.append(float(fj[0]))
        h_j_path.append(float(hj[0]))
    return np.asarray(f_i_path), np.asarray(h_i_path), np.asarray(f_j_path), np.asarray(h_j_path)

def _parse_values(text: str) -> np.ndarray:
    try:
        vals = [float(x) for x in text.replace('\n', ',').split(',') if x.strip()]
        return np.asarray(vals, dtype=float)
    except Exception:
        return np.array([], dtype=float)

# --- Sidebar ---
st.title("Ideal Gnostic Cycle Explorer - [Machine Gnostics]")
st.markdown("Explore bi-geometric movement of a datum across fidelity (fi) and irrelevance (hi) coordinates, and understand how the Scale parameter (S) shapes local distributions.")

with st.sidebar:
    st.header("Data Input")
    mode = st.radio("Choose input mode", ["Type", "Sample", "Upload"], index=0)
    uploaded = None
    sample_kind = None
    typed = None
    if mode == "Upload":
        uploaded = st.file_uploader("Upload CSV with one numeric column", type=["csv"]) 
        col_name = st.text_input("Column name (for uploaded CSV)", value="value")
    elif mode == "Sample":
        sample_kind = st.selectbox("Sample dataset", ["Synthetic bimodal", "Linear + noise", "Random uniform"], index=0)
    else:
        typed = st.text_area("Observed Z values (comma or newline)", value="11, 12, 13, 14, 15", height=80)

    st.header("Parameters")
    data_form = st.selectbox("Data form", ["a", "m"], index=0, help="'a' additive, 'm' multiplicative")
    Z0 = st.number_input("True value Z0", value=13.0)
    auto_bounds = st.checkbox("Auto bounds (DL/DU from data)", value=True)
    if auto_bounds:
        DL = None
        DU = None
    else:
        DL = st.number_input("Lower bound DL", value=0.0)
        DU = st.number_input("Upper bound DU", value=20.0)

    st.header("S Settings")
    S_min = st.slider("S range min", 0.1, 3.0, 0.5, 0.05)
    S_max = st.slider("S range max", 0.1, 3.0, 2.0, 0.05)
    S_steps = st.slider("S steps", 5, 60, 30, 1)
    plot_mode = st.radio("Plot mode", ["Single", "Multiple"], index=1)
    k_show = st.slider("Samples to show (multiple)", 1, 20, 5, 1)
    plot_height = st.slider("Movement plot height", 300, 900, 540, 30, help="Controls the height of Quantification/Estimation visuals")

# --- Data preparation ---
if mode == "Upload" and uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if col_name not in df.columns:
            st.error(f"Column '{col_name}' not found in uploaded file.")
            st.stop()
        data = df[col_name].dropna().to_numpy()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
elif mode == "Sample":
    rng = np.random.default_rng(7)
    if sample_kind == "Synthetic bimodal":
        a = rng.normal(loc=8.5, scale=0.8, size=200)
        b = rng.normal(loc=12.0, scale=1.0, size=200)
        data = np.concatenate([a, b])
    elif sample_kind == "Linear + noise":
        x = rng.uniform(5, 15, 400)
        y = 0.6 * x + rng.normal(0, 0.8, 400)
        data = y
    else:  # Random uniform
        data = rng.uniform(5.0, 15.0, 400)
else:
    data = _parse_values(typed)
    if data.size == 0:
        st.warning("No observed values parsed. Using defaults: 11, 12, 13, 14, 15")
        data = np.array([11.0, 12.0, 13.0, 14.0, 15.0], dtype=float)

if auto_bounds:
    DL = float(np.min(np.concatenate([data.reshape(-1), np.array([Z0])])))
    DU = float(np.max(np.concatenate([data.reshape(-1), np.array([Z0])])))

S_vals = np.linspace(float(S_min), float(S_max), int(S_steps))

st.subheader("Movement Visuals")
left, right = st.columns(2)

# Sample selection
if plot_mode == "Single":
    sel_idx = st.slider("Select datum index", min_value=0, max_value=int(len(data) - 1), value=0)
    idxs = [int(sel_idx)]
else:
    idxs = list(select_nearest_indices(data, Z0, int(k_show)))

with left:
    st.markdown("**Quantification (Minkowskian): fj vs hj**")
    fig_q = go.Figure()
    # Hyperbola context
    x_h = np.linspace(1.0, 2.6, 240)
    y_h = np.sqrt(np.clip(x_h**2 - 1.0, 0.0, None))
    fig_q.add_trace(go.Scatter(x=x_h, y=y_h, mode='lines', name='x^2 - y^2 = 1', line=dict(color='lightgray', dash='dash')))
    for i in idxs:
        fi_path, hi_path, fj_path, hj_path = path_for_index(data, i, Z0, S_vals, data_form, DL if DL is not None else float(np.min(data)), DU if DU is not None else float(np.max(data)))
        fig_q.add_trace(go.Scatter(x=fj_path, y=hj_path, mode='markers+lines', name=f"Z={data[i]:.2f}",
                                   marker=dict(symbol='x', size=6), line=dict(color='red')))
    fig_q.add_trace(go.Scatter(x=[1.0], y=[0.0], mode='markers+text', name='Ideal (1,0)',
                               text=['Ideal'], textposition='bottom left', marker=dict(size=10, color='black')))
    fig_q.update_layout(height=int(plot_height * 1.2), margin=dict(l=10, r=10, t=30, b=10), xaxis_title='fj (weight)', yaxis_title='hj (irrelevance)')
    st.plotly_chart(fig_q, width='stretch')

with right:
    st.markdown("**Estimation (Euclidean): fi vs hi**")
    fig_e = go.Figure()
    th = np.linspace(0, 2*np.pi, 300)
    fig_e.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode='lines', name='x^2 + y^2 = 1', line=dict(color='lightgray', dash='dash')))
    for i in idxs:
        fi_path, hi_path, fj_path, hj_path = path_for_index(data, i, Z0, S_vals, data_form, DL if DL is not None else float(np.min(data)), DU if DU is not None else float(np.max(data)))
        max_fi = float(np.nanmax(fi_path)) if np.any(np.isfinite(fi_path)) else 1.0
        f_norm = np.clip(fi_path / (max_fi + np.finfo(float).eps), 0.0, 1.0)
        fig_e.add_trace(go.Scatter(x=f_norm, y=hi_path, mode='markers+lines', name=f"Z={data[i]:.2f}",
                                   marker=dict(symbol='x', size=6), line=dict(color='green')))
    fig_e.add_trace(go.Scatter(x=[1.0], y=[0.0], mode='markers+text', name='Ideal (1,0)',
                               text=['Ideal'], textposition='bottom left', marker=dict(size=10, color='black')))
    fig_e.update_layout(height=int(plot_height * 1.2), margin=dict(l=10, r=10, t=30, b=10), xaxis_title='fi (fidelity)', yaxis_title='hi (irrelevance)')
    # Keep the unit circle visually circular by enforcing equal axis scale
    fig_e.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_e, width='stretch')

st.subheader("Notes")
st.markdown("- Movement is computed directly from R = Z/Z0 in the infinite domain using DataConversion and GnosticsCharacteristics.\n- Adjust Z0, observed Z, bounds (DL/DU), and S-range to see changes.\n- Plot either a single datum or multiple nearest to Z0.")

st.info("Run: `streamlit run tests/apps/gnostic_cycle_app.py` from the project root.")

# author
st.markdown("---")
st.markdown("**Author**: Nirmal Parmar, [Machine Gnostics](https://machinegnostics.info)")
