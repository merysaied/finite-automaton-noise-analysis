# ============================================================
# app.py - Modern Automaton Robustness Dashboard (Streamlit)
# Run with:  python -m streamlit run app.py
# Requires:  streamlit, matplotlib, numpy, pandas, scikit-learn
# ============================================================
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

# ------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Automaton Robustness Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Color palette used by the CSS and by every chart
# ------------------------------------------------------------
BG = "#0B1020"        # page background
CARD = "#131A2E"      # card background
BORDER = "#222B45"    # card border
TEXT = "#E6E9F5"      # main text
MUTED = "#8A93B2"     # secondary text
ACCENT = "#6C63FF"    # violet accent
TEAL = "#2EC4B6"      # positive / safe
RED = "#FF5C7A"       # negative / danger
BLUE = "#4CC9F0"      # info / active marker
AMBER = "#FFB703"     # warning

# ------------------------------------------------------------
# Global CSS: dark theme, cards, tabs, gradient hero banner
# ------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* Page background and spacing */
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #1a1f3d 0%, {BG} 55%); color: {TEXT}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}

/* Hide default Streamlit chrome */
#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

/* Hero banner */
.hero {{
    background: linear-gradient(135deg, {ACCENT} 0%, #3A86FF 55%, {TEAL} 100%);
    border-radius: 22px; padding: 2.2rem 2.4rem; margin-bottom: 1.6rem;
    box-shadow: 0 20px 50px rgba(108, 99, 255, 0.25);
}}
.hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 800; color: white; letter-spacing: -0.5px; }}
.hero p  {{ margin: .5rem 0 0 0; color: rgba(255,255,255,.88); font-size: 1.02rem; }}
.hero .pill {{
    display: inline-block; margin: 1rem .5rem 0 0; padding: .3rem .8rem; border-radius: 999px;
    background: rgba(255,255,255,.18); color: white; font-size: .8rem; font-weight: 600;
}}

/* KPI cards */
.kpi {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px;
    padding: 1.1rem 1.3rem; height: 100%;
    transition: transform .15s ease, border-color .15s ease;
}}
.kpi:hover {{ transform: translateY(-3px); border-color: {ACCENT}; }}
.kpi .label {{ color: {MUTED}; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }}
.kpi .value {{ font-size: 2rem; font-weight: 800; margin-top: .2rem; }}
.kpi .sub   {{ color: {MUTED}; font-size: .82rem; margin-top: .15rem; }}

/* Generic card + section title */
.card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.2rem 1.4rem; }}
.section-title {{ font-size: 1.15rem; font-weight: 700; margin: .2rem 0 .8rem 0; color: {TEXT}; }}

/* State chips (accepting rooms) */
.chip {{
    display: inline-block; margin: .2rem .3rem .2rem 0; padding: .3rem .75rem; border-radius: 999px;
    font-size: .82rem; font-weight: 600; border: 1px solid {TEAL}; color: {TEAL}; background: rgba(46,196,182,.1);
}}
.chip.start {{ border-color: {ACCENT}; color: {ACCENT}; background: rgba(108,99,255,.12); }}

/* Status banner */
.status {{ border-radius: 16px; padding: 1rem 1.3rem; font-weight: 600; margin: 1rem 0; }}
.status.safe   {{ background: rgba(46,196,182,.12); border: 1px solid {TEAL}; color: {TEAL}; }}
.status.danger {{ background: rgba(255,92,122,.12); border: 1px solid {RED};  color: {RED}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 12px 12px 0 0; padding: .6rem 1.1rem; color: {MUTED}; font-weight: 600;
}}
.stTabs [aria-selected="true"] {{ color: white !important; background: {CARD}; border-bottom: 2px solid {ACCENT}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Matplotlib dark theme so charts blend into the cards
# ------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "grid.color": BORDER, "grid.alpha": 0.7, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": BG, "legend.edgecolor": BORDER, "legend.labelcolor": TEXT,
})


# ------------------------------------------------------------
# Small HTML helpers
# ------------------------------------------------------------
def kpi(label, value, sub="", color=TEXT):
    """Return the HTML of one KPI card (kept on one line to avoid markdown code blocks)."""
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def section(title):
    """Render a section title."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Load the artifacts saved by train_model.py
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts(path="model_artifacts.pkl"):
    """Load the pickle file once and cache it between Streamlit reruns."""
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("`model_artifacts.pkl` not found. Run `python train_model.py` first.")
    st.stop()

clf = artifacts["model"]                        # trained decision tree (the automaton)
X_test = artifacts["X_test"]                    # test features (DataFrame)
y_test = artifacts["y_test"]                    # test labels
Q_states = artifacts["Q"]                       # all states ["q_0", "q_1", ...]
F_accept = artifacts["F"]                       # accepting states (leaves predicting "survived")
q0_start = artifacts["q0"]                      # start state (root node)
results_manifest = artifacts["results"]         # {scenario: [mean accuracy per noise level]}
noise_levels = artifacts["noise_levels"]        # list of noise levels (0.0 ... 1.0)
results_std = artifacts.get("results_std", {})  # {scenario: [std per noise level]}
eps_fill = artifacts["age_mean"]                # epsilon value: the train mean age
baseline = artifacts["baseline_acc"]            # majority-class accuracy on the test set
n_repeats = artifacts.get("n_repeats", 20)      # repetitions used in the analysis
age_min, age_max = artifacts.get("age_range", (0, 80))  # valid age range

# Helper values used in texts
fmt = lambda s: s if isinstance(s, str) else f"q_{s}"   # pretty state name
n_states = len(Q_states)
accept_sorted = sorted(F_accept, key=lambda s: int(str(s).split("_")[-1]))
feat_names = list(X_test.columns)
importances = clf.feature_importances_
top_feature = feat_names[int(np.argmax(importances))]


# ------------------------------------------------------------
# Noise functions: identical logic to train_model.py
# ------------------------------------------------------------
def corrupt_age(X_data, level, rng):
    """Add Gaussian noise proportional to the age, then clip to a valid range."""
    X_noisy = X_data.copy()
    if level > 0:
        noise = level * rng.standard_normal(len(X_noisy)) * X_noisy["Age"]
        X_noisy["Age"] = (X_noisy["Age"] + noise).clip(age_min, age_max)
    return X_noisy


def corrupt_sex(X_data, level, rng):
    """Flip the sex value (0 <-> 1) for a random fraction `level` of passengers."""
    X_noisy = X_data.copy()
    if level > 0:
        mask = rng.random(len(X_noisy)) < level
        X_noisy.loc[mask, "Sex"] = 1 - X_noisy.loc[mask, "Sex"]
    return X_noisy


def corrupt_pclass(X_data, level, rng):
    """Replace the ticket class with a random class for a fraction `level`."""
    X_noisy = X_data.copy()
    if level > 0:
        mask = rng.random(len(X_noisy)) < level
        random_classes = rng.choice([1, 2, 3], size=len(X_noisy))
        X_noisy.loc[mask, "Pclass"] = random_classes[mask]
    return X_noisy


def corrupt_epsilon(X_data, level, rng):
    """Simulate missing data (epsilon): lost ages are replaced by eps_fill."""
    X_noisy = X_data.copy()
    if level > 0:
        mask = rng.random(len(X_noisy)) < level
        X_noisy.loc[mask, "Age"] = eps_fill
    return X_noisy


def corrupt_all(X_data, level, rng):
    """Apply age noise, sex flips and class changes one after another."""
    X_noisy = corrupt_age(X_data, level, rng)
    X_noisy = corrupt_sex(X_noisy, level, rng)
    X_noisy = corrupt_pclass(X_noisy, level, rng)
    return X_noisy


# Scenario name -> noise function (names must match the keys in `results`)
SCENARIO_FUNCS = {
    "Age": corrupt_age,
    "Sex": corrupt_sex,
    "Pclass": corrupt_pclass,
    "All_Combined": corrupt_all,
    "Epsilon_Loss": corrupt_epsilon,
}

# Friendly descriptions shown in the sidebar
SCENARIO_INFO = {
    "Age": "Gaussian noise added to passenger ages.",
    "Sex": "Sex value flipped for a random share of passengers.",
    "Pclass": "Ticket class replaced by a random class.",
    "All_Combined": "Age noise + sex flips + class changes together.",
    "Epsilon_Loss": "Ages lost and replaced by the mean age (ε-transition).",
}

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    plot_key = st.selectbox("Noise scenario", list(results_manifest.keys()))
    st.caption(SCENARIO_INFO.get(plot_key, ""))
    noise_slider = st.slider("Noise rate", 0.0, 1.0, 0.2, 0.05, format="%.2f")
    seed_live = st.number_input("Random seed (live simulation)", 0, 9999, 42, 1)
    st.divider()
    st.caption(f"Averages in Plot A use {n_repeats} repeated runs per noise level.")

# ------------------------------------------------------------
# Live simulation (computed once, used by several tabs)
# ------------------------------------------------------------
rng = np.random.default_rng(int(seed_live))
X_model_input = SCENARIO_FUNCS[plot_key](X_test, noise_slider, rng)
y_pred = clf.predict(X_model_input)
live_acc = accuracy_score(y_test, y_pred)
clean_acc = results_manifest[plot_key][0]       # accuracy with zero noise
drop = (clean_acc - live_acc) * 100             # accuracy lost because of noise
is_safe = live_acc >= baseline

# ============================================================
# HERO BANNER
# ============================================================
st.markdown(
    f"""
<div class="hero">
  <h1>🤖 Decision Tree as an Automaton</h1>
  <p>Explore how noisy inputs push a finite automaton toward failure.</p>
  <span class="pill">{n_states} states</span>
  <span class="pill">{len(F_accept)} accepting</span>
  <span class="pill">{len(X_test)} test samples</span>
  <span class="pill">Scenario: {plot_key}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# TOP KPI ROW
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Live accuracy", f"{live_acc * 100:.1f}%",
                f"at {noise_slider * 100:.0f}% noise", TEAL if is_safe else RED),
            unsafe_allow_html=True)
c2.markdown(kpi("Clean accuracy", f"{clean_acc * 100:.1f}%", "no noise", TEXT),
            unsafe_allow_html=True)
c3.markdown(kpi("Baseline", f"{baseline * 100:.1f}%", "always predict majority", AMBER),
            unsafe_allow_html=True)
c4.markdown(kpi("Accuracy drop", f"-{max(drop, 0):.1f} pts", "vs. clean model",
                RED if drop > 10 else BLUE),
            unsafe_allow_html=True)

# Safe-zone banner
if is_safe:
    st.markdown('<div class="status safe">✅ SAFE ZONE — the automaton still beats the baseline.</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="status danger">⚠️ CRITICAL LIMIT EXCEEDED — accuracy fell below the baseline!</div>',
                unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_auto, tab_rob, tab_live, tab_feat = st.tabs(
    ["🏢 Automaton", "📉 Robustness", "🔬 Live Simulation", "🧠 Feature Importance"]
)

# ------------------------------------------------------------
# TAB 1: Formal definition + tree diagram
# ------------------------------------------------------------
with tab_auto:
    left, right = st.columns([1, 1.7])

    with left:
        section("Formal definition")
        # Q (state set), q0 (start) and F (accepting states) rendered as chips
        st.markdown(
            f'<div class="card"><b>Q</b> — {n_states} states (tree nodes)<br><br>'
            f'<b>q₀</b> — start state<br><span class="chip start">{fmt(q0_start)}</span><br><br>'
            f'<b>F</b> — accepting states (predict “Survived”)<br>'
            + "".join(f'<span class="chip">{fmt(s)}</span>' for s in accept_sorted)
            + '<br><br><b>Σ</b> — input alphabet: '
            + ", ".join(f"<code>{c}</code>" for c in feat_names)
            + "</div>",
            unsafe_allow_html=True,
        )

    with right:
        section("Automaton diagram")
        fig_tree, ax_tree = plt.subplots(figsize=(11, 5.5))
        plot_tree(clf, feature_names=feat_names, class_names=["Died", "Survived"],
                  filled=True, rounded=True, fontsize=7, ax=ax_tree, impurity=False)
        fig_tree.tight_layout()
        st.pyplot(fig_tree, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: Plot A - accuracy vs noise (mean ± std, baseline)
# ------------------------------------------------------------
with tab_rob:
    section(f"Accuracy vs. noise — {plot_key}")
    fig1, ax1 = plt.subplots(figsize=(10, 4.8))

    # Mean accuracy for the selected scenario (percent)
    mean = np.array(results_manifest[plot_key]) * 100
    ax1.plot(noise_levels, mean, marker="o", color=ACCENT, linewidth=2.6,
             markersize=7, label="System accuracy")

    # Optional ±1 standard deviation band
    std = results_std.get(plot_key)
    if std is not None:
        std = np.array(std) * 100
        ax1.fill_between(noise_levels, mean - std, mean + std,
                         color=ACCENT, alpha=0.22, label="±1 σ")

    # Baseline and currently selected noise level
    ax1.axhline(y=baseline * 100, color=AMBER, linestyle="-.", linewidth=1.6,
                label=f"Baseline ({baseline * 100:.0f}%)")
    ax1.axvline(x=noise_slider, color=BLUE, linestyle="--", linewidth=1.6,
                label=f"Active noise ({noise_slider * 100:.0f}%)")

    ax1.set_xlabel("Noise rate")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)
    ax1.legend(loc="best")
    st.pyplot(fig1, use_container_width=True)

    # Comparison of all scenarios at the selected noise level
    section("All scenarios at a glance")
    idx = int(np.argmin(np.abs(np.array(noise_levels) - noise_slider)))  # nearest analysed level
    rows = [{"Scenario": k, "Accuracy at nearest level (%)": round(v[idx] * 100, 1)}
            for k, v in results_manifest.items()]
    df_cmp = pd.DataFrame(rows).sort_values("Accuracy at nearest level (%)", ascending=False)
    st.caption(f"Nearest analysed noise level: {noise_levels[idx] * 100:.0f}%")
    st.dataframe(df_cmp, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# TAB 3: Live simulation (confusion matrix + age histogram)
# ------------------------------------------------------------
with tab_live:
    col_a, col_b = st.columns(2)

    # Confusion matrix of the noisy predictions
    with col_a:
        section("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cmap = LinearSegmentedColormap.from_list("dark_violet", [CARD, ACCENT])
        fig2, ax2 = plt.subplots(figsize=(5, 4.4))
        ax2.imshow(cm, cmap=cmap)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, cm[i, j], ha="center", va="center",
                         fontsize=18, fontweight="bold", color="white")
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Died", "Survived"])
        ax2.set_yticklabels(["Died", "Survived"])
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        for spine in ax2.spines.values():
            spine.set_visible(False)
        st.pyplot(fig2, use_container_width=True)

    # Age distribution before / after noise
    with col_b:
        section("Age distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 4.4))
        ax3.hist(X_test["Age"], bins=20, alpha=0.65, color=TEAL, label="Original")
        ax3.hist(X_model_input["Age"], bins=20, alpha=0.65, color=RED,
                 label=f"After noise ({plot_key})")
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Count")
        ax3.grid(True, axis="y")
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)

    # Short explanation of the live result
    st.markdown(
        f'<div class="card">At <b>{noise_slider * 100:.0f}%</b> noise on <b>{plot_key}</b>, '
        f'the automaton classifies <b>{live_acc * 100:.1f}%</b> of passengers correctly, '
        f'compared with <b>{clean_acc * 100:.1f}%</b> on clean data. '
        f'Live results come from one random run (seed {int(seed_live)}), so they can differ '
        f'slightly from the {n_repeats}-run average in the Robustness tab.</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# TAB 4: Feature importance (actual column names)
# ------------------------------------------------------------
with tab_feat:
    section("Which input symbols drive the transitions?")
    order = np.argsort(importances)                 # sort so the longest bar is on top
    fig4, ax4 = plt.subplots(figsize=(10, 3.6))
    bar_colors = [ACCENT if feat_names[i] == top_feature else BORDER for i in order]
    ax4.barh([feat_names[i] for i in order], importances[order], color=bar_colors, height=0.55)
    for y_pos, i in enumerate(order):
        ax4.text(importances[i] + 0.01, y_pos, f"{importances[i] * 100:.1f}%",
                 va="center", color=TEXT, fontsize=10)
    ax4.set_xlabel("Importance")
    ax4.set_xlim(0, max(importances) * 1.15)
    ax4.grid(True, axis="x")
    st.pyplot(fig4, use_container_width=True)

    st.markdown(
        f'<div class="card">📊 <b>How to read this chart</b><br>'
        f'• <b>Live analysis:</b> the longest bar is <code>{top_feature}</code>; the automaton builds '
        f'its transition functions (δ) mostly around this feature.<br>'
        f'• Features with short bars have little influence on state transitions.</div>',
        unsafe_allow_html=True,
    )