# =============================================================================
# app.py — Customer Churn Risk Predictor
# Streamlit web application for real-time churn prediction with SHAP
# =============================================================================
# PLACE THIS FILE IN YOUR PROJECT ROOT (same level as /data, /models)
#
# RUN LOCALLY:
#   streamlit run app.py
#
# DEPLOY:
#   Push to GitHub → Streamlit Cloud → https://streamlit.io/cloud
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title          = "Churn Risk Predictor",
    page_icon           = "📡",
    layout              = "wide",
    initial_sidebar_state = "expanded"
)

# ── Colour palette (consistent with Days 3-9) ─────────────────────────────────
CLR_RETAIN = "#1D9E75"
CLR_CHURN  = "#E24B4A"
CLR_NEUT   = "#378ADD"
CLR_WARN   = "#F4A83A"
CLR_GREY   = "#9E9E9E"
CLR_DARK   = "#2C3E50"
CLR_BG     = "#F4F6F9"

# ── Paths — app.py lives at project root, models/ is a sibling folder ─────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── OHE category definitions — must match Day 4 preprocessing EXACTLY ─────────
OHE_CATEGORIES = {
    "MultipleLines"    : ["No", "No phone service", "Yes"],
    "InternetService"  : ["DSL", "Fiber optic", "No"],
    "OnlineSecurity"   : ["No", "No internet service", "Yes"],
    "OnlineBackup"     : ["No", "No internet service", "Yes"],
    "DeviceProtection" : ["No", "No internet service", "Yes"],
    "TechSupport"      : ["No", "No internet service", "Yes"],
    "StreamingTV"      : ["No", "No internet service", "Yes"],
    "StreamingMovies"  : ["No", "No internet service", "Yes"],
    "Contract"         : ["Month-to-month", "One year", "Two year"],
    "PaymentMethod"    : [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"
    ],
}


# =============================================================================
# LOAD ARTIFACTS  (cached — loads once per session, not on every interaction)
# =============================================================================
@st.cache_resource
def load_artifacts():
    """Load all model artifacts. Returns (model, features, meta, scaler, stats, error)."""
    try:
        model          = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
        feature_names  = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        metadata       = joblib.load(os.path.join(MODELS_DIR, "model_metadata.pkl"))
        scaler         = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        training_stats = joblib.load(os.path.join(MODELS_DIR, "training_stats.pkl"))
        return model, feature_names, metadata, scaler, training_stats, None
    except Exception as e:
        return None, None, None, None, None, str(e)


@st.cache_resource
def load_explainer(_model):
    """Load SHAP TreeExplainer (cached — creation takes ~5 seconds first time)."""
    return shap.TreeExplainer(_model)


# =============================================================================
# PREPROCESSING  (mirrors Day 4 pipeline exactly)
# =============================================================================
def preprocess_input(ui, feature_names, scaler, training_stats):
    """
    Transform raw sidebar inputs into the exact feature vector the model expects.

    Pipeline:
      1. Binary-encode demographic & billing flags
      2. Scale [tenure, MonthlyCharges, TotalCharges] with saved MinMaxScaler
      3. Compute engineered features (charge_per_month, high_value, num_addons, etc.)
      4. One-hot encode all multi-class categoricals
      5. Align columns to model's expected feature order, fill missing with 0

    Parameters
    ----------
    ui             : dict — raw user inputs from the sidebar
    feature_names  : list — ordered feature names from feature_names.pkl
    scaler         : fitted MinMaxScaler from scaler.pkl
    training_stats : dict — p75_monthly and dataset averages

    Returns
    -------
    pd.DataFrame   : single-row DataFrame, column-aligned to model expectations
    """
    row = {}

    # ── 1. Binary features ────────────────────────────────────────────────────
    row["gender"]           = 1 if ui["gender"] == "Male" else 0
    row["SeniorCitizen"]    = int(ui["SeniorCitizen"])
    row["Partner"]          = 1 if ui["Partner"] == "Yes" else 0
    row["Dependents"]       = 1 if ui["Dependents"] == "Yes" else 0
    row["PhoneService"]     = 1 if ui["PhoneService"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if ui["PaperlessBilling"] == "Yes" else 0

    # ── 2. Scale numerical features ───────────────────────────────────────────
    tenure  = float(ui["tenure"])
    monthly = float(ui["MonthlyCharges"])
    total   = tenure * monthly          # estimated TotalCharges

    scaled               = scaler.transform([[tenure, monthly, total]])[0]
    row["tenure"]         = scaled[0]
    row["MonthlyCharges"] = scaled[1]
    row["TotalCharges"]   = scaled[2]   # may be dropped by model; included for safety

    # ── 3. Engineered features (Day 4 Task 6) ─────────────────────────────────
    row["charge_per_month"] = monthly if tenure == 0 else total / tenure
    row["high_value"]       = 1 if monthly >= training_stats["p75_monthly"] else 0

    addon_bases = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    row["num_addons"]    = sum(1 for f in addon_bases if ui.get(f) == "Yes")
    row["tenure_bucket"] = 0 if tenure <= 12 else (1 if tenure <= 36 else 2)
    row["mtm_contract"]  = 1 if ui["Contract"] == "Month-to-month" else 0

    # ── 4. One-hot encoding ───────────────────────────────────────────────────
    for col, categories in OHE_CATEGORIES.items():
        for cat in categories:
            row[f"{col}_{cat}"] = 1 if ui.get(col) == cat else 0

    # ── 5. Align to model's expected feature set ──────────────────────────────
    df = pd.DataFrame([row])
    for feat in feature_names:           # add any columns the model expects
        if feat not in df.columns:
            df[feat] = 0
    return df[feature_names]             # exact order model was trained with


# =============================================================================
# VISUALISATION HELPERS
# =============================================================================
def create_gauge(probability, threshold):
    """
    Semi-circle gauge showing churn probability.
    Needle points to current prediction.
    Dashed line marks the decision threshold.
    """
    if probability < 0.40:
        color, tier, icon = CLR_RETAIN, "LOW RISK",    "🟢"
    elif probability < threshold:
        color, tier, icon = CLR_WARN,   "MEDIUM RISK", "🟡"
    else:
        color, tier, icon = CLR_CHURN,  "HIGH RISK",   "🔴"

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            color="#E8ECEF", linewidth=22, solid_capstyle="round")

    # Coloured fill arc
    fill_end   = np.pi - (probability * np.pi)
    theta_fill = np.linspace(np.pi, fill_end, 300)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=color, linewidth=22, solid_capstyle="round")

    # Needle + hub
    needle_angle = np.pi - (probability * np.pi)
    ax.annotate(
        "", xy=(0.68 * np.cos(needle_angle), 0.68 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=CLR_DARK, lw=2.8, mutation_scale=16)
    )
    ax.plot(0, 0, "o", color=CLR_DARK, markersize=10, zorder=6)

    # Decision threshold marker
    t_angle = np.pi - (threshold * np.pi)
    ax.plot(
        [0.05 * np.cos(t_angle), 0.96 * np.cos(t_angle)],
        [0.05 * np.sin(t_angle), 0.96 * np.sin(t_angle)],
        color=CLR_GREY, lw=1.8, ls="--", alpha=0.65, zorder=4
    )
    ax.text(
        1.1 * np.cos(t_angle), 1.1 * np.sin(t_angle),
        f"Threshold\n{threshold:.0%}",
        ha="center", fontsize=7.5, color=CLR_GREY, va="center"
    )

    # Centre text
    ax.text(0, 0.22, f"{probability * 100:.1f}%",
            ha="center", va="center", fontsize=30, fontweight="bold", color=color)
    ax.text(0, -0.07, "Churn Probability",
            ha="center", va="center", fontsize=10, color="#777")

    # Scale end labels
    ax.text(-1.18, -0.07, "0%",   ha="center", fontsize=8.5, color=CLR_GREY)
    ax.text( 1.18, -0.07, "100%", ha="center", fontsize=8.5, color=CLR_GREY)

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-0.28, 1.28)
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    return fig, color, tier, icon


def create_shap_chart(shap_vals, feat_names, n=12):
    """
    Horizontal bar chart of the top N SHAP feature contributions.
    Red  = pushes toward churn   |  Green = pushes toward retained.
    """
    shap_dict  = dict(zip(feat_names, shap_vals))
    sorted_fts = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

    features = [item[0] for item in sorted_fts[::-1]]   # reverse: top at top
    values   = [item[1] for item in sorted_fts[::-1]]

    # Readable display labels
    def clean(name):
        replacements = [
            ("Contract_",        "Contract: "),
            ("InternetService_", "Internet: "),
            ("PaymentMethod_",   "Payment: "),
            ("OnlineSecurity_",  "OnlineSec: "),
            ("OnlineBackup_",    "Backup: "),
            ("DeviceProtection_","DeviceProt: "),
            ("TechSupport_",     "TechSupp: "),
            ("StreamingTV_",     "StreamTV: "),
            ("StreamingMovies_", "StreamMov: "),
            ("MultipleLines_",   "MultiLine: "),
        ]
        for old, new in replacements:
            name = name.replace(old, new)
        return name.replace("_", " ")

    labels = [clean(f) for f in features]
    colors = [CLR_CHURN if v > 0 else CLR_RETAIN for v in values]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(labels, values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.65)
    ax.axvline(0, color=CLR_DARK, linewidth=0.8, alpha=0.35)

    for bar_obj, val in zip(bars, values):
        offset = 0.0005 if val >= 0 else -0.0005
        ha     = "left" if val >= 0 else "right"
        ax.text(val + offset, bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=8, color=CLR_DARK)

    ax.legend(
        handles=[
            mpatches.Patch(color=CLR_CHURN,  label="↑ Increases churn risk"),
            mpatches.Patch(color=CLR_RETAIN, label="↓ Decreases churn risk"),
        ],
        fontsize=9, loc="lower right"
    )

    ax.set_title("Feature Contributions to This Prediction  (SHAP Values)",
                 fontsize=11, color=CLR_DARK, pad=8, fontweight="normal")
    ax.set_xlabel("SHAP Value  (contribution to churn probability)",
                  fontsize=8.5, color="#666")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDD")
    ax.spines["bottom"].set_color("#DDD")

    plt.tight_layout()
    return fig


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================
def get_recommendations(shap_dict, user_input, probability):
    """
    Generate up to 4 actionable retention recommendations based on
    the top positive SHAP drivers for this specific prediction.
    """
    recs = []
    top_risk = {k.lower(): v for k, v in shap_dict.items() if v > 0.005}

    # ── Rule 1: Month-to-month contract ───────────────────────────────────────
    if (user_input["Contract"] == "Month-to-month" and
            any("month-to-month" in k for k in top_risk)):
        recs.append({
            "icon"    : "📋",
            "title"   : "Offer Annual Contract Upgrade",
            "detail"  : ("Month-to-month customers churn at ~43% vs 11% for 1-year plans. "
                         "Offer a 10–15% monthly discount to lock in an annual contract now."),
            "priority": "HIGH",
            "color"   : CLR_CHURN,
        })

    # ── Rule 2: No online security ─────────────────────────────────────────────
    if (user_input.get("OnlineSecurity") in ("No", "No internet service") and
            any("onlinesec" in k or "onlinesecurity_no" in k for k in top_risk)):
        recs.append({
            "icon"    : "🛡️",
            "title"   : "Bundle Security Add-On",
            "detail"  : ("Customers without Online Security churn at ~42%. "
                         "Offer a 3-month free trial of the Security + Tech Support bundle."),
            "priority": "HIGH",
            "color"   : CLR_CHURN,
        })

    # ── Rule 3: Short tenure (critical early window) ───────────────────────────
    if user_input["tenure"] <= 6:
        recs.append({
            "icon"    : "🎯",
            "title"   : "Early Onboarding Intervention",
            "detail"  : (f"Customer is only {user_input['tenure']} month(s) old — "
                         f"the highest-risk window. Schedule a proactive onboarding call "
                         f"within 48 hours to address friction points."),
            "priority": "HIGH",
            "color"   : CLR_CHURN,
        })

    # ── Rule 4: Fiber optic + high charges ────────────────────────────────────
    if (user_input.get("InternetService") == "Fiber optic" and
            user_input["MonthlyCharges"] > 80):
        recs.append({
            "icon"    : "💡",
            "title"   : "Investigate Service Experience",
            "detail"  : ("High-charge Fiber Optic customers churn at ~42%. "
                         "Check recent support tickets and offer a proactive service audit "
                         "or speed upgrade."),
            "priority": "MEDIUM",
            "color"   : CLR_WARN,
        })

    # ── Rule 5: Electronic check payment method ────────────────────────────────
    if (user_input.get("PaymentMethod") == "Electronic check" and
            any("electronic check" in k for k in top_risk)):
        recs.append({
            "icon"    : "💳",
            "title"   : "Encourage Auto-Pay Enrollment",
            "detail"  : ("Electronic check users churn at ~45% — highest of all payment "
                         "methods. Offer a 5% monthly discount for switching to automatic "
                         "bank transfer or credit card."),
            "priority": "MEDIUM",
            "color"   : CLR_WARN,
        })

    # ── Rule 6: Low engagement (no streaming, no add-ons) ────────────────────
    if (user_input.get("StreamingTV") == "No" and
            user_input.get("StreamingMovies") == "No" and
            user_input.get("num_addons", 0) == 0):
        recs.append({
            "icon"    : "📺",
            "title"   : "Upsell Engagement Bundle",
            "detail"  : ("Low service usage signals disengagement and easy competitor "
                         "switching. A streaming + security bundle trial increases "
                         "perceived value and stickiness."),
            "priority": "LOW",
            "color"   : CLR_NEUT,
        })

    # ── Fallback: low-risk customer ────────────────────────────────────────────
    if not recs:
        recs.append({
            "icon"    : "✅",
            "title"   : "Low Risk — Standard Monitoring",
            "detail"  : (f"Churn probability is only {probability * 100:.1f}%. "
                         f"No immediate intervention needed. Re-assess at next "
                         f"contract renewal milestone."),
            "priority": "LOW",
            "color"   : CLR_RETAIN,
        })

    return recs[:4]   # cap at 4 recommendations


# =============================================================================
# SIDEBAR — INPUT FORM
# =============================================================================
def build_sidebar(training_stats):
    """Renders sidebar input controls. Returns (user_input dict, predict_clicked bool)."""

    st.sidebar.markdown(
        """
        <div style='text-align:center; padding:12px 0 6px 0;'>
            <span style='font-size:2.4rem;'>📡</span><br>
            <span style='font-size:1.1rem; font-weight:700; color:#2C3E50;'>
                Churn Risk Predictor
            </span><br>
            <span style='font-size:0.82rem; color:#9E9E9E;'>
                Telco Customer Analytics · Day 10
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # ── SECTION 1: Customer Profile ───────────────────────────────────────────
    st.sidebar.markdown("**👤 Customer Profile**")
    c1, c2 = st.sidebar.columns(2)
    gender  = c1.selectbox("Gender",       ["Male", "Female"],  key="s_gender")
    senior  = c2.selectbox("Senior (65+)", ["No", "Yes"],       key="s_senior")
    c3, c4 = st.sidebar.columns(2)
    partner = c3.selectbox("Partner",      ["No", "Yes"],       key="s_partner")
    depend  = c4.selectbox("Dependents",   ["No", "Yes"],       key="s_depend")
    st.sidebar.markdown("---")

    # ── SECTION 2: Service Plan ───────────────────────────────────────────────
    st.sidebar.markdown("**📋 Service & Contract**")

    tenure = st.sidebar.slider(
        "Tenure (months)", 0, 72, 12, 1,
        help=(f"Months with the company. "
              f"Avg: {training_stats['mean_tenure']:.0f} mo. "
              f"0–6 months = highest churn risk ({training_stats['early_churn_rate']:.0f}%)")
    )
    contract = st.sidebar.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        help=(f"Month-to-month churn rate: {training_stats['mtm_churn_rate']:.0f}% "
              f"vs Two Year: ~3%")
    )
    internet = st.sidebar.selectbox(
        "Internet Service",
        ["Fiber optic", "DSL", "No"],
        help=(f"Fiber Optic churn rate: {training_stats['fiber_churn_rate']:.0f}%")
    )
    c5, c6 = st.sidebar.columns(2)
    phone  = c5.selectbox("Phone",   ["Yes", "No"],         key="s_phone")
    ml_opts = (["No", "Yes"] if phone == "Yes" else ["No phone service"])
    multilines = c6.selectbox("Multi Lines", ml_opts,       key="s_multilines")
    st.sidebar.markdown("---")

    # ── SECTION 3: Add-on Services ────────────────────────────────────────────
    st.sidebar.markdown("**🛡️ Add-on Services**")

    if internet != "No":
        ca, cb = st.sidebar.columns(2)
        online_sec  = ca.selectbox("Online Security",   ["No", "Yes"], key="s_osec")
        online_back = cb.selectbox("Online Backup",     ["No", "Yes"], key="s_oback")
        cc, cd = st.sidebar.columns(2)
        dev_prot    = cc.selectbox("Device Protection", ["No", "Yes"], key="s_devp")
        tech_supp   = cd.selectbox("Tech Support",      ["No", "Yes"], key="s_tech")
        ce, cf = st.sidebar.columns(2)
        stream_tv   = ce.selectbox("Streaming TV",      ["No", "Yes"], key="s_stv")
        stream_mov  = cf.selectbox("Streaming Movies",  ["No", "Yes"], key="s_smov")
    else:
        online_sec = online_back = dev_prot = "No internet service"
        tech_supp  = stream_tv   = stream_mov = "No internet service"
        st.sidebar.caption("⚠️  Add-ons unavailable without internet.")
    st.sidebar.markdown("---")

    # ── SECTION 4: Billing ────────────────────────────────────────────────────
    st.sidebar.markdown("**💳 Billing**")

    monthly_charges = st.sidebar.slider(
        "Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5,
        help=(f"Avg: ${training_stats['mean_monthly']:.0f}/mo. "
              f"Higher charges correlate with Fiber Optic plans and higher churn.")
    )
    payment = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        help=(f"Electronic check churn rate: "
              f"{training_stats['echeck_churn_rate']:.0f}% — highest of all methods")
    )
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"], key="s_paper")
    st.sidebar.markdown("---")

    # ── Predict button ─────────────────────────────────────────────────────────
    predict_clicked = st.sidebar.button(
        "🔍  Predict Churn Risk",
        use_container_width=True,
        type="primary"
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style='text-align:center; font-size:0.73rem; color:#CCC; margin-top:8px;'>
            10-Day ML Portfolio Build<br>
            XGBoost · SHAP · Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    user_input = {
        "gender"           : gender,
        "SeniorCitizen"    : 1 if senior == "Yes" else 0,
        "Partner"          : partner,
        "Dependents"       : depend,
        "tenure"           : tenure,
        "PhoneService"     : phone,
        "MultipleLines"    : multilines,
        "InternetService"  : internet,
        "OnlineSecurity"   : online_sec,
        "OnlineBackup"     : online_back,
        "DeviceProtection" : dev_prot,
        "TechSupport"      : tech_supp,
        "StreamingTV"      : stream_tv,
        "StreamingMovies"  : stream_mov,
        "Contract"         : contract,
        "PaperlessBilling" : paperless,
        "PaymentMethod"    : payment,
        "MonthlyCharges"   : monthly_charges,
    }

    return user_input, predict_clicked


# =============================================================================
# MAIN APP
# =============================================================================
def main():

    # ── Load artifacts ─────────────────────────────────────────────────────────
    model, feature_names, metadata, scaler, training_stats, load_error = load_artifacts()

    if load_error:
        st.error(f"❌ Could not load model artifacts: {load_error}")
        st.info(
            "**Setup steps:**\n"
            "1. Run `notebooks/day10_streamlit_app.py` first "
            "(generates `models/scaler.pkl` + `models/training_stats.pkl`)\n"
            "2. Ensure `models/final_model.pkl` exists from Day 8\n"
            "3. Run: `streamlit run app.py` from the project root"
        )
        st.stop()

    explainer         = load_explainer(model)
    optimal_threshold = metadata["optimal_threshold"]
    model_name_str    = metadata["model_name"]
    roc_auc           = metadata["test_metrics"]["ROC-AUC"]

    # ── Page header ────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([7, 3])
    with col_h1:
        st.markdown(
            """
            <h1 style='margin-bottom:2px; color:#2C3E50; font-size:1.9rem;'>
                📡 Customer Churn Risk Predictor
            </h1>
            <p style='color:#9E9E9E; font-size:0.92rem; margin-top:0;'>
                Real-time churn prediction &nbsp;·&nbsp;
                Telco Customer Analytics Project &nbsp;·&nbsp; 7,043 Customers Trained
            </p>
            """,
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown(
            f"""
            <div style='background:{CLR_BG}; border-radius:10px;
                        padding:10px 16px; border-left:4px solid {CLR_NEUT};
                        margin-top:8px;'>
                <div style='font-size:0.76rem; color:#888; margin-bottom:2px;'>
                    Model
                </div>
                <div style='font-weight:700; color:{CLR_DARK}; font-size:1.05rem;'>
                    {model_name_str}
                </div>
                <div style='font-size:0.8rem; color:#666;'>
                    ROC-AUC&nbsp;<b>{roc_auc}</b>
                    &nbsp;·&nbsp;
                    Threshold&nbsp;<b>{optimal_threshold:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    user_input, predict_clicked = build_sidebar(training_stats)

    # ── Welcome state (before first prediction) ────────────────────────────────
    if not predict_clicked:
        st.markdown(
            """
            <div style='background:#F4F6F9; border-radius:14px;
                        padding:36px 48px; text-align:center; margin:24px 0;'>
                <div style='font-size:3.5rem; margin-bottom:8px;'>👈</div>
                <h3 style='color:#2C3E50; margin:0 0 8px;'>
                    Configure a customer profile in the sidebar
                </h3>
                <p style='color:#888; font-size:1rem; max-width:520px;
                          margin:0 auto;'>
                    Adjust the customer attributes on the left, then click
                    <b>🔍 Predict Churn Risk</b> to see an instant prediction
                    with SHAP explanations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # How it works
        with st.expander("ℹ️  How this app works", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            cards = [
                (CLR_NEUT,   "① Configure",
                 "Use the sidebar to set any customer's demographic, "
                 "service plan, add-ons, and billing details."),
                (CLR_WARN,   "② Predict",
                 f"The <b>{model_name_str}</b> model (ROC-AUC {roc_auc}) "
                 f"estimates the probability of churn in milliseconds."),
                (CLR_CHURN,  "③ Explain",
                 "SHAP values reveal <i>which features drove the prediction</i>, "
                 "making it interpretable for business decisions."),
            ]
            for col, (color, title, text) in zip([col_a, col_b, col_c], cards):
                with col:
                    st.markdown(
                        f"""
                        <div style='background:white; border-radius:10px;
                                    padding:18px; border-top:4px solid {color};
                                    height:100%;'>
                            <b style='font-size:1rem; color:{CLR_DARK};'>{title}</b>
                            <p style='font-size:0.87rem; color:#666;
                                      margin:6px 0 0;'>{text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        return

    # ── Run prediction ─────────────────────────────────────────────────────────
    with st.spinner("Computing churn risk..."):
        input_df    = preprocess_input(user_input, feature_names, scaler, training_stats)
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction  = int(probability >= optimal_threshold)

        # Compute SHAP values
        shap_raw = explainer.shap_values(input_df)
        if isinstance(shap_raw, list):
            shap_vals = shap_raw[1][0]   # Random Forest: class-1 array
        else:
            shap_vals = shap_raw[0]      # XGBoost: single array

        shap_dict = dict(zip(feature_names, shap_vals))

    # ── ROW 1: Gauge  |  Risk tier  |  vs-Average comparison ──────────────────
    col_g, col_t, col_v = st.columns([2.2, 1.5, 1.8])

    with col_g:
        fig_gauge, tier_color, tier_label, tier_icon = create_gauge(
            probability, optimal_threshold
        )
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

    with col_t:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Risk badge
        st.markdown(
            f"""
            <div style='background:{tier_color}1A; border:2.5px solid {tier_color};
                        border-radius:14px; padding:18px 12px; text-align:center;
                        margin-bottom:12px;'>
                <div style='font-size:2.2rem;'>{tier_icon}</div>
                <div style='font-size:1.4rem; font-weight:800; color:{tier_color};
                            letter-spacing:1.5px; margin:4px 0;'>
                    {tier_label}
                </div>
                <div style='font-size:0.82rem; color:#666;'>
                    {"Immediate action recommended"
                     if tier_label == "HIGH RISK"
                     else ("Monitor & engage" if tier_label == "MEDIUM RISK"
                           else "No action needed")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Binary prediction label
        if prediction == 1:
            pred_html = f"<b style='color:{CLR_CHURN};'>⚠️ Predicted: Will Churn</b>"
        else:
            pred_html = f"<b style='color:{CLR_RETAIN};'>✅ Predicted: Will Retain</b>"

        st.markdown(
            f"""
            <div style='background:white; border-radius:8px; padding:10px 14px;
                        border:1px solid #E8E8E8; text-align:center;
                        font-size:0.92rem;'>
                {pred_html}<br>
                <span style='font-size:0.78rem; color:#AAA;'>
                    decision threshold = {optimal_threshold:.2f}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_v:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<b style='color:{CLR_DARK}; font-size:0.95rem;'>"
            f"📊 vs Dataset Average</b>",
            unsafe_allow_html=True
        )

        tenure_val  = user_input["tenure"]
        tenure_diff = tenure_val - training_stats["mean_tenure"]
        t_color     = CLR_CHURN if tenure_val < training_stats["mean_tenure"] else CLR_RETAIN

        monthly_val  = user_input["MonthlyCharges"]
        monthly_diff = monthly_val - training_stats["mean_monthly"]
        m_color      = CLR_CHURN if monthly_val > training_stats["mean_monthly"] else CLR_RETAIN

        contract_colors = {"Month-to-month": CLR_CHURN,
                           "One year"       : CLR_WARN,
                           "Two year"       : CLR_RETAIN}
        c_color = contract_colors.get(user_input["Contract"], CLR_GREY)
        i_color = (CLR_CHURN if user_input["InternetService"] == "Fiber optic"
                   else CLR_RETAIN if user_input["InternetService"] == "No"
                   else CLR_WARN)

        cards_data = [
            ("📅 Tenure",
             f"{tenure_val} mo",
             f"{'↓' if tenure_diff < 0 else '↑'} {abs(tenure_diff):.0f} mo vs avg {training_stats['mean_tenure']:.0f} mo",
             t_color),
            ("💰 Monthly Charges",
             f"${monthly_val:.0f}",
             f"{'↑' if monthly_diff > 0 else '↓'} ${abs(monthly_diff):.0f} vs avg ${training_stats['mean_monthly']:.0f}",
             m_color),
            ("📋 Contract",        user_input["Contract"],          "",  c_color),
            ("🌐 Internet Service", user_input["InternetService"],  "",  i_color),
        ]

        for label, value, delta, color in cards_data:
            delta_html = (f"<div style='font-size:0.76rem; color:{color};'>{delta}</div>"
                          if delta else "")
            st.markdown(
                f"""
                <div style='background:white; border-radius:8px; padding:8px 12px;
                            border:1px solid #EEE; border-left:3.5px solid {color};
                            margin-bottom:7px;'>
                    <div style='font-size:0.77rem; color:#888;'>{label}</div>
                    <div style='font-weight:700; color:{CLR_DARK}; font-size:0.97rem;'>
                        {value}
                    </div>
                    {delta_html}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── ROW 2: SHAP chart  |  Recommendations ─────────────────────────────────
    col_s, col_r = st.columns([1.6, 1.0])

    with col_s:
        st.markdown(
            f"<h4 style='color:{CLR_DARK}; margin-bottom:8px; font-size:1.05rem;'>"
            f"🔍 Why this prediction? &nbsp;<span style='font-weight:400;"
            f"font-size:0.85rem; color:{CLR_GREY};'>(SHAP Explanation)</span></h4>",
            unsafe_allow_html=True
        )
        fig_shap = create_shap_chart(shap_vals, feature_names)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        with st.expander("📖  How to read this chart"):
            st.markdown(
                f"""
                - **Red bars →** push prediction toward
                  <span style='color:{CLR_CHURN}'>**churn**</span>
                - **Green bars →** push prediction toward
                  <span style='color:{CLR_RETAIN}'>**retained**</span>
                - Bar length = the strength of that feature's influence
                  on *this specific customer*
                - Values are **SHAP scores** (SHapley Additive exPlanations),
                  a game-theory-based measure of each feature's exact contribution
                  to the final probability
                """,
                unsafe_allow_html=True
            )

    with col_r:
        st.markdown(
            f"<h4 style='color:{CLR_DARK}; margin-bottom:8px; font-size:1.05rem;'>"
            f"💡 Recommended Actions</h4>",
            unsafe_allow_html=True
        )

        recs = get_recommendations(shap_dict, user_input, probability)

        for rec in recs:
            p_colors = {"HIGH": CLR_CHURN, "MEDIUM": CLR_WARN, "LOW": CLR_RETAIN}
            p_color  = p_colors.get(rec["priority"], CLR_GREY)

            st.markdown(
                f"""
                <div style='background:white; border-radius:10px; padding:13px 15px;
                            border:1px solid #EEE; border-left:4px solid {p_color};
                            margin-bottom:10px;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:5px;'>
                        <b style='color:{CLR_DARK}; font-size:0.91rem;'>
                            {rec['icon']} &nbsp;{rec['title']}
                        </b>
                        <span style='background:{p_color}1A; color:{p_color};
                                     font-size:0.7rem; font-weight:700;
                                     padding:2px 8px; border-radius:20px;
                                     margin-left:8px; white-space:nowrap;'>
                            {rec['priority']}
                        </span>
                    </div>
                    <span style='font-size:0.81rem; color:#666; line-height:1.5;'>
                        {rec['detail']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── ROW 3: Full profile (expandable) ──────────────────────────────────────
    with st.expander("📋  Full customer profile & feature breakdown"):
        col_l, col_r2 = st.columns(2)

        with col_l:
            st.markdown("**Raw input values**")
            profile_data = {
                "Feature" : list(user_input.keys()),
                "Value"   : [str(v) for v in user_input.values()],
            }
            st.dataframe(
                pd.DataFrame(profile_data),
                use_container_width=True,
                hide_index=True
            )

        with col_r2:
            st.markdown("**Top 15 features by |SHAP| impact**")
            shap_df = pd.DataFrame({
                "Feature"      : feature_names,
                "Model Value"  : input_df.iloc[0].values.round(4),
                "SHAP Value"   : [round(v, 4) for v in shap_vals],
            })
            shap_df["_abs"] = shap_df["SHAP Value"].abs()
            shap_df = (shap_df
                       .sort_values("_abs", ascending=False)
                       .drop("_abs", axis=1)
                       .head(15)
                       .reset_index(drop=True))
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='text-align:center; padding:14px; color:#CCC; font-size:0.78rem;
                    border-top:1px solid #EEE; margin-top:18px;'>
            Customer Churn Analytics & Prediction Project &nbsp;·&nbsp;
            10-Day ML Portfolio Build &nbsp;·&nbsp;
            Model: {model_name_str} (ROC-AUC: {roc_auc}) &nbsp;·&nbsp;
            Built with Streamlit + SHAP + XGBoost
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
if __name__ == "__main__":
    main()
