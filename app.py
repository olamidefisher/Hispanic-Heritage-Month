import io
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error, r2_score

# ReportLab Dependencies
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hispanic Heritage Historical Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & METADATA DICTIONARIES
# -----------------------------------------------------------------------------
COUNTRY_MAP = {
    "Mexico": "MX",
    "Brazil": "BR",
    "Argentina": "AR",
    "Colombia": "CO",
    "Chile": "CL",
    "Peru": "PE",
    "Venezuela": "VE",
    "Ecuador": "EC",
    "Uruguay": "UY"
}

INDICATOR_MAP = {
    "Life Expectancy": {
        "code": "SP.DYN.LE00.IN",
        "units": "Years",
        "def": "Life expectancy at birth indicates the number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life."
    },
    "Unemployment Rate": {
        "code": "SL.UEM.TOTL.NE.ZS",
        "units": "% of total labor force",
        "def": "Unemployment, total (% of total labor force) based on national estimates."
    },
    "GDP per Capita": {
        "code": "NY.GDP.PCAP.CD",
        "units": "Current USD ($)",
        "def": "Gross domestic product divided by midyear population. Note: Represents overall economic output per person, NOT average household income or personal wealth."
    },
    "Crude Birth Rate": {
        "code": "SP.DYN.CBRT.IN",
        "units": "Per 1,000 people",
        "def": "Crude birth rate indicates the number of live births occurring during the year per 1,000 midyear population. Note: This is a rate, NOT total birth count."
    },
    "Net Migration": {
        "code": "SM.POP.NETM",
        "units": "Persons (5-year total / annual estimates)",
        "def": "Net migration is the net total number of migrants (immigrants minus emigrants) over a five-year period or annual estimates. Note: NOT gross emigration count."
    },
    "Intentional Homicide Rate": {
        "code": "VC.IHR.PSRC.P5",
        "units": "Per 100,000 people",
        "def": "Intentional homicides are estimates of unlawful homicides intentionally inflicted per 100,000 population."
    },
    "Consumer Price Inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "units": "Annual %",
        "def": "Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services."
    },
    "Central Government Debt": {
        "code": "GC.DOD.TOTL.GD.ZS",
        "units": "% of GDP",
        "def": "Gross central government debt as a percentage of GDP."
    },
    "Total Reserves (inc. Gold)": {
        "code": "FI.RES.TOTL.CD",
        "units": "Current USD ($)",
        "def": "Total reserves comprise holdings of monetary gold, special drawing rights, reserves of IMF members, and foreign exchange under the control of monetary authorities."
    },
    "External Balance on Goods/Services": {
        "code": "NE.RSB.GNFS.ZS",
        "units": "% of GDP",
        "def": "External balance on goods and services (formerly resource balance) equals exports of goods and services minus imports of goods and services as a share of GDP."
    }
}

# -----------------------------------------------------------------------------
# 3. WORLD BANK API FETCHING WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank_data(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 1000}
    try:
        response = requests.get(url, params=params, timeout=12)
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code} from World Bank API."
        
        json_data = response.json()
        if not isinstance(json_data, list) or len(json_data) < 2 or json_data[1] is None:
            return None, "No record payload returned by World Bank API for this query."
        
        records = json_data[1]
        parsed_rows = []
        for r in records:
            yr = int(r["date"])
            val = r["value"]
            if val is not None:
                parsed_rows.append({"Year": yr, "Value": float(val)})
                
        if not parsed_rows:
            return None, "World Bank returned records, but all annual values were null/missing."
            
        df = pd.DataFrame(parsed_rows).sort_values("Year").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, f"Network or parsing exception: {str(e)}"

# -----------------------------------------------------------------------------
# 4. MATH & POLYNOMIAL MODELING HELPERS
# -----------------------------------------------------------------------------
def fit_polynomial_model(df: pd.DataFrame, degree: int, base_year: int):
    t_vals = df["Year"].values - base_year
    y_vals = df["Value"].values
    
    if len(np.unique(t_vals)) < degree + 1:
        return None, f"Insufficient distinct years ({len(np.unique(t_vals))}) to fit a degree {degree} polynomial (requires at least {degree + 1})."
        
    coeffs = np.polyfit(t_vals, y_vals, degree)
    poly_func = np.poly1d(coeffs)
    
    y_pred = poly_func(t_vals)
    r2 = r2_score(y_vals, y_pred)
    mae = mean_absolute_error(y_vals, y_pred)
    residuals = y_vals - y_pred
    
    terms = []
    deg_len = len(coeffs)
    for i, c in enumerate(coeffs):
        p = deg_len - 1 - i
        if p == 0:
            terms.append(f"{c:+.4e}")
        elif p == 1:
            terms.append(f"{c:+.4e}·t")
        else:
            terms.append(f"{c:+.4e}·t^{p}")
    eq_str = "f(t) = " + " ".join(terms)
    
    model_dict = {
        "coeffs": coeffs,
        "poly_func": poly_func,
        "degree": degree,
        "base_year": base_year,
        "r2": r2,
        "mae": mae,
        "residuals": residuals,
        "eq_str": eq_str,
        "obs_count": len(df),
        "min_year": int(df["Year"].min()),
        "max_year": int(df["Year"].max()),
        "mean_val": float(np.mean(y_vals))
    }
    return model_dict, None

def eval_poly_single(model_dict, target_year: int):
    t = target_year - model_dict["base_year"]
    pred = float(model_dict["poly_func"](t))
    is_in_range = (model_dict["min_year"] <= target_year <= model_dict["max_year"])
    return pred, is_in_range

# -----------------------------------------------------------------------------
# 5. DERIVATIVE & CALCULUS ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def get_symbolic_derivatives(poly_func):
    d1 = np.polyder(poly_func, 1)
    d2 = np.polyder(poly_func, 2)
    return d1, d2

def format_poly_latex(poly_func, var_str="t"):
    coeffs = poly_func.coefficients
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = deg - i
        if abs(c) < 1e-12:
            continue
        c_str = f"{c:.3e}" if (abs(c) < 0.01 or abs(c) > 1000) else f"{c:.3f}"
        if p == 0:
            terms.append(f"{c_str}")
        elif p == 1:
            terms.append(f"{c_str}{var_str}")
        else:
            terms.append(f"{c_str}{var_str}^{{{p}}}")
    if not terms:
        return "0"
    res = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            res += " - " + t[1:]
        else:
            res += " + " + t
    return res

def analyze_calculus_features(model_dict, search_min_yr, search_max_yr):
    poly = model_dict["poly_func"]
    d1, d2 = get_symbolic_derivatives(poly)
    base_yr = model_dict["base_year"]
    
    t_min = search_min_yr - base_yr
    t_max = search_max_yr - base_yr
    
    d1_roots = np.roots(d1)
    real_d1_roots = [r.real for r in d1_roots if np.isreal(r) and t_min <= r.real <= t_max]
    real_d1_roots = sorted(list(set(np.round(real_d1_roots, 4))))
    
    critical_points = []
    for r_t in real_d1_roots:
        yr = r_t + base_yr
        val = poly(r_t)
        d2_val = d2(r_t)
        if d2_val > 1e-6:
            classification = "Local Minimum (d'' > 0)"
        elif d2_val < -1e-6:
            classification = "Local Maximum (d'' < 0)"
        else:
            classification = "Stationary Point"
        critical_points.append({
            "Year": round(yr, 2),
            "t": round(r_t, 2),
            "Value": round(val, 4),
            "Class": classification
        })
        
    d2_roots = np.roots(d2)
    real_d2_roots = [r.real for r in d2_roots if np.isreal(r) and t_min <= r.real <= t_max]
    real_d2_roots = sorted(list(set(np.round(real_d2_roots, 4))))
    
    inflection_points = []
    for r_t in real_d2_roots:
        eps = 0.05
        d2_left = d2(r_t - eps)
        d2_right = d2(r_t + eps)
        if (d2_left * d2_right) < 0:
            yr = r_t + base_yr
            val = poly(r_t)
            inflection_points.append({
                "Year": round(yr, 2),
                "t": round(r_t, 2),
                "Value": round(val, 4),
                "Confirmed": "Yes (Sign Change)"
            })
            
    t_dense = np.linspace(t_min, t_max, 500)
    y_dense = poly(t_dense)
    abs_min_idx = np.argmin(y_dense)
    abs_max_idx = np.argmax(y_dense)
    
    abs_extrema = {
        "Abs_Min": {"Year": round(t_dense[abs_min_idx] + base_yr, 2), "Value": round(y_dense[abs_min_idx], 4)},
        "Abs_Max": {"Year": round(t_dense[abs_max_idx] + base_yr, 2), "Value": round(y_dense[abs_max_idx], 4)}
    }
    
    return critical_points, inflection_points, abs_extrema

# -----------------------------------------------------------------------------
# 6. EXPORT GENERATORS (EXCEL & PDF)
# -----------------------------------------------------------------------------
def generate_excel_export(df_working, model_dict, batch_df, country, indicator, units, is_user_modified):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        summary_data = [
            ["Country", country],
            ["Indicator", indicator],
            ["Units", units],
            ["Scenario Status", "User-Modified Scenario" if is_user_modified else "Original Source Data"],
            ["Polynomial Degree", model_dict["degree"]],
            ["Base Year", model_dict["base_year"]],
            ["Equation", model_dict["eq_str"]],
            ["In-Sample R²", model_dict["r2"]],
            ["In-Sample MAE", model_dict["mae"]],
            ["Observation Count", model_dict["obs_count"]],
            ["Historical Range", f"{model_dict['min_year']} - {model_dict['max_year']}"],
            ["Mean Value", model_dict["mean_val"]]
        ]
        df_sum = pd.DataFrame(summary_data, columns=["Parameter", "Value"])
        df_sum.to_excel(writer, sheet_name="Model Summary", index=False)
        df_working.to_excel(writer, sheet_name="Working Data", index=False)
        
        if batch_df is not None and not batch_df.empty:
            batch_df.to_excel(writer, sheet_name="Model Estimates", index=False)
            
    return output.getvalue()

def generate_pdf_export(df_working, model_dict, batch_df, fig_plot, country, indicator, units, is_user_modified):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1E3A8A'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12)
    
    story.append(Paragraph("Hispanic Heritage Data Explorer: Research Report", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Country:</b> {country} | <b>Indicator:</b> {indicator} ({units})", body_style))
    story.append(Paragraph(f"<b>Data Status:</b> {'User-Modified Scenario' if is_user_modified else 'Authentic World Bank Source Data'}", body_style))
    story.append(Paragraph(f"<b>Generated On:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 10))
    
    metrics_data = [
        ["Polynomial Degree", str(model_dict["degree"]), "Base Year (t=0)", str(model_dict["base_year"])],
        ["In-Sample R²", f"{model_dict['r2']:.4f}", "In-Sample MAE", f"{model_dict['mae']:.4f}"],
        ["Observation Count", str(model_dict["obs_count"]), "Historical Mean", f"{model_dict['mean_val']:.4f}"]
    ]
    t_metrics = Table(metrics_data, colWidths=[120, 120, 120, 120])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Equation:</b> {model_dict['eq_str']}", body_style))
    story.append(Spacer(1, 10))
    
    img_buf = io.BytesIO()
    fig_plot.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=480, height=240))
    
    doc.build(story)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# 7. MAIN STREAMLIT APPLICATION
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">Hispanic Heritage Historical Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">STEM Data Science & Calculus Lab</div>', unsafe_allow_html=True)

    st.sidebar.header("🕹️ Controls")
    pathway = st.sidebar.selectbox("Mathematics Pathway", ["Calculus", "Precalculus", "Honors Algebra 2"])
    
    country_name = st.sidebar.selectbox("Primary Country", list(COUNTRY_MAP.keys()), index=0)
    country_code = COUNTRY_MAP[country_name]
    
    indicator_name = st.sidebar.selectbox("Indicator", list(INDICATOR_MAP.keys()), index=0)
    indicator_meta = INDICATOR_MAP[indicator_name]
    
    comp_country_options = ["None"] + [c for c in COUNTRY_MAP.keys() if c != country_name]
    comp_country_name = st.sidebar.selectbox("Comparison Country", comp_country_options, index=0)
    
    st.sidebar.markdown("---")
    poly_degree = st.sidebar.slider("Polynomial Degree (d)", min_value=3, max_value=8, value=3)
    graph_increment = st.sidebar.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1)
    forward_horizon = st.sidebar.slider("Forward Extrapolation (Years)", min_value=0, max_value=50, value=15)
    holdout_years_count = st.sidebar.slider("Holdout Validation (Years)", min_value=0, max_value=15, value=5)

    if "edited_df" not in st.session_state:
        st.session_state.edited_df = None
    if "data_key" not in st.session_state:
        st.session_state.data_key = ""

    curr_yr = datetime.datetime.now().year
    start_yr = curr_yr - 70
    end_yr = curr_yr - 1

    current_data_key = f"{country_code}_{indicator_meta['code']}_{start_yr}_{end_yr}"
    
    if st.session_state.data_key != current_data_key:
        raw_df, err = fetch_world_bank_data(country_code, indicator_meta["code"], start_yr, end_yr)
        if err:
            st.error(f"Data Notice: {err}")
            st.session_state.edited_df = pd.DataFrame(columns=["Year", "Value"])
        else:
            st.session_state.edited_df = raw_df.copy()
        st.session_state.data_key = current_data_key
        st.session_state.is_modified = False

    df_comp = None
    if comp_country_name != "None":
        comp_code = COUNTRY_MAP[comp_country_name]
        df_comp, _ = fetch_world_bank_data(comp_code, indicator_meta["code"], start_yr, end_yr)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data & Regression",
        "🧮 Function Analysis",
        "🎯 Predictions & Validation",
        "⚡ Contextual Simulation",
        "📑 Exports"
    ])

    with tab1:
        st.subheader(f"Historical Indicator Data: {country_name}")
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### Data Editing Table")
            if st.button("🔄 Reset to Source Data"):
                raw_df, _ = fetch_world_bank_data(country_code, indicator_meta["code"], start_yr, end_yr)
                if raw_df is not None:
                    st.session_state.edited_df = raw_df.copy()
                    st.session_state.is_modified = False
                    st.rerun()

            edited_input = st.data_editor(st.session_state.edited_df, num_rows="dynamic", use_container_width=True, key="table_editor")
            if not edited_input.equals(st.session_state.edited_df):
                st.session_state.edited_df = edited_input.copy()
                st.session_state.is_modified = True

            if st.session_state.get("is_modified", False):
                st.warning("⚠️ User-Modified Scenario")
            else:
                st.success("✅ Authentic World Bank Source Data")

        df_active = st.session_state.edited_df.dropna().sort_values("Year").reset_index(drop=True)
        
        with col_right:
            st.markdown("### Centered Polynomial Regression Fit")
            if len(df_active) < poly_degree + 1:
                st.error(f"Cannot fit degree {poly_degree} polynomial. Requires at least {poly_degree + 1} valid years.")
            else:
                if holdout_years_count > 0 and len(df_active) > (poly_degree + holdout_years_count + 2):
                    df_train = df_active.iloc[:-holdout_years_count].copy()
                    df_holdout = df_active.iloc[-holdout_years_count:].copy()
                else:
                    df_train = df_active.copy()
                    df_holdout = pd.DataFrame()

                base_year = int(df_train["Year"].min())
                model, err = fit_polynomial_model(df_train, poly_degree, base_year)
                
                if err:
                    st.error(err)
                else:
                    st.code(model["eq_str"], language="text")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("In-Sample R²", f"{model['r2']:.4f}")
                    m2.metric("In-Sample MAE", f"{model['mae']:.4f}")
                    m3.metric("Observed Years (n)", f"{model['obs_count']}")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    df_sampled = df_train.iloc[::graph_increment]
                    ax.scatter(df_sampled["Year"], df_sampled["Value"], color="#1D4ED8", label=f"Observed ({country_name})", s=30)
                    
                    if not df_holdout.empty:
                        ax.scatter(df_holdout["Year"], df_holdout["Value"], color="#D97706", label="Holdout Actuals", s=35)

                    full_min_yr = int(df_active["Year"].min())
                    full_max_yr = int(df_active["Year"].max()) + forward_horizon
                    t_smooth = np.linspace(full_min_yr - base_year, full_max_yr - base_year, 300)
                    yr_smooth = t_smooth + base_year
                    y_smooth = model["poly_func"](t_smooth)
                    
                    mask_in = yr_smooth <= int(df_train["Year"].max())
                    ax.plot(yr_smooth[mask_in], y_smooth[mask_in], color="#1E3A8A", label="Fitted Polynomial")
                    
                    mask_out = yr_smooth > int(df_train["Year"].max())
                    if np.any(mask_out):
                        ax.plot(yr_smooth[mask_out], y_smooth[mask_out], color="#DC2626", linestyle="--", label="Extrapolated Curve")

                    if df_comp is not None and not df_comp.empty:
                        ax.plot(df_comp["Year"], df_comp["Value"], color="#059669", linestyle=":", label=f"Comparison ({comp_country_name})")

                    ax.set_title(f"{indicator_name} (Degree {poly_degree})")
                    ax.set_xlabel("Year")
                    ax.set_ylabel(indicator_meta["units"])
                    ax.grid(True, alpha=0.4)
                    ax.legend(fontsize=8)
                    st.pyplot(fig)
                    
                    st.session_state.active_model = model
                    st.session_state.df_train = df_train
                    st.session_state.df_holdout = df_holdout

    with tab2:
        st.subheader(f"Function Analysis — {pathway} Mode")
        if "active_model" in st.session_state:
            model = st.session_state.active_model
            poly = model["poly_func"]
            d1, d2 = get_symbolic_derivatives(poly)
            
            st.latex(rf"f(t) = {format_poly_latex(poly, 't')}")
            if pathway in ["Calculus", "Precalculus"]:
                st.latex(rf"f'(t) = {format_poly_latex(d1, 't')}")
            if pathway == "Calculus":
                st.latex(rf"f''(t) = {format_poly_latex(d2, 't')}")

            eval_yr = st.slider("Select Year", min_value=model["min_year"], max_value=model["max_year"]+20, value=model["base_year"]+10)
            t0 = eval_yr - model["base_year"]
            st.write(f"**Instantaneous Rate of Change at {eval_yr}:** `{d1(t0):.4f}` {indicator_meta['units']}/year")

    with tab3:
        st.subheader("🎯 Predictions & Holdout Validation")
        if "active_model" in st.
