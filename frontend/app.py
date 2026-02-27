import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="EAM Control Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib
import numpy as np
import os
import sys
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ─────────────────────────────────────────────
# DIAGNOSTIC BOOTSTRAP
# ─────────────────────────────────────────────
try:
    # Load environment variables
    load_dotenv()

    # Streamlit Cloud Secret Mapping
    if hasattr(st, "secrets"):
        try:
            for key, value in st.secrets.items():
                if key not in os.environ:
                    os.environ[key] = str(value)
        except Exception:
            pass # Ignore if secrets are not available locally

    # Add backend, ml, and data to path
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    
    for folder in ['backend', 'ml', 'data']:
        folder_path = os.path.join(ROOT_DIR, folder)
        if folder_path not in sys.path:
            sys.path.append(folder_path)

    from database import SessionLocal, engine, Base
    import models
    from models import WorkOrder
    from predict import predict_asset_risk, predict_fleet_risk
    from chatbot_service import generate_response, get_maintenance_recommendation, get_signal_insights, get_executive_briefing, get_sensor_summary

    # INITIALIZE DATABASE
    Base.metadata.create_all(bind=engine)

except Exception as e:
    st.error("🚨 **Critical Startup Error Detected**")
    st.exception(e)
    st.write("---")
    st.write("Current sys.path:", sys.path)
    st.stop()

# ─────────────────────────────────────────────
# CUSTOM CSS - PROFESSIONAL LIGHT MODE
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .st-emotion-cache-16txtl3 {
        background-color: #f8f9fa !important;
        color: #1f2937 !important;
    }
    
    /* Ensure all headers and text are visible */
    h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
        color: #1f2937 !important;
    }
    
    /* Input and Select box Fixes */
    .stSelectbox [data-testid="stMarkdownContainer"], .stTextInput input {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Navigation Bar adjustment */
    .st-emotion-cache-16txtl3 {
        padding-top: 0rem;
    }

    /* Card Styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 5px;
    }

    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
    }

    .delta-pos { background-color: #dcfce7; color: #166534; }
    .delta-neg { background-color: #fee2e2; color: #991b1b; }
    .delta-neutral { background-color: #f3f4f6; color: #374151; }
    
    /* Active Maintenance List */
    .maintenance-item {
        padding: 12px;
        border-bottom: 1px solid #f3f4f6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .maintenance-item:last-child { border-bottom: none; }
    
    .status-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .priority-critical { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .priority-high { background-color: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load ML models"""
    try:
        # Standardize paths for both local and cloud deployment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), 'ml', 'models')
        
        cost_model = joblib.load(os.path.join(models_dir, 'cost_predictor.pkl'))
        return None, None, None, cost_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

comp_model, comp_scaler, comp_features, cost_model = load_models()

@st.cache_data(show_spinner=False)
def cached_get_maintenance_recommendation(data):
    return get_maintenance_recommendation(data)

def get_db_data():
    """Fetch all necessary data"""
    # Join assets with industries to get actual industry name
    query_assets = """
    SELECT a.*, i.name as industry_name 
    FROM assets a
    JOIN industries i ON a.industry_id = i.id
    """
    df_assets = pd.read_sql(query_assets, engine)
    
    query_wo = "SELECT * FROM work_orders"
    df_wo = pd.read_sql(query_wo, engine)
    
    query_costs = "SELECT * FROM cost_records"
    df_costs = pd.read_sql(query_costs, engine)
    
    return df_assets, df_wo, df_costs

df_assets, df_wo, df_costs = get_db_data()


# ─────────────────────────────────────────────
# EMAIL DISPATCHER (SENDGRID)
# ─────────────────────────────────────────────

def send_email_via_sendgrid(asset_name, tech_recommendation, risk_level, asset_id, rul, liability):
    """Sends a professional intervention alert via SendGrid."""
    sg_key = os.getenv("SENDGRID_API_KEY")
    # Priority: Env Var > Hardcoded Placeholder
    from_email = os.getenv("SENDGRID_FROM_EMAIL") 
    to_email = os.getenv("SENDGRID_TO_EMAIL")
    
    if not sg_key:
        return False, "SendGrid API Key missing in .env"
        
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=f"🚨 AI ALERT: {risk_level} Risk Detected for {asset_name}",
        html_content=f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; border: 1px solid #e5e7eb; padding: 30px; border-radius: 15px; background-color: #ffffff; color: #1f2937;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #ef4444; margin: 0; font-size: 24px;">⚠️ HIGH RISK INTERVENTION REQUIRED</h1>
            </div>
            <p style="font-size: 16px;">Predictive Engine has detected a <strong>{risk_level}</strong> risk signature for <strong>{asset_name}</strong> (ID: {asset_id}).</p>
            
            <div style="background-color: #f3f4f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin-top: 0; color: #111827;">AI Diagnostics Summary:</h3>
                <ul style="list-style: none; padding: 0;">
                    <li style="margin-bottom: 8px;">🕒 <strong>Remaining Useful Life:</strong> {rul} Days</li>
                    <li style="margin-bottom: 8px;">💸 <strong>Predicted Liability:</strong> ${liability:,.2f}</li>
                    <li style="margin-bottom: 8px;">🛠️ <strong>Dispatch Strategy:</strong> {tech_recommendation}</li>
                </ul>
            </div>
            
            <p style="font-size: 14px; line-height: 1.5;">This alert was generated because of abnormal sensor deviations. Immediate intervention is advised to prevent catastrophic component failure and unplanned downtime.</p>
            
            <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0;"/>
            <p style="font-size: 12px; color: #6b7280; text-align: center;">EAM Platform | AI-Driven Asset Intelligence</p>
        </div>
        """
    )
    try:
        sg = SendGridAPIClient(sg_key)
        response = sg.send(message)
        if response.status_code in [200, 201, 202]:
            return True, "Email sent successfully"
        else:
            return False, f"SendGrid error: {response.status_code}"
    except Exception as e:
        print(f"DEBUG: SendGrid Exception: {str(e)}")
        return False, str(e)

# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def format_currency_smart(value):
    """Formats currency to $M or $K based on scale."""
    if value == 0:
        return "$0"
    if value >= 1000000:
        return f"${value/1000000:.1f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:,.0f}"

def metric_card(label, value, delta, delta_type, tooltip=""):
    tooltip_html = f' title="{tooltip}" style="cursor:help;"' if tooltip else ""
    return f"""
    <div class="metric-card"{tooltip_html}>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <span class="metric-delta {delta_type}">{delta}</span>
    </div>
    """

@st.cache_data(show_spinner="Analyzing signals...", ttl=3600)
def cached_get_signal_insights(signals_json, asset_metadata):
    return get_signal_insights(signals_json, asset_metadata)

@st.cache_data(show_spinner="AI is interpreting sensor telemetry...", ttl=300)
def cached_get_sensor_summary(sensors_data, asset_type):
    return get_sensor_summary(sensors_data, asset_type)

@st.cache_data(show_spinner="⚡ AI performing lightning-fast fleet analysis...", ttl=600)
def get_fleet_predictions_v3():
    """
    Fetch fleet-wide predictions using the optimized bulk engine.
    Uses st.cache_data for cross-session efficiency.
    """
    # Bulk inference for all assets
    all_preds = predict_fleet_risk()
    
    # Load cost model for financial projection
    _, _, _, local_cost_model = load_models()
    
    now = pd.Timestamp.now()
    for p in all_preds:
        try:
            purchase_date = pd.to_datetime(p['purchase_date'])
            age_days = (now - purchase_date).days
            cost_input = pd.DataFrame([{
                'priority': 'critical' if p['risk_level'] == 'Critical' else 'high',
                'asset_type': p['asset_type'],
                'industry': p['industry_name'],
                'asset_age_days': max(0, age_days),
                'description_len': 50
            }])
            p['predicted_cost'] = float(np.expm1(local_cost_model.predict(cost_input)[0]))
        except:
            p['predicted_cost'] = 0
            
    return all_preds

def batch_predict_assets(filtered_assets):
    """
    Optimized version of batch_predict_assets.
    Retrieves the cached fleet risks and filters for the requested assets.
    """
    all_preds = get_fleet_predictions_v3()
    
    # Map for fast lookup by asset ID
    preds_dict = {p['id']: p for p in all_preds}
    
    result = []
    for idx, asset in filtered_assets.iterrows():
        if asset['id'] in preds_dict:
            result.append(preds_dict[asset['id']])
            
    return result

# ── STRATEGY STATE INITIALIZATION ──
RAW_STRATEGIES = ["All", "Predictive", "Preventive", "Ad-hoc / On-demand"]

# Calculate counts for labels (used only in Asset Monitor)
all_fleet_preds = get_fleet_predictions_v3()
if not all_fleet_preds:
    # Handle empty state for fresh deployment
    pred_df = pd.DataFrame(columns=['id', 'risk_level', 'asset_type', 'industry_name'])
else:
    pred_df = pd.DataFrame(all_fleet_preds)

count_holistic = len(df_assets)

# Safe column access using .get or checking empty
if not pred_df.empty:
    count_predictive = len(pred_df[pred_df['risk_level'].isin(['Critical', 'High Risk'])])
    count_preventive = len(pred_df[pred_df['risk_level'] == 'Warning'])
else:
    count_predictive = 0
    count_preventive = 0

# Ad-hoc logic matching refined filter
if not df_wo.empty and all(col in df_wo.columns for col in ['status', 'title', 'priority', 'asset_id']):
    active_wo_mask = (df_wo['status'] == 'in_progress')
    adhoc_keywords = ["Emergency"]
    keyword_mask = df_wo['title'].str.contains('|'.join(adhoc_keywords), case=False, na=False)
    priority_mask = df_wo['priority'].isin(['critical', 'high'])
    count_adhoc = df_wo[active_wo_mask & keyword_mask & priority_mask]['asset_id'].nunique()
else:
    count_adhoc = 0

STRATEGY_MAP_WITH_COUNTS = {
    "All": f"All ({count_holistic})",
    "Predictive": f"Predictive ({count_predictive})",
    "Preventive": f"Preventive ({count_preventive})",
    "Ad-hoc / On-demand": f"Ad-hoc / On-demand ({count_adhoc})"
}
STRATEGY_OPTIONS_WITH_COUNTS = list(STRATEGY_MAP_WITH_COUNTS.values())
STRATEGY_REVERSE_MAP_WITH_COUNTS = {v: k for k, v in STRATEGY_MAP_WITH_COUNTS.items()}

if 'selected_strategy_key' not in st.session_state:
    st.session_state.selected_strategy_key = "All"

if 'acknowledged_alerts' not in st.session_state:
    st.session_state.acknowledged_alerts = set()

# ── SHARED HELPERS ──

# ─────────────────────────────────────────────
# NAVIGATION (TOP BAR)
# ─────────────────────────────────────────────

options = ["Executive Overview", "Asset Monitor", "Cost Prediction"]

selected = option_menu(
    menu_title=None,
    options=options,
    icons=["bar-chart-fill", "activity", "cpu"],
    default_index=0,
    orientation="horizontal",
    key="main_navigation_menu",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff", "border-bottom": "1px solid #e5e7eb"},
        "icon": {"color": "#6b7280", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#f3f4f6", "color": "#374151"},
        "nav-link-selected": {"background-color": "#2563eb", "color": "#ffffff"},
    }
)


@st.dialog("🚨 Top 10 Priority Actions", width="large")
def show_priority_list(active_wo, predicted_critical_df, filtered_assets):
    st.write("These work orders require immediate attention based on AI risk analysis.")
    if active_wo.empty:
        st.success("✅ No active work orders.")
        return
        
    wo_with_assets = active_wo.merge(
        filtered_assets[['id', 'name', 'asset_type']],
        left_on='asset_id', right_on='id', how='left', suffixes=('', '_asset')
    )
    if not predicted_critical_df.empty:
        wo_enriched = wo_with_assets.merge(
            predicted_critical_df[['id', 'risk_level', 'risk_score', 'rul', 'predicted_cost']],
            left_on='asset_id', right_on='id', how='left', suffixes=('', '_pred')
        )
    else:
        wo_enriched = wo_with_assets.copy()
        for _col in ['risk_level', 'risk_score', 'rul', 'predicted_cost']:
            wo_enriched[_col] = None

    risk_order = {'Critical': 0, 'High Risk': 1, 'Warning': 2}
    wo_enriched['_sort_key'] = wo_enriched['risk_level'].map(risk_order).fillna(99)
    wo_enriched = wo_enriched.sort_values('_sort_key').head(10)
    
    for _, wrow in wo_enriched.iterrows():
        wrisk        = wrow.get('risk_level', None)
        wrisk_valid  = wrisk and pd.notna(wrisk)
        wasset       = wrow.get('name', 'Unknown Asset')
        wtitle       = str(wrow.get('title', wrow.get('description', 'Work Order')))[:50]
        wpriority    = str(wrow.get('priority', 'N/A')).title()
        wtype        = wrow.get('asset_type', '')
        wscore       = wrow.get('risk_score', None)
        wrul         = wrow.get('rul', None)
        wcost        = wrow.get('predicted_cost', None)

        if wrisk == 'Critical':   wbadge = "🔴 CRITICAL"
        elif wrisk == 'High Risk': wbadge = "🟠 HIGH RISK"
        elif wrisk == 'Warning':   wbadge = "🟡 WARNING"
        else:                      wbadge = "⚪ Healthy"

        # Auto-expand
        with st.expander(f"{wbadge}  |  {wtitle}  —  {wasset}", expanded=wrisk in ['Critical', 'High Risk']):
            _a, _b, _c = st.columns(3)
            _a.metric("Asset", wasset)
            _b.metric("Type", wtype)
            _c.metric("Priority", wpriority)
            if wrisk_valid:
                st.divider()
                _d, _e, _f = st.columns(3)
                _d.metric("🤖 AI Total Risk", f"{wscore:.0f}%" if wscore and pd.notna(wscore) else "N/A")
                _e.metric("⏱️ Predicted RUL", f"{int(wrul)} days" if wrul and pd.notna(wrul) else "N/A")
                _f.metric("💰 Est. Liability", f"${wcost:,.0f}" if wcost and pd.notna(wcost) else "N/A")
                if wrisk == 'Critical':
                    st.error("⚠️ Imminent failure — escalate immediately.")
                    if st.button("Dispatch Emergency Crew", key=f"dispatch_action_{wrow['id']}"):
                        st.success("Crew Dispatched!")
                elif wrisk == 'High Risk':
                    st.warning("⚠️ High failure probability — schedule urgent inspection.")
                    if st.button("Schedule Inspection", key=f"inspect_action_{wrow['id']}"):
                        st.success("Inspection Scheduled.")
            else:
                st.info("ℹ️ No AI risk data available yet.")

@st.dialog("🚨 Action Center: Critical Work Orders", width="large")
def show_action_center(active_wo, predicted_critical_df, filtered_assets):
    st.write("These work orders require immediate attention based on AI risk analysis.")
    if active_wo.empty:
        st.success("✅ No active work orders.")
        return
        
    wo_with_assets = active_wo.merge(
        filtered_assets[['id', 'name', 'asset_type']],
        left_on='asset_id', right_on='id', how='left', suffixes=('', '_asset')
    )
    if not predicted_critical_df.empty:
        wo_enriched = wo_with_assets.merge(
            predicted_critical_df[['id', 'risk_level', 'risk_score', 'rul', 'predicted_cost']],
            left_on='asset_id', right_on='id', how='left', suffixes=('', '_pred')
        )
    else:
        wo_enriched = wo_with_assets.copy()
        for _col in ['risk_level', 'risk_score', 'rul', 'predicted_cost']:
            wo_enriched[_col] = None

    risk_order = {'Critical': 0, 'High Risk': 1, 'Warning': 2}
    wo_enriched['_sort_key'] = wo_enriched['risk_level'].map(risk_order).fillna(99)
    wo_enriched = wo_enriched.sort_values('_sort_key')
    
    for _, wrow in wo_enriched.iterrows():
        wrisk        = wrow.get('risk_level', None)
        wrisk_valid  = wrisk and pd.notna(wrisk)
        wasset       = wrow.get('name', 'Unknown Asset')
        wtitle       = str(wrow.get('title', wrow.get('description', 'Work Order')))[:50]
        wpriority    = str(wrow.get('priority', 'N/A')).title()
        wtype        = wrow.get('asset_type', '')
        wscore       = wrow.get('risk_score', None)
        wrul         = wrow.get('rul', None)
        wcost        = wrow.get('predicted_cost', None)

        if wrisk == 'Critical':   wbadge = "🔴 CRITICAL"
        elif wrisk == 'High Risk': wbadge = "🟠 HIGH RISK"
        elif wrisk == 'Warning':   wbadge = "🟡 WARNING"
        else:                      wbadge = "⚪ Healthy"

        # Auto-expand criticals
        with st.expander(f"{wbadge}  |  {wtitle}  —  {wasset}", expanded=(wrisk == 'Critical')):
            _a, _b, _c = st.columns(3)
            _a.metric("Asset", wasset)
            _b.metric("Type", wtype)
            _c.metric("Priority", wpriority)
            if wrisk_valid:
                st.divider()
                _d, _e, _f = st.columns(3)
                _d.metric("🤖 AI Total Risk", f"{wscore:.0f}%" if wscore and pd.notna(wscore) else "N/A")
                _e.metric("⏱️ Predicted RUL", f"{int(wrul)} days" if wrul and pd.notna(wrul) else "N/A")
                _f.metric("💰 Est. Liability", f"${wcost:,.0f}" if wcost and pd.notna(wcost) else "N/A")
                if wrisk == 'Critical':
                    st.error("⚠️ Imminent failure — escalate immediately.")
                    if st.button("Dispatch Emergency Crew", key=f"dispatch_action_{wrow['id']}"):
                        st.success("Crew Dispatched!")
                elif wrisk == 'High Risk':
                    st.warning("⚠️ High failure probability — schedule urgent inspection.")
                    if st.button("Schedule Inspection", key=f"inspect_action_{wrow['id']}"):
                        st.success("Inspection Scheduled.")
            else:
                st.info("ℹ️ No AI risk data available yet.")


# ─────────────────────────────────────────────
# SECTION 1: EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────

if selected == "Executive Overview":
    # st.markdown("### Predictive Maintenance Control Center")
    st.markdown("### Asset Risk Analysis for Industrial Machinery")
    
    # ─── STRATEGY VIEW FILTER ───
    selected_strategy = st.selectbox(
        "Select Strategic View:", 
        RAW_STRATEGIES, 
        index=RAW_STRATEGIES.index(st.session_state.selected_strategy_key),
        help="All: All Assets | Predictive: AI Risks | Preventive: Scheduled Tasks | Ad-hoc: Unplanned Repairs"
    )
    st.session_state.selected_strategy_key = selected_strategy
    
    # Base Data
    filtered_assets = df_assets.copy()
    filtered_wo = df_wo.copy()
    filtered_costs = df_costs.copy()

    if selected_strategy == "Predictive":
        # 1. Get AI Predictions for ALL assets first
        with st.spinner("Analyzing fleet-wide predictive risks..."):
            all_preds = batch_predict_assets(df_assets)
            critical_ids = [p['id'] for p in all_preds if p['risk_level'] in ['Critical', 'High Risk']]
        
        filtered_assets = df_assets[df_assets['id'].isin(critical_ids)]
        filtered_wo = df_wo[df_wo['asset_id'].isin(critical_ids)]
        filtered_costs = df_costs[df_costs['work_order_id'].isin(filtered_wo['id'])]

    elif selected_strategy == "Preventive":
        # AI-Driven Prevention: Focused strictly on 'Warning' signals (Early intervention)
        with st.spinner("Scanning for early-stage deviations..."):
            all_preds = batch_predict_assets(df_assets)
            warning_ids = [p['id'] for p in all_preds if p['risk_level'] == 'Warning']
        
        filtered_assets = df_assets[df_assets['id'].isin(warning_ids)]
        filtered_wo = df_wo[df_wo['asset_id'].isin(warning_ids)]
        filtered_costs = df_costs[df_costs['work_order_id'].isin(filtered_wo['id'])]

    elif selected_strategy == "Ad-hoc / On-demand":
        # Narrow down strictly to assets with ACTIVE EMERGENCY repairs
        active_wo_mask = (df_wo['status'] == 'in_progress')
        adhoc_keywords = ["Emergency"]
        keyword_mask = df_wo['title'].str.contains('|'.join(adhoc_keywords), case=False, na=False)
        priority_mask = df_wo['priority'].isin(['critical', 'high'])
        
        adhoc_asset_ids = df_wo[active_wo_mask & keyword_mask & priority_mask]['asset_id'].unique().tolist()
        
        filtered_assets = df_assets[df_assets['id'].isin(adhoc_asset_ids)]
        filtered_wo = df_wo[df_wo['asset_id'].isin(adhoc_asset_ids)]
        filtered_costs = df_costs[df_costs['work_order_id'].isin(filtered_wo['id'])]

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. METRICS ROW
    # Filter for high priority active work orders
    # Usage filtered dataframes
    # critical_wo_metric = filtered_wo[filtered_wo['priority'].isin(['critical', 'high'])].sort_values('priority', ascending=True)
    
    # ---------------------------------------------------------
    # 🧠 AI-POWERED ANALYSIS (CACHED)
    # ---------------------------------------------------------
    all_predictions = batch_predict_assets(filtered_assets)
    
    # Filter for significant alerts for the KPI and the primary list
    predicted_critical_assets = [p for p in all_predictions if p['risk_level'] in ['Critical', 'High Risk']]
    
    # Handle Database side-effects (AI Work Order Generation) separately from the cache
    db = SessionLocal()
    new_wo_created = False
    
    for asset_pred in predicted_critical_assets:
        if asset_pred['risk_level'] in ['Critical', 'High Risk']:
            # Check if active AI generated work order exists
            existing_wo = db.query(WorkOrder).filter(
                WorkOrder.asset_id == asset_pred['id'], 
                WorkOrder.status == 'in_progress', 
                WorkOrder.title.like('%AI GENERATED%')
            ).first()
            if not existing_wo:
                import random
                techs = ['Tech A. Miller', 'Tech B. Johnson', 'Eng C. Smith', 'Tech D. Garcia']
                new_wo = WorkOrder(
                    asset_id=asset_pred['id'],
                    title=f"AI GENERATED: {asset_pred['risk_level']} Intervention Required - {asset_pred['name']}",
                    description=f"AI detected {asset_pred['risk_level']} risk ({asset_pred['risk_score']}%). Expected failure in {asset_pred['rul']} days.",
                    status='in_progress',
                    priority='critical' if asset_pred['risk_level'] == 'Critical' else 'high',
                    assigned_to=random.choice(techs),
                    created_at=datetime.utcnow()
                )
                db.add(new_wo)
                db.commit()
                new_wo_created = True
                new_wo_created = True
                
    db.close()
    
    if new_wo_created:
        st.cache_data.clear() # Clear cache to fetch new WOs next refresh
        df_assets, df_wo, df_costs = get_db_data() # Refresh data immediately for this run
        filtered_wo = df_wo[df_wo['asset_id'].isin(filtered_assets['id'])]
    
    # Use ALL predictions for the action center display, but sorted
    if all_predictions:
        all_predictions_df = pd.DataFrame(all_predictions)
    else:
        # Create empty DF with expected columns to avoid KeyError
        all_predictions_df = pd.DataFrame(columns=['id', 'name', 'risk_score', 'risk_level', 'predicted_cost', 'rul', 'asset_type', 'industry_name'])
        
    # This is for the KPI: Only count the truly critical ones
    if not all_predictions_df.empty and 'risk_level' in all_predictions_df.columns:
        truly_critical_df = all_predictions_df[all_predictions_df['risk_level'].isin(['Critical', 'High Risk'])]
    else:
        truly_critical_df = all_predictions_df.copy() # Empty with same columns
    
    if not all_predictions_df.empty:
        all_predictions_df = all_predictions_df.sort_values('risk_score', ascending=False)
            
    total_assets = len(filtered_assets)
    active_wo = filtered_wo[filtered_wo['status'] == 'in_progress']
    total_spend = filtered_costs['total_cost'].sum()
    
    # ─── CALCULATE AI METRICS ───
    if selected_strategy == "All":
        # Global view: Total exposure including early-stage warnings (everything not healthy)
        risky_assets_df = all_predictions_df[all_predictions_df['risk_level'] != 'Healthy']
        predicted_liability = risky_assets_df['predicted_cost'].sum() if not risky_assets_df.empty else 0
        liability_label = "Global Risk Exposure"
        liability_sublabel = "Total value across all alerts"
    elif selected_strategy in ["Preventive", "Ad-hoc / On-demand"]:
        # Specific view: Sum everything currently filtered
        predicted_liability = all_predictions_df['predicted_cost'].sum() if not all_predictions_df.empty else 0
        liability_label = "Financial Liability" if selected_strategy == "Ad-hoc / On-demand" else "Potential Risk Value"
        liability_sublabel = "Estimated repair cost" if selected_strategy == "Ad-hoc / On-demand" else "Estimated cost of early repair"
    else: # Predictive
        # Actionable view: Strictly focus on the imminent threats (Critical/High)
        predicted_liability = truly_critical_df['predicted_cost'].sum() if not truly_critical_df.empty else 0
        liability_label = "Financial Liability"
        liability_sublabel = "Cost of ignoring alerts"
        
    # Assume catching a failure early saves 60% of the replacement/catastrophic cost
    # Based on: Early repair ~40% of cost, Catastrophic failure ~100% of cost.
    ai_cost_avoidance = predicted_liability * 0.60 
    
    # ─── AI MORNING BRIEFING ───
    st.markdown("##### AI Executive Briefing")
    
    # Construct strictly factual payload
    # For top_failures, show Warnings if we are in Preventive, otherwise show Criticals
    top_list = all_predictions_df if selected_strategy == "Preventive" else truly_critical_df
    
    # Context-aware alert naming for briefing
    if selected_strategy == "Ad-hoc / On-demand":
        alert_type_name = "Emergency Interventions"
        alert_count_briefing = len(filtered_assets) # In ad-hoc, everything filtered is an emergency
    else:
        alert_type_name = "Critical Risks"
        alert_count_briefing = len(truly_critical_df)

    telemetry_payload = {
        "total_assets": total_assets,
        "active_work_orders": len(active_wo),
        "total_maintenance_spend": total_spend,
        "critical_alerts_count": alert_count_briefing,
        "predicted_liability": predicted_liability,
        "alert_type_label": alert_type_name,
        "imminent_failure_examples": ", ".join([f"{row['name']} ({row['risk_score']}%)" for idx, row in top_list.head(10).iterrows()]) if not top_list.empty else "None"
    }
    
    with st.spinner("AI is analyzing real-time factory state..."):
        ai_briefing = get_executive_briefing(telemetry_payload)
    
    st.info(f"**{ai_briefing}**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─── PROACTIVE KPIs ───
    cols_metrics = st.columns(4)
    
    view_label = "Total across fleet" if selected_strategy == "All" else f"Scope: {selected_strategy} View"
    
    if selected_strategy == "Preventive":
        alert_label = "Warning AI Alerts"
        alert_count = len(all_predictions_df[all_predictions_df['risk_level'] == 'Warning'])
        alert_sublabel = "Early-stage risks detected"
    elif selected_strategy == "Predictive":
        alert_label = "Critical AI Alerts"
        alert_count = len(truly_critical_df)
        alert_sublabel = "Imminent Failures Predicted"
    elif selected_strategy == "Ad-hoc / On-demand":
        alert_label = "Operational Load"
        alert_count = len(filtered_assets)
        alert_sublabel = "Assets requiring emergency care"
    else: # All
        alert_label = "Global Risk Count"
        alert_count = len(truly_critical_df)
        warning_count = len(all_predictions_df[all_predictions_df['risk_level'] == 'Warning'])
        healthy_count = len([p for p in all_predictions if p['risk_level'] == 'Healthy'])
        alert_sublabel = f"{alert_count} Risks | {warning_count} Warnings | {healthy_count} Healthy"
    
    metric_data = [
        ("Total Assets Monitored", f"{len(df_assets)}", "Total Fleet Scale", "delta-neutral", "Total number of machines currently transmitting sensor data to the system."),
        (alert_label, f"{alert_count}", alert_sublabel, "delta-neg", "Number of assets the AI model predicts require attention based on current strategy."),
        (liability_label, format_currency_smart(predicted_liability), liability_sublabel, "delta-neg", "Estimated financial impact of the assets currently in this operational view."),
        ("AI Cost Avoidance", format_currency_smart(ai_cost_avoidance), "Saved by intervening now", "delta-pos", "Estimated capital saved by pulling these assets offline now for minor preventative maintenance, rather than letting them catastrophically fail.")
    ]

    for i, col in enumerate(cols_metrics):
        label, value, delta, delta_type, tooltip = metric_data[i]
        col.markdown(metric_card(label, value, delta, delta_type, tooltip), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    
    # 2. MAIN CONTENT AREA: AI ACTION CENTER
    st.markdown("##### ⚡ AI Action Center")
    st.caption("AI-prioritized interventions based on failure probability, imminent risk, and economic impact.")
    
    # ── SHARED GLASSMORPHISM CSS ──
    st.markdown("""
        <style>
        @keyframes pulse_alert { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .glass-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.05);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    if all_predictions_df.empty:
        st.success("✅ No risks detected by AI across monitored assets.")
    else:
        # Wrap the cards in a fixed-height, scrollable container
        with st.container(height=650):
            # Create a visually distinct layout for the action center instead of columns
            for idx, row in all_predictions_df.iterrows():
                # Dynamic Interventions Data (Live from DB)
                asset_id_row = row['id']
                assoc_wo = df_wo[(df_wo['asset_id'] == asset_id_row) & (df_wo['status'] == 'in_progress')].sort_values('created_at', ascending=False)
                
                wrisk = row['risk_level']
                wasset = row['name']
                wscore = row['risk_score']
                
                # Keep numeric RUL for logic, use wrul_display for UI
                rul_val = row['rul']
                wrul_display = "N/A" if pd.isna(rul_val) else f"{int(rul_val)}"
                wcost = row['predicted_cost']
                
                # Context-Aware Badge selection
                has_emergency = not assoc_wo.empty and any("Emergency" in str(t) for t in assoc_wo['title'])

                if has_emergency and selected_strategy == "Ad-hoc / On-demand":
                    wbadge = "⚒️ EMERGENCY"
                    header_color = "#fef2f2" # Light Red/Amber
                    border_color = "#dc2626"
                elif wrisk == 'Critical':
                    wbadge = "🔴 CRITICAL"
                    header_color = "#fee2e2"
                    border_color = "#ef4444"
                elif wrisk == 'High Risk':
                    wbadge = "🟠 HIGH RISK"
                    header_color = "#ffedd5"
                    border_color = "#f97316"
                elif wrisk == 'Warning':
                    wbadge = "🟡 WARNING"
                    header_color = "#fef9c3"
                    border_color = "#eab308"
                else:
                    wbadge = "🟢 HEALTHY"
                    header_color = "#f0fdf4"
                    border_color = "#22c55e"
                
                # Simple AI Logic for Technician Dispatch Recommendation
                if wrisk == 'Critical' and not pd.isna(rul_val) and rul_val <= 7:
                    tech_recommendation = "Dispatch MASTER Technician (Immediate response required to prevent catastrophic failure in < 7 days)."
                elif wcost == 0:
                    tech_recommendation = "Dispatch OEM/WARRANTY Tech (Asset is covered under active warranty. $0 out-of-pocket repair)."
                elif wcost > 50000:
                    tech_recommendation = "Dispatch SENIOR Technician (High liability repair requires experienced oversight)."
                else:
                    tech_recommendation = "Dispatch JUNIOR/INTERMEDIATE Technician (Standard repair, optimize for lower hourly rate)."
    
                if not assoc_wo.empty:
                    latest_wo = assoc_wo.iloc[0]
                    tech_name = latest_wo['assigned_to'] or "Unassigned"
                    wo_status = latest_wo['status'].upper().replace('_', ' ')
                    # Intervention clock calculation (hours ago)
                    try:
                        # Use UTC aware parsing and handle mixed formats
                        created_at = pd.to_datetime(latest_wo['created_at'], format='ISO8601')
                        hours_diff = (pd.Timestamp.utcnow() - created_at.tz_localize('UTC')).total_seconds() / 3600
                        hours_ago = round(hours_diff, 1)
                    except:
                        hours_ago = 0.1 # Fallback
                    
                    status_info = f"""<div style="display: flex; gap: 15px; margin-top: 10px; padding-top: 10px; border-top: 1px dashed #e5e7eb;">
<div style="flex:1;"><small style="color: #6b7280; font-size:10px; text-transform:uppercase;">ASSIGNED TO</small><br><b style="font-size:13px;">{tech_name}</b></div>
<div style="flex:1;"><small style="color: #6b7280; font-size:10px; text-transform:uppercase;">STATUS</small><br><b style="color: #2563eb; font-size:13px;">{wo_status}</b></div>
<div style="flex:1;"><small style="color: #6b7280; font-size:10px; text-transform:uppercase;">RESP. CLOCK</small><br><b style="font-size:13px;">{hours_ago} hrs</b></div>
</div>"""
                else:
                    status_info = """<div style="display: flex; gap: 15px; margin-top: 10px; padding-top: 10px; border-top: 1px dashed #e5e7eb;">
<div style="flex:1;"><b style="color: #ef4444; font-size:13px;">NO ACTIVE DISPATCH</b></div>
</div>"""
    
                # ── PREMIUM GLASSMORPHISM CARD DESIGN ──
                # Check if this alert is unacknowledged
                alert_key = f"alert_{row['id']}_{row['risk_score']}"
                is_new = alert_key not in st.session_state.acknowledged_alerts and wrisk in ['Critical', 'High Risk']
                new_badge = '<span style="background:#ef4444; color:white; padding:2px 6px; border-radius:10px; font-size:9px; vertical-align:middle; margin-left:8px; animation: pulse_alert 2s infinite;">NEW ALERT</span>' if is_new else ""

                st.markdown(f"""
<div class="glass-card" style="border-left: 6px solid {border_color};">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<span style="font-weight: 800; font-size: 15px; color: #1f2937;">{wbadge} | {wasset} {new_badge}</span>
<span style="font-size: 11px; color: #6b7280; font-weight: 600; text-transform: uppercase;">ID: {row['id']}</span>
</div>
<div style="display: flex; gap: 15px; margin-bottom: 18px;">
<div style="flex: 1; background: rgba(0,0,0,0.03); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.05);">
<div style="font-size: 10px; color: #6b7280; text-transform: uppercase; font-weight: 700;">Risk Level</div>
<div style="font-size: 18px; font-weight: 800; color: {border_color};">{wscore}%</div>
</div>
<div style="flex: 1; background: rgba(0,0,0,0.03); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.05);">
<div style="font-size: 10px; color: #6b7280; text-transform: uppercase; font-weight: 700;">RUL Estimate</div>
<div style="font-size: 18px; font-weight: 800; color: #111827;">{wrul_display} <small style="font-size: 11px;">Days</small></div>
</div>
<div style="flex: 1; background: rgba(0,0,0,0.03); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.05);">
<div style="font-size: 10px; color: #6b7280; text-transform: uppercase; font-weight: 700;">Liability</div>
<div style="font-size: 18px; font-weight: 800; color: #111827;">${wcost:,.0f}</div>
</div>
</div>
<div style="padding: 12px; background: {header_color}60; border-radius: 8px; border: 1px solid {border_color}30; margin-bottom: 15px;">
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
<span style="font-size: 16px;">🤖</span>
<strong style="color: #374151; font-size: 12px; text-transform: uppercase;">AI Dispatch Recommendation</strong>
</div>
<div style="font-size: 13px; color: #111827;">{tech_recommendation}</div>
</div>
{status_info}
</div>
""", unsafe_allow_html=True)
                
                # Action Buttons underneath each card
                col1, col2, col3 = st.columns([1.5, 1.2, 3])
                with col1:
                    if not assoc_wo.empty:
                        # Intervention is already happening - show tracker status
                        st.markdown(f"""
                            <div style="background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px 12px; text-align: center; font-size: 13px; font-weight: 600;">
                                Intervention in Progress
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Unified Dispatch Button
                        btn_label = "Acknowledge & Dispatch" if is_new else "Approve AI Dispatch"
                        if st.button(btn_label, key=f"dispatch_action_{row['id']}", type="primary", use_container_width=True):
                            db = SessionLocal()
                            try:
                                new_wo = WorkOrder(
                                    asset_id=int(row['id']),
                                    title=f"AI GENERATED: {row['risk_level']} Intervention Required - {row['name']}",
                                    description=f"Manually Approved AI Dispatch for {row['risk_level']} risk.",
                                    status='in_progress',
                                    priority='critical' if row['risk_level'] == 'Critical' else 'high',
                                    created_at=datetime.now(timezone.utc)
                                )
                                db.add(new_wo)
                                db.commit()
                                # Mark as acknowledged if it was new
                                if is_new:
                                    st.session_state.acknowledged_alerts.add(alert_key)
                                
                                # 🚨 TRIGGER REAL-TIME SENDGRID EMAIL 🚨
                                with st.spinner("Executing secure email dispatch..."):
                                    success, msg = send_email_via_sendgrid(
                                        asset_name=row['name'],
                                        tech_recommendation=tech_recommendation,
                                        risk_level=row['risk_level'],
                                        asset_id=row['id'],
                                        rul=wrul_display,
                                        liability=row['predicted_cost']
                                    )
                                    if success:
                                        st.success(f"✅ Dispatched & Email Alert sent to maintenance lead!")
                                        st.toast("📧 Email sent successfully!", icon="📬")
                                        import time
                                        time.sleep(1.5)
                                        st.rerun()
                                    else:
                                        st.error(f"⚠️ Dispatch Alert Error: {msg}")
                                        st.info("Check if SENDGRID_API_KEY is valid and your sender email is verified in SendGrid.")
                            except Exception as e:
                                st.error(f"DB Error: {e}")
                                db.rollback()
                            db.close()
                with col2:
                    if st.button("Human Review", key=f"review_action_{row['id']}", use_container_width=True):
                        recipient = os.getenv("SENDGRID_TO_EMAIL")
                        if not recipient:
                            st.warning("⚠️ No Reliability Team email configured (SENDGRID_TO_EMAIL missing in .env).")
                        else:
                            with st.spinner("Notifying Reliability Team..."):
                                # We send a 'Review Request' instead of a 'Dispatch Dispatch'
                                success, msg = send_email_via_sendgrid(
                                    asset_name=row['name'],
                                    tech_recommendation=f"HUMAN REVIEW REQUESTED: Peer review needed for {row['risk_level']} risk assessment.",
                                    risk_level=f"REVIEW: {row['risk_level']}",
                                    asset_id=row['id'],
                                    rul=wrul_display,
                                    liability=row['predicted_cost']
                                )
                                if success:
                                    st.success("✅ Review Alert sent to the respective Reliability Team.")
                                else:
                                    st.error(f"❌ Failed to notify: {msg}")
                
                st.markdown("<br>", unsafe_allow_html=True)




# ─────────────────────────────────────────────

elif selected == "Asset Monitor":
    st.markdown("###  AI Diagnostics Engine")
    st.caption("AI continuously monitors sensor data to detect anomalies, predict component failures, and estimate repair costs — before breakdowns happen.")
    st.markdown("---")

    # ── FILTERS ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns([1, 3])
    current_strategy_label = STRATEGY_MAP_WITH_COUNTS[st.session_state.selected_strategy_key]
    strategy_view_label = f1.selectbox(
        "Strategic View", 
        STRATEGY_OPTIONS_WITH_COUNTS, 
        index=STRATEGY_OPTIONS_WITH_COUNTS.index(current_strategy_label),
        help="Filter assets based on maintenance strategy."
    )
    strategy_view = STRATEGY_REVERSE_MAP_WITH_COUNTS[strategy_view_label]
    st.session_state.selected_strategy_key = strategy_view
    
    filtered_assets = df_assets.copy()
    if strategy_view == "Predictive":
        with st.spinner("Refining for predictive risks..."):
            all_preds = batch_predict_assets(df_assets)
            critical_ids = [p['id'] for p in all_preds if p['risk_level'] in ['Critical', 'High Risk']]
        filtered_assets = df_assets[df_assets['id'].isin(critical_ids)]
    elif strategy_view == "Preventive":
        with st.spinner("Filtering for early-warning assets..."):
            all_preds = batch_predict_assets(df_assets)
            warning_ids = [p['id'] for p in all_preds if p['risk_level'] == 'Warning']
        filtered_assets = df_assets[df_assets['id'].isin(warning_ids)]
    elif strategy_view == "Ad-hoc / On-demand":
        active_wo_mask = (df_wo['status'] == 'in_progress')
        adhoc_keywords = ["Emergency"]
        keyword_mask = df_wo['title'].str.contains('|'.join(adhoc_keywords), case=False, na=False)
        priority_mask = df_wo['priority'].isin(['critical', 'high'])
        adhoc_asset_ids = df_wo[active_wo_mask & keyword_mask & priority_mask]['asset_id'].unique().tolist()
        filtered_assets = df_assets[df_assets['id'].isin(adhoc_asset_ids)]

    # ── ASSET SELECTION & INTELLIGENT SYNC ─────────────────────────────────────
    asset_id = f2.selectbox(
        "Asset",
        options=filtered_assets["id"].unique(),
        key="asset_monitor_selector",
        format_func=lambda x: f"{filtered_assets[filtered_assets['id']==x]['name'].iloc[0]}  ({filtered_assets[filtered_assets['id']==x]['asset_type'].iloc[0]})"
    )

    # Trigger Sync: If the user picks an asset, automatically align the Strategic View
    if "last_synced_asset" not in st.session_state:
        st.session_state.last_synced_asset = None

    if st.session_state.last_synced_asset != asset_id:
        st.session_state.last_synced_asset = asset_id
        
        # Predict once for the newly selected asset to determine its correct strategy
        db = SessionLocal()
        risk_profile = predict_asset_risk(asset_id, db)
        db.close()
        
        detected_risk = risk_profile.get("risk_level", "Healthy")
        
        # Map Risk Level to the corresponding Strategy
        target_strategy = "All"
        if detected_risk in ["Critical", "High Risk"]:
            target_strategy = "Predictive"
        elif detected_risk == "Warning":
            target_strategy = "Preventive"
            
        # Only switch if there's a mismatch and we're not in a manual Ad-hoc state
        if st.session_state.selected_strategy_key != target_strategy and st.session_state.selected_strategy_key != "Ad-hoc / On-demand":
            st.session_state.selected_strategy_key = target_strategy
            st.rerun()

    if filtered_assets.empty or asset_id not in filtered_assets["id"].values:
        st.warning(f"No assets found for the '{strategy_view}' view.")
        st.stop()

    selected_asset = filtered_assets[filtered_assets["id"] == asset_id].iloc[0]

    # ── AI RISK KPIs ──────────────────────────────────────────────────────────
    try:
        db = SessionLocal()
        risk_assessment = predict_asset_risk(asset_id, db)
        db.close()

        risk_score   = risk_assessment.get("risk_score", 0)
        risk_level   = risk_assessment.get("risk_level", "Unknown")
        rul          = risk_assessment.get("predicted_days_to_failure", 0) or 0
        health_pct   = max(0, 100 - risk_score)
        
        # Calculate Asset Age and Warranty Status
        purchase_date = pd.to_datetime(selected_asset['purchase_date'])
        warranty_expiry = pd.to_datetime(selected_asset['warranty_expiry']) if pd.notnull(selected_asset['warranty_expiry']) else None
        
        now = datetime.now()
        age_days      = (now - purchase_date).days
        years         = age_days // 365
        months        = (age_days % 365) // 30
        age_str       = f"{years}y {months}m" if years > 0 else f"{months} months"
        
        is_active_warranty = warranty_expiry is not None and now < warranty_expiry

        if risk_level == "Warning":      risk_color = "#f59e0b"
        elif risk_level == "High Risk":  risk_color = "#ef4444"
        elif risk_level == "Critical":   risk_color = "#7f1d1d"
        else:                            risk_color = "#10b981"

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(metric_card("Overall Health",
                                f"{health_pct:.1f}%",
                                "100% = no risk | <70% = action needed",
                                "delta-pos" if health_pct > 70 else "delta-neg"), unsafe_allow_html=True)
        k2.markdown(metric_card("Remaining Useful Life",
                                f"{rul} days",
                                "AI-estimated days before likely failure",
                                "delta-neutral"), unsafe_allow_html=True)
        k3.markdown(metric_card("AI Risk Level",
                                risk_level,
                                "Healthy → Warning → High Risk → Critical",
                                "delta-neg" if risk_level in ["Critical","High Risk"] else "delta-neutral"), unsafe_allow_html=True)
        k4.markdown(metric_card("Warranty Status",
                                "Active Coverage" if is_active_warranty else "Expired Coverage",
                                f"Age: {age_str} | Exp: {warranty_expiry.strftime('%b %Y') if warranty_expiry else 'N/A'}",
                                "delta-pos" if is_active_warranty else "delta-neg"), unsafe_allow_html=True)


    except Exception as e:
        st.error(f"AI Prediction Error: {e}")
        risk_level = "Unknown"
        risk_score = 0
        rul = 0
        risk_color = "#9CA3AF"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABBED MAIN CONTENT ───────────────────────────────────────────────────
    # ── SINGLE-VIEW LAYOUT: EVERYTHING AT ONCE ────────────────────────────────
    # Fetch sensor data (fetch more history, then filter in pandas)
    sensors_raw = pd.read_sql(
        f"SELECT * FROM sensors WHERE asset_id = {asset_id} ORDER BY timestamp DESC LIMIT 3000",
        engine
    )

    col_left, col_right = st.columns([3, 2])

    # ── LEFT: TELEMETRY ────────────────────────────────────────────────────────
    with col_left:
        st.markdown("##### Sensor Intelligence")
        st.caption("Asset telemetry and physical indicators.")
        
        sensors = pd.DataFrame() # Bound the variable
        
        # Add AI Summary
        if not sensors_raw.empty:
            # We want to give the AI the very latest reading, not the whole history
            latest_readings = sensors_raw.head(5)
            sensor_text = ", ".join([f"{row['sensor_type']}: {row['value']:.1f}" for idx, row in latest_readings.iterrows()])
            
            with st.spinner("AI is interpreting sensor telemetry..."):
                sensor_insight = cached_get_sensor_summary(sensor_text, selected_asset['asset_type'])
                
            st.info(f"**AI Rapid Assessment:** {sensor_insight}")

        if not sensors_raw.empty:
            sensors_raw['timestamp'] = pd.to_datetime(sensors_raw['timestamp'], format='ISO8601')
            
            # --- TIME RANGE FILTER ---
            time_filter = st.radio(
                "Time Range", 
                ["Latest 30 Days", "Latest 6 Months", "All Time"], 
                index=0, horizontal=True, label_visibility="collapsed"
            )
            
            now = pd.Timestamp.now()

            if time_filter == "Latest 30 Days":
                cutoff = now - pd.Timedelta(days=30)
            elif time_filter == "Latest 6 Months":
                cutoff = now - pd.Timedelta(days=180)
            else:
                cutoff = sensors_raw['timestamp'].min()
                
            sensors = sensors_raw[sensors_raw['timestamp'] >= cutoff]

            # Sensor summary KPI row
            if not sensors.empty:
                sensor_types = sensors['sensor_type'].unique().tolist()
                unit_map = {'temperature': '°C', 'vibration': 'mm/s', 'pressure': 'Bar', 'rpm': 'RPM', 'current': 'A'}
                sensor_help = {
                    'temperature': 'Avg operating temp (°C). High avg = thermal stress. Spike risk if near max threshold.',
                    'vibration':   'Avg vibration (mm/s). Elevated readings indicate imbalance or bearing wear.',
                    'pressure':    'Avg pressure (Bar). Deviations suggest seal failure or blockage.',
                    'rpm':         'Avg rotational speed. Drop or spike signals motor/drive issues.',
                    'current':     'Avg electrical current (A). Overcurrent = winding failure risk.',
                }
                kcols = st.columns(len(sensor_types))
                for idx_s, stype in enumerate(sensor_types):
                    s_data = sensors[sensors['sensor_type'] == stype]['value']
                    unit   = unit_map.get(stype, '')
                    kcols[idx_s].metric(
                        label=stype.title(),
                        value=f"{s_data.iloc[0]:.1f} {unit}" if not s_data.empty else f"0 {unit}",
                        delta=f"Max: {s_data.max():.1f} {unit}",
                        delta_color="off",
                        help=sensor_help.get(stype, f"Latest {stype} reading.")
                    )

                fig = px.line(sensors, x="timestamp", y="value", color="sensor_type",
                              color_discrete_sequence=px.colors.qualitative.G10, template='plotly_white')
                fig.update_layout(
                    height=260, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis_gridcolor="#e5e7eb", yaxis_gridcolor="#e5e7eb",
                    margin=dict(t=5, l=0, r=0, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sensor data available for the selected time range.")
            
            st.caption(f"{selected_asset['asset_type']} · {selected_asset['location']} · {selected_asset['manufacturer']}")
        else:
            st.info("No sensor data available.")

    # ── RIGHT: AI PREDICTION PANEL ──────────────────────────────────────────
    with col_right:
        st.markdown("##### AI Prediction Panel")
        if risk_level == "Healthy":
            st.success("**AI Status: Normal Operation**")
            st.caption("AI is continuously monitoring all sensors. No anomalies detected — all readings within learned normal ranges.")
        elif not sensors.empty:
            # AI has identified a risk.
            features = {}
            installed_sensors = set()
            for stype in ['temperature', 'vibration', 'pressure', 'rpm', 'current']:
                vals = sensors[sensors['sensor_type'] == stype]['value'].values
                if len(vals) > 0:
                    installed_sensors.add(stype)
                    features[f'{stype}_avg'] = float(np.mean(vals))
                    features[f'{stype}_max'] = float(np.max(vals))
                    features[f'{stype}_std'] = float(np.std(vals))
                    mid = len(vals) // 2
                    features[f'{stype}_trend'] = float(np.mean(vals[mid:]) - np.mean(vals[:mid])) if mid > 0 else 0.0
                else:
                    features[f'{stype}_avg'] = 50.0  # Match training defaults

            try:
                # Use Global Risk assessment directly
                pred_label = f"{risk_level} Degradation Pattern"
                prob_max = risk_score / 100.0
                
                # Assign confidence logic
                if prob_max > 0.85:   detail = f"Urgent: {prob_max:.1%} mathematical confidence of failure."
                elif prob_max > 0.60: detail = f"Elevated Risk: {prob_max:.1%} confidence of accelerated degradation."
                else:                 detail = f"Warning: {prob_max:.1%} likelihood of developing stress."

                display_label = pred_label
                if risk_level == "Warning":
                    st.warning(f"**⚠️ AI Early Warning: {display_label}** — {detail}")
                else:
                    st.error(f"**🔴 AI Predicted Failure: {display_label}** — {detail}")



                # ── COST BREAKDOWN ─────────────────────────────────────────
                purchase_date = pd.to_datetime(selected_asset['purchase_date'])
                age_days      = (datetime.now() - purchase_date).days
                priority_str  = ('critical' if risk_level == 'Critical' else
                                 'high'     if risk_level == 'High Risk' else 'medium')
                try:
                    # Model was trained on log(cost), convert back with expm1
                    est_cost = float(np.expm1(cost_model.predict(pd.DataFrame([{
                        'priority': priority_str, 'asset_type': selected_asset['asset_type'],
                        'industry': selected_asset['industry_name'], 'asset_age_days': age_days,
                        'description_len': len(f"Predicted failure of {detail}")
                    }]))[0]))
                except:
                    est_cost = 2450.0
                try:
                    baseline = float(np.expm1(cost_model.predict(pd.DataFrame([{
                        'priority': 'low', 'asset_type': selected_asset['asset_type'],
                        'industry': selected_asset['industry_name'], 'asset_age_days': age_days,
                        'description_len': len(f"Planned maintenance of {detail}")
                    }]))[0]))
                except:
                    baseline = est_cost * 0.35
                premium = max(0, est_cost - baseline)

                # Historical proportions
                hist_wo   = df_wo[df_wo['priority'].str.lower() == priority_str]
                hist_costs = df_costs[df_costs['work_order_id'].isin(hist_wo['id'])]
                wo_t = hist_wo.merge(df_assets[['id','asset_type']], left_on='asset_id', right_on='id', how='left')
                idc  = 'id_x' if 'id_x' in wo_t.columns else 'id'
                ref_costs = hist_costs[hist_costs['work_order_id'].isin(
                    wo_t[wo_t['asset_type'] == selected_asset['asset_type']][idc].values
                )]
                if len(ref_costs) < 5:
                    ref_costs = hist_costs
                if not ref_costs.empty and ref_costs['total_cost'].sum() > 0:
                    avg_t = ref_costs['total_cost'].mean()
                    lp = ref_costs['labor_cost'].mean() / avg_t
                    pp = ref_costs['parts_cost'].mean() / avg_t
                    op = ref_costs['other_cost'].mean() / avg_t
                    avg_hours = ref_costs['labor_hours'].mean()
                    avg_rate  = (ref_costs['labor_cost'] / ref_costs['labor_hours'].replace(0,1)).mean()
                else:
                    lp, pp, op, avg_hours, avg_rate = 0.55, 0.30, 0.15, 3.0, 100

                part_map = {"Bearing": "Roller bearings, races", "Cooling": "Fan blades, coolant",
                            "Electrical": "Windings, capacitors", "Seal": "Hydraulic seals, O-rings",
                            "General": "Assorted components", "Preventive": "Filters, lubricants"}
                parts_lbl = part_map.get(pred_label, "Replacement parts")

                st.markdown("**💰 Predictive Cost Intelligence**")
                st.caption("AI estimates repair costs based on asset type, failure mode, and urgency level.")
                ca, cb, cc = st.columns(3)
                ca.metric(
                    "👷 Labor",
                    f"${baseline*lp:,.0f}",
                    f"{avg_hours:.0f}h × ${avg_rate:.0f}/hr",
                    delta_color="off",
                    help=f"Technician hours ({avg_hours:.1f}h avg) × hourly rate (${avg_rate:.0f}/hr). "
                         f"Priority 'low' applied for 'Act Now' scenario."
                )
                cb.metric(
                    "🔩 Parts",
                    f"${baseline*pp:,.0f}",
                    parts_lbl,
                    delta_color="off",
                    help=f"Replacement components for predicted {display_label} failure: {parts_lbl}. "
                         f"Cost based on standard sourcing for 'Act Now' scenario."
                )
                cc.metric(
                    "📦 Other",
                    f"${baseline*op:,.0f}",
                    "Downtime + transport",
                    delta_color="off",
                    help="Other costs including calculated downtime impact and logistics."
                )

                mult_color = "#dc2626" if risk_level == "Critical" else "#ea580c" if risk_level == "High Risk" else "#d97706"
                if risk_level == "Warning":
                    cost_msg = f'✅ Act now at <b style="color:#16a34a">${baseline:,.0f}</b> — waiting adds <b style="color:#dc2626">+${premium:,.0f}</b>'
                else:
                    cost_msg = f'⚠️ <b style="color:{mult_color}">${est_cost:,.0f} total</b> vs planned <b style="color:#16a34a">${baseline:,.0f}</b> (+${premium:,.0f} urgency)'

                st.markdown(
                    f'<div style="padding:6px 10px;background:#fff8f8;border-left:3px solid {mult_color};'
                    f'border-radius:6px;font-size:11px;color:#374151;margin-bottom:8px">{cost_msg}</div>',
                    unsafe_allow_html=True
                )

                # ── AI REASONING ──────────────────────────────────────────
                st.markdown("**🧠 AI Failure Analysis**")
                st.caption("AI explains WHY it flagged this asset based on sensor anomalies.")

                # Note: contrib is now calculated earlier to be used in col_left
                # Collect signal data for LLM
                signals_for_ai = []
                # First, ensure we have a snapshot of all active sensor averages
                for stype in installed_sensors:
                    val = features.get(f"{stype}_avg", 0)
                    unit = {'temperature':'°C','vibration':'mm/s','pressure':'Bar','rpm':'RPM','current':'A'}.get(stype,'')
                    signals_for_ai.append(f"{stype.title()}: {val:.1f}{unit}")
                    
                # Calculate simple statistical drift for specific deviations
                if risk_level != "Healthy":
                    deviations = []
                    for stype in installed_sensors:
                        avg_val = features.get(f"{stype}_avg", 0)
                        max_val = features.get(f"{stype}_max", 0)
                        std_val = features.get(f"{stype}_std", 1) # Prevent div by 0
                        
                        # We approximate anomaly if the current point (represented by max since it's the latest high watermark)
                        # deviates significantly from the mean based on standard deviations.
                        if std_val > 0:
                            z_score = abs(max_val - avg_val) / std_val
                            if z_score > 1.5: # 1.5 standard deviations indicates some drift
                                deviations.append({'sensor': stype, 'z_score': z_score, 'max': max_val})
                                
                    # Sort deviations by highest severity
                    deviations = sorted(deviations, key=lambda x: x['z_score'], reverse=True)
                    
                    # Extract the top 2 highest drifting sensors for context
                    for dev in deviations[:2]:
                        stype = dev['sensor']
                        val = dev['max']
                        unit = {'temperature':'°C','vibration':'mm/s','pressure':'Bar','rpm':'RPM','current':'A'}.get(stype,'')
                        signals_for_ai.append(f"Significant Anomaly: {stype.title()} spiked to {val:.1f}{unit} ({dev['z_score']:.1f} standard deviations above normal)")
                        
                    # Find notable trends (e.g., dropping pressure)
                    for stype in installed_sensors:
                        trend     = features.get(f"{stype}_trend", 0)
                        unit      = {'temperature':'°C','vibration':'mm/s','pressure':'Bar','rpm':'RPM','current':'A'}.get(stype,'')
                        avg_val   = features.get(f"{stype}_avg", 1) # Prevent div by zero
                        
                        # Trend magnitude relative to the average
                        if abs(trend) / avg_val > 0.05: # >5% shift across the time window
                            direction = 'Rising' if trend > 0 else 'Dropping'
                            signals_for_ai.append(f"Trend Warning: {stype.title()} is {direction} rapidly ({trend:+.1f}{unit})")

                # FORCE analysis if risk is detected
                if signals_for_ai or risk_level != "Healthy":
                    asset_md = {
                        "asset_type": selected_asset['asset_type'],
                        "industry": selected_asset['industry_name'],
                        "predicted_failure": f"{pred_label} ({detail})",
                        "diagnostic_version": "v1.2" # Final prompt enforcement bust
                    }
                    
                    # Fallback if no specific signals were picked up
                    signals_text = ", ".join(signals_for_ai) if signals_for_ai else "Subtle shifts in multiple operating sensors"
                    
                    with st.spinner("AI analyzing technical signals..."):
                        dynamic_insight = cached_get_signal_insights(
                            f"Risk Level: {risk_level}. Signals: {signals_text}", 
                            asset_md
                        )
                    
                    # Display as a professional cohesive alert
                    st.markdown(f"""
                    <div style="padding:10px; background:#fff; border-radius:8px; border:1px solid #e2e8f0; font-size:13px; color:#374151;">
                        <b style="color:#2563eb;">🧠 AI Diagnostic Report:</b><br>
                        {dynamic_insight}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show raw signals context
                    if signals_for_ai:
                        st.markdown("<br>", unsafe_allow_html=True)
                        cols_sig = st.columns(2)
                        for i, sig in enumerate(signals_for_ai[:6]):
                            cols_sig[i%2].caption(f"🔍 {sig}")
                else:
                    st.success("All sensor readings are within AI-defined normal ranges.", icon="✅")
                    
                # ── NEW: SERVICE & INTERVENTION HISTORY ──────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### Service & Intervention History")
                st.caption("Chronological audit trail of human and AI interventions.")
                
                # Fetch live data for this asset
                asset_work_orders = df_wo[df_wo['asset_id'] == asset_id].copy()
                
                if not asset_work_orders.empty:
                    # Robust parsing for mixed timestamp formats (with/without microseconds)
                    asset_work_orders['created_at'] = pd.to_datetime(asset_work_orders['created_at'], format='ISO8601')
                    asset_work_orders = asset_work_orders.sort_values('created_at', ascending=False)
                    
                    # ── SCROLLABLE HISTORY CONTAINER ──
                    with st.container(height=450):
                        for _, wo in asset_work_orders.iterrows():
                            status_color = "#2563eb" if wo['status'] == 'in_progress' else "#16a34a" if wo['status'] == 'completed' else "#6b7280"
                            status_icon = "👷" if wo['status'] == 'in_progress' else "✅" if wo['status'] == 'completed' else "📝"
                            
                            # Create a nested layout for the history item to allow a button on the right
                            h_col1, h_col2 = st.columns([5, 1])
                            
                            with h_col1:
                                st.markdown(f"""
                                <div style="border-left: 3px solid {status_color}; padding-left: 15px; margin-bottom: 20px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span style="font-weight: 700; font-size: 14px; color: #111827;">{wo['title']}</span>
                                        <span style="font-size: 11px; color: {status_color}; font-weight: 700;">{status_icon} {wo['status'].upper()}</span>
                                    </div>
                                    <div style="font-size: 12px; color: #4b5563; margin-top: 4px;">{wo['description']}</div>
                                    <div style="display: flex; gap: 15px; margin-top: 8px;">
                                        <div style="font-size: 11px; color: #6b7280;"><b>BY:</b> {wo['assigned_to'] or 'Unassigned'}</div>
                                        <div style="font-size: 11px; color: #6b7280;"><b>ON:</b> {wo['created_at'].strftime('%Y-%m-%d')}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with h_col2:
                                if wo['status'] == 'in_progress':
                                    # Shorter text and tighter width
                                    if st.button("📢 Alert", key=f"resend_alert_{wo['id']}", use_container_width=False, help="Trigger a priority email alert for this intervention."):
                                        with st.spinner("Dispatching..."):
                                            # We use the scoped variables from the Asset Monitor section
                                            p_cost = risk_assessment.get("predicted_cost", 0)
                                            p_rul  = risk_assessment.get("predicted_days_to_failure", "N/A")
                                            
                                            success, msg = send_email_via_sendgrid(
                                                asset_name=selected_asset['name'],
                                                tech_recommendation=f"Immediate action for '{wo['title']}'. Assigned to {wo['assigned_to']}.",
                                                risk_level=risk_level,
                                                asset_id=asset_id,
                                                rul=p_rul,
                                                liability=p_cost
                                            )
                                            if success:
                                                st.success("✅ Priority Alert Dispatched Successfully!")
                                                st.toast("Email sent.")
                                            else:
                                                st.error(f"❌ {msg}")
                else:
                    st.caption("No historical maintenance records found for this asset.")


            except Exception as e:
                st.error(f"AI Error: {e}")
        else:
            st.warning("Sensor data not available. Ensure that the selected time range covers active asset data.")






# ─────────────────────────────────────────────
# SECTION 3: PREDICTIVE LAB
# ─────────────────────────────────────────────

elif selected == "Cost Prediction":
    st.markdown("### Preventive Cost Simulator")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Scenario Configuration")
            st.caption("Simulate the cost difference between **Planned Maintenance** vs. **Emergency Repairs**:")
            
            # Priority (Hypothetical)
            priority_options = ["low", "medium", "high", "critical"]
            priority_labels = {
                "low": "Planned (Low Urgency)",
                "medium": "Standard Repair",
                "high": "Urgent Response",
                "critical": "🚨 Emergency"
            }
            
            priority_display = st.select_slider(
                "Urgency of Repair", 
                options=priority_options,
                value="medium",
                format_func=lambda x: priority_labels[x],
                help="Emergency repairs cost 3x more due to overtime and expedited shipping."
            )
            
            # Age (Years -> Days)
            age_years = st.slider(
                "Asset Age (Years)", 
                min_value=0.0, max_value=20.0, value=3.0, step=0.5,
                help="Older assets require more expensive parts."
            )
            age_days = int(age_years * 365)
            
            # Complexity (Scale 1-10 -> Description Length)
            complexity_scale = st.slider(
                "Repair Complexity (Scale 1-10)", 
                min_value=1, max_value=10, value=3,
                help="1 = Simple filter change. 10 = Complete engine overhaul."
            )
            # Map 1-10 scale to description length (proxy for complexity in model)
            # 1 -> 10 chars, 10 -> 200 chars
            complexity_len = int(np.interp(complexity_scale, [1, 10], [10, 200]))
            
            asset_type = st.selectbox("Asset Class", df_assets['asset_type'].unique())
            
            is_warranty_active = st.toggle("Asset Under Active Warranty", value=False)
            
        with c2:
            st.markdown("##### AI Cost Prediction")
            st.caption("Estimated bill based on historical invoice data:")
            
            if st.button("Calculate Impact", type="primary"):
                # Predict
                if cost_model:
                    input_df = pd.DataFrame([{
                        'priority': priority_display, # Use the raw value (low/med/high)
                        'asset_type': asset_type,
                        'industry': "Manufacturing", # default baseline
                        'asset_age_days': age_days,
                        'description_len': complexity_len
                    }])
                    pred_cost = float(np.expm1(cost_model.predict(input_df)[0]))
                    
                    # ---------------------------------------------------------
                    # NEW: Calculate Real Math (Stop AI Hallucinations)
                    # ---------------------------------------------------------
                    purchase_price = {
                        "Robotic Arm": 150000,
                        "Conveyor System": 85000,
                        "Centrifuge": 250000,
                        "Industrial Boiler": 120000,
                        "Delivery Truck": 65000,
                        "Wind Turbine": 2000000,
                        "CNC Machine": 180000
                    }.get(asset_type, 100000) # Default baseline
                    
                    useful_life_years = 15.0
                    # Straight-line depreciation, floor at 10% salvage value
                    residual_value = max(purchase_price * 0.10, purchase_price * (1.0 - (age_years / useful_life_years)))
                    repair_ratio = (pred_cost / residual_value) * 100
                    
                    if is_warranty_active:
                        pred_cost = 0.0
                        repair_ratio = 0.0
                        
                        # Apply 85% discount for parts coverage; remaining 15% covers labor, logistics, and downtime
                        pred_cost = pred_cost * 0.15
                        repair_ratio = (pred_cost / residual_value) * 100 if residual_value > 0 else 0
                        
                    st.metric("Estimated Cost Impact (Discounted)", f"${pred_cost:,.2f}")
                    
                    # ---------------------------------------------------------
                    # NEW: AI STRATEGIC ADVICE (LLM)
                    # ---------------------------------------------------------
                    with st.spinner("Analyzing repair vs. replace strategy..."):
                        advice_data = {
                            "asset_type": asset_type,
                            "age_years": age_years,
                            "predicted_cost": pred_cost,
                            "residual_value": residual_value,
                            "repair_ratio": repair_ratio,
                            "under_warranty": is_warranty_active,
                            "priority": priority_labels[priority_display], 
                            "complexity": complexity_scale
                        }
                        recommendation = cached_get_maintenance_recommendation(advice_data)
                        
                        # Final Display: Combine LLM Advice + System Insight
                        
                        # Insight Logic determining Color & Context
                        if priority_display == "critical":
                            st.error(f"**Advice:**\n\n{recommendation}\n\n**Key Driver:** ⚠️ CRITICAL PRIORITY increases labor costs by ~300%.")
                        elif age_years > 8:
                            st.warning(f"**Advice:**\n\n{recommendation}\n\n**Key Driver:** ⚠️ High asset age (>8yr) correlates with 20% higher parts costs.")
                        elif complexity_scale > 7:
                            st.info(f"**Advice:**\n\n{recommendation}\n\n**Key Driver:** ℹ️ High complexity suggests specialized labor.")
                        else:
                            st.success(f"**Advice:**\n\n{recommendation}\n\n**Key Driver:** ✅ Standard maintenance parameters.")

# ─────────────────────────────────────────────
# SECTION 4: PERSISTENT AI ASSISTANT WIDGET
# ─────────────────────────────────────────────

import streamlit.components.v1 as components

# CSS for the popover button styling + chat window
st.markdown("""
<style>
/* ── Pulse glow animation for the FAB ── */
@keyframes pulseGlow {
    0% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
    50% { box-shadow: 0 4px 25px rgba(102, 126, 234, 0.7), 0 0 40px rgba(118, 75, 162, 0.3); }
    100% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
}

/* ── Circular FAB Button ── */
div[data-testid="stPopover"] button[data-testid="baseButton-secondary"] {
    width: 64px !important;
    height: 64px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: transparent !important; /* Hide the emoji text */
    border: 3px solid rgba(255,255,255,0.25) !important;
    animation: pulseGlow 2.5s ease-in-out infinite !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 0 !important;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease !important;
    min-width: 64px !important;
    max-width: 64px !important;
    cursor: pointer !important;
    position: relative !important;
    /* SVG chat bubble icon as background */
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h14l4 4V4c0-1.1-.9-2-2-2zm0 15.17L18.83 16H4V4h16v13.17zM7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 30px 30px !important;
}

div[data-testid="stPopover"] button[data-testid="baseButton-secondary"]:hover {
    transform: scale(1.15) !important;
    animation: none !important;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.7) !important;
    border-color: rgba(255,255,255,0.5) !important;
}

div[data-testid="stPopover"] button[data-testid="baseButton-secondary"]:active {
    transform: scale(0.95) !important;
}

/* Hide the emoji text completely */
div[data-testid="stPopover"] button[data-testid="baseButton-secondary"] p {
    font-size: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
    visibility: hidden !important;
}

/* ── Notification Badge ── */
div[data-testid="stPopover"] button[data-testid="baseButton-secondary"]::after {
    content: '';
    position: absolute;
    top: 2px;
    right: 2px;
    width: 14px;
    height: 14px;
    background: #ef4444;
    border-radius: 50%;
    border: 2px solid white;
    box-shadow: 0 2px 6px rgba(239, 68, 68, 0.5);
    animation: statusPulse 2s ease-in-out infinite;
}

/* ── Popover Body (Chat Window) ── */
div[data-testid="stPopoverBody"] {
    width: 350px !important;
    max-width: 85vw !important;
    max-height: 50vh !important;
    border-radius: 16px !important;
    box-shadow: 0 15px 40px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.05) !important;
    padding: 0 !important;
    overflow: auto !important;
}

/* Kill internal Streamlit padding wrappers inside popover */
div[data-testid="stPopoverBody"] > div {
    padding: 0 !important;
}

div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── Chat Header ── */
.chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 14px 18px;
    font-weight: 600;
    font-size: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    border-radius: 0;
}

.chat-header .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.chat-header .status-dot {
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px rgba(74, 222, 128, 0.6);
    animation: statusPulse 2s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.chat-header .status-badge {
    font-size: 10px;
    font-weight: 500;
    background: rgba(255,255,255,0.2);
    padding: 3px 8px;
    border-radius: 10px;
    letter-spacing: 0.3px;
}

/* ── Chat Input ── */
div[data-testid="stPopoverBody"] div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0px;
    background: white;
    padding: 10px 12px;
    border-top: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# The native popover button acting as the circular Chat Icon
with st.popover("", icon=":material/chat:"):
    st.markdown('''
        <div class="chat-header">
            <div class="header-left">
                <span class="status-dot"></span>
                <span>AI Maintenance Assistant</span>
            </div>
            <span class="status-badge">Online</span>
        </div>
    ''', unsafe_allow_html=True)
    
    # Message scroll area
    messages_container = st.container(height=200, border=False)
    
    with messages_container:
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    

    if prompt := st.chat_input("Ask about assets, costs, or maintenance logs..."):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            chat_predicted_assets = batch_predict_assets(df_assets)
            chat_predicted_df = pd.DataFrame(chat_predicted_assets)
            
            if not chat_predicted_df.empty:
                chat_predicted_df = chat_predicted_df.sort_values('risk_score', ascending=False)
                chat_predicted_df['urgent_repair_cost'] = chat_predicted_df['predicted_cost'] * 0.40
                chat_predicted_df['extra_delay_penalty'] = chat_predicted_df['predicted_cost'] * 0.60
                critical_list = chat_predicted_df[['id', 'name', 'industry_name', 'risk_level', 'risk_score', 'predicted_cost', 'urgent_repair_cost', 'extra_delay_penalty']].head(30).to_dict('records')
                all_assets_mapping = chat_predicted_df[['id', 'name', 'risk_level']].to_dict('records')
            else:
                critical_list = "No critical assets"
                all_assets_mapping = []
                
            active_wos = df_wo[df_wo['status'] == 'in_progress']
            priority_wos = active_wos[active_wos['priority'].isin(['critical', 'high'])]
            priority_wos = priority_wos.sort_values(by=['priority', 'created_at'], ascending=[True, False]).head(15)
            active_wos_list = priority_wos[['id', 'asset_id', 'title', 'priority', 'status', 'assigned_to']].to_dict('records')
            
            chat_liability = chat_predicted_df['predicted_cost'].sum() if not chat_predicted_df.empty else 0
            chat_savings = chat_liability * 0.60
            predictive_df = chat_predicted_df[chat_predicted_df['risk_level'].isin(['Critical', 'High Risk'])]
            preventive_df = chat_predicted_df[chat_predicted_df['risk_level'] == 'Warning']
            predict_savings = predictive_df['predicted_cost'].sum() * 0.60 if not predictive_df.empty else 0
            prevent_savings = preventive_df['predicted_cost'].sum() * 0.60 if not preventive_df.empty else 0
            
            fleet_metadata = {
                "available_industries": df_assets['industry_name'].unique().tolist(),
                "available_asset_types": df_assets['asset_type'].unique().tolist(),
                "monitored_assets_mapping": all_assets_mapping
            }
            
            context_payload = {
                "fleet_metadata": fleet_metadata,
                "total_active_work_orders_count": len(active_wos),
                "total_fleet_predicted_liability": chat_liability,
                "total_fleet_cost_avoidance": chat_savings,
                "predictive_maintenance_savings": predict_savings,
                "preventive_maintenance_savings": prevent_savings,
                "critical_assets": critical_list,
                "priority_work_orders": active_wos_list if not priority_wos.empty else "No active priority work orders"
            }
            
            ai_reply = generate_response(prompt, context=context_payload)
        except Exception as e:
            ai_reply = f"🔌 Connection or processing error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        st.rerun()

# JavaScript injection to force the popover element to float at bottom-right.
# Uses individual style.setProperty() to avoid destroying Streamlit's styles.
# MutationObserver is debounced and only watches childList (not attributes).
components.html("""
<script>
(function() {
    const doc = window.parent.document;
    let debounceTimer = null;
    
    function floatPopover() {
        const popovers = doc.querySelectorAll('div[data-testid="stPopover"]');
        if (popovers.length === 0) return;
        
        const chatPopover = popovers[popovers.length - 1];
        
        // Use setProperty instead of cssText to preserve Streamlit's existing styles
        chatPopover.style.setProperty('position', 'fixed', 'important');
        chatPopover.style.setProperty('bottom', '80px', 'important');
        chatPopover.style.setProperty('right', '30px', 'important');
        chatPopover.style.setProperty('z-index', '999999', 'important');
        chatPopover.style.setProperty('width', 'auto', 'important');
        
        // Walk up ancestors and fix any overflow:hidden that clips the fixed element
        let parent = chatPopover.parentElement;
        let depth = 0;
        while (parent && parent !== doc.body && depth < 15) {
            const cs = window.parent.getComputedStyle(parent);
            if (cs.overflow === 'hidden' || cs.overflowX === 'hidden' || cs.overflowY === 'hidden') {
                parent.style.setProperty('overflow', 'visible', 'important');
            }
            parent = parent.parentElement;
            depth++;
        }
    }
    
    // Run on initial load with staggered delays
    floatPopover();
    setTimeout(floatPopover, 300);
    setTimeout(floatPopover, 800);
    setTimeout(floatPopover, 1500);
    setTimeout(floatPopover, 3000);
    
    // Debounced MutationObserver - only watches childList (NOT attributes)
    // This prevents infinite loops when we modify styles
    const observer = new MutationObserver(function() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(floatPopover, 100);
    });
    
    observer.observe(doc.body, {
        childList: true,
        subtree: true
        // NO attributes: true — that would cause infinite loops
    });
})();
</script>
""", height=0, width=0)

