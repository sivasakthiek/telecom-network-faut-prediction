import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import datetime
import dotenv
dotenv.load_dotenv()
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import zipfile
import io

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_merge, clean_data
from src.features import build_features
from src.predictor import get_location_freq_map, explain, predict
from src.health_score import compute_health_score
from src.recommender import recommend_action
from src.rag_service import RAGService
from src.llm_service import LLMService

# ==============================================================================
# CENTRAL COLOR SYSTEM CONSTANTS
# ==============================================================================
COLOR_CRITICAL = "#FF3B3B"  # Saturated, high-contrast urgent red
COLOR_WARNING  = "#ffba20"  # Amber
COLOR_HEALTHY  = "#00dbe9"  # Cyan

COLOR_BG_PAGE = "#111317"
COLOR_SURFACE = "#1e2024"
COLOR_BORDER  = "#3b494b"
COLOR_TEXT_PRI= "#e2e2e8"
COLOR_TEXT_MUT= "#b9cacb"

# Streamlit Page Config
st.set_page_config(
    page_title="Telecom Network Intelligence - NOC Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for NOC Command Center Theme
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');/* Global Reset & Background */
html, body, .stApp {{
    background-color: {COLOR_BG_PAGE} !important;
    color: {COLOR_TEXT_PRI} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Geist', sans-serif !important;
    font-weight: 600 !important;
    color: {COLOR_TEXT_PRI} !important;
    letter-spacing: -0.02em;
}}

.stMarkdown, p, label {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {COLOR_TEXT_MUT};
}}

/* Custom Header Styling */
.noc-header-title {{
    font-family: 'Geist', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRI};
    margin-bottom: 4px;
}}

.noc-header-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {COLOR_TEXT_MUT};
    margin-bottom: 16px;
}}

/* Compact Status Legend */
.legend-container {{
    display: flex;
    gap: 20px;
    align-items: center;
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 8px 16px;
    width: fit-content;
    margin-bottom: 24px;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 500;
}}

.dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}

.dot.critical {{ background-color: {COLOR_CRITICAL}; box-shadow: 0 0 8px rgba(255, 59, 59, 0.6); }}
.dot.warning {{ background-color: {COLOR_WARNING}; box-shadow: 0 0 8px rgba(255, 186, 32, 0.6); }}
.dot.healthy {{ background-color: {COLOR_HEALTHY}; box-shadow: 0 0 8px rgba(0, 219, 233, 0.6); }}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background-color: {COLOR_SURFACE} !important;
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}}

div[data-testid="stMetricLabel"] > label {{
    color: {COLOR_TEXT_MUT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}

div[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}

/* Panel Containers */
.noc-panel {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 20px;
    margin-top: 16px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}}

.noc-panel-title {{
    font-family: 'Geist', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {COLOR_TEXT_PRI};
    margin-bottom: 6px;
}}

.noc-panel-subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: {COLOR_TEXT_MUT};
    margin-bottom: 16px;
}}

/* Device Incident Card Styling */
.device-card-grid {{
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 12px;
}}

.device-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: transform 0.15s ease;
}}

.device-card:hover {{
    border-color: {COLOR_HEALTHY};
}}

.device-card.critical {{ border-left: 5px solid {COLOR_CRITICAL} !important; }}
.device-card.warning {{ border-left: 5px solid {COLOR_WARNING} !important; }}
.device-card.healthy {{ border-left: 5px solid {COLOR_HEALTHY} !important; }}

.device-info {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.device-id {{
    font-family: 'Geist', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRI};
}}

.device-loc {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {COLOR_TEXT_MUT};
}}

.device-score-container {{
    display: flex;
    align-items: center;
    gap: 20px;
}}

.device-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
}}

.device-score.critical {{ color: {COLOR_CRITICAL}; }}
.device-score.warning {{ color: {COLOR_WARNING}; }}
.device-score.healthy {{ color: {COLOR_HEALTHY}; }}

.status-pill {{
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.status-pill.critical {{
    background-color: rgba(255, 59, 59, 0.15);
    color: {COLOR_CRITICAL};
    border: 1px solid rgba(255, 59, 59, 0.3);
}}
.status-pill.warning {{
    background-color: rgba(255, 186, 32, 0.15);
    color: {COLOR_WARNING};
    border: 1px solid rgba(255, 186, 32, 0.3);
}}
.status-pill.healthy {{
    background-color: rgba(0, 219, 233, 0.15);
    color: {COLOR_HEALTHY};
    border: 1px solid rgba(0, 219, 233, 0.3);
}}

/* Custom text input styling */
.stTextInput > div > div > input {{
    background-color: #1e2024 !important;
    color: #e2e2e8 !important;
    border: 1px solid #3b494b !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: #00dbe9 !important;
    box-shadow: 0 0 8px rgba(0, 219, 233, 0.3) !important;
}}

/* Selectbox and Multiselect Styling */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background-color: #1e2024 !important;
    color: #e2e2e8 !important;
    border: 1px solid #3b494b !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
div[data-baseweb="select"] {{
    background-color: #1e2024 !important;
    border-radius: 6px !important;
}}
div[data-baseweb="tag"] {{
    background-color: rgba(0, 219, 233, 0.15) !important;
    border: 1px solid rgba(0, 219, 233, 0.4) !important;
    color: #00dbe9 !important;
    border-radius: 4px !important;
}}
.stNumberInput > div > div > input {{
    background-color: #1e2024 !important;
    color: #e2e2e8 !important;
    border: 1px solid #3b494b !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Tab styling */
button[data-baseweb="tab"] {{
    font-family: 'Geist', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #b9cacb !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: #00dbe9 !important;
    border-bottom-color: #00dbe9 !important;
}}

/* Custom button styling */
.stButton > button {{
    background-color: #1e2024 !important;
    color: #e2e2e8 !important;
    border: 1px solid #3b494b !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    border-color: #00dbe9 !important;
    color: #00dbe9 !important;
    background-color: rgba(0, 219, 233, 0.05) !important;
}}

/* Align inspect button cleanly with device-card height */
.inspect-btn-container .stButton > button {{
    height: 68px !important;
    margin-top: 12px !important;
}}

/* Scoped Dispatch Tech CTA button styling */
.dispatch-container .stButton > button {{
    background-color: #00dbe9 !important;
    color: #111317 !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'Geist', sans-serif !important;
    box-shadow: 0 0 12px rgba(0, 219, 233, 0.4) !important;
    height: 48px !important;
    font-size: 1rem !important;
    margin-top: 10px !important;
}}
.dispatch-container .stButton > button:hover {{
    background-color: #00f3ff !important;
    box-shadow: 0 0 20px rgba(0, 219, 233, 0.7) !important;
    color: #111317 !important;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_test_data(num_samples: int = 100):
    """
    Loads real raw data, extracts 100 test split records, and computes predictions + health scores.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
    
    raw_df = load_and_merge(data_dir)
    
    _, test_raw = train_test_split(
        raw_df, test_size=0.20, random_state=42, stratify=raw_df['fault_severity']
    )
    
    sample_records = test_raw.head(num_samples).copy()
    
    loc_map = get_location_freq_map(models_dir=os.path.dirname(model_path), data_dir=data_dir)
    model = joblib.load(model_path)
    
    X_clean, _, _ = clean_data(sample_records, location_freq_map=loc_map)
    X_feat = build_features(X_clean)
    
    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in X_feat.columns:
            X_feat[col] = 0
    X_feat = X_feat[expected_cols]
    
    probs = model.predict_proba(X_feat)
    preds = model.predict(X_feat)
    
    records_list = []
    for i in range(len(sample_records)):
        row = sample_records.iloc[i]
        p = probs[i].tolist()
        pred_c = int(preds[i])
        score, status = compute_health_score(p)
        
        records_list.append({
            'id': int(row['id']),
            'location': str(row['location']),
            'severity_type': str(row['severity_type']),
            'fault_severity': int(row['fault_severity']),
            'predicted_class': pred_c,
            'p0': p[0],
            'p1': p[1],
            'p2': p[2],
            'health_score': score,
            'status': status
        })
        
    return pd.DataFrame(records_list)


# 1. Load Real Processed Test Data & Raw Records
df_results = load_and_process_test_data(num_samples=100)

@st.cache_data
def load_raw_test_records(num_samples: int = 100) -> dict:
    """
    Loads raw test records in wide format, returns them mapped by device ID.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    raw_df = load_and_merge(data_dir)
    _, test_raw = train_test_split(
        raw_df, test_size=0.20, random_state=42, stratify=raw_df['fault_severity']
    )
    sample_records = test_raw.head(num_samples).copy()
    sample_records['id'] = sample_records['id'].astype(int)
    return sample_records.set_index('id').to_dict(orient='index')

raw_records = load_raw_test_records(num_samples=100)


def translate_feature_name(name: str) -> str:
    """
    Translates raw model feature names into human-readable signal names.
    """
    # 1. Total volume
    if name == 'total_volume':
        return "Overall event volume"
    # 2. Severity type
    if name == 'severity_type':
        return "Reported severity classification"
    # 3. Location
    if name == 'location':
        return "Location risk frequency"
    # 4. Summary counts
    if name == 'num_active_log_features':
        return "Number of distinct log signals involved"
    if name == 'num_event_types':
        return "Number of distinct event trigger types involved"
    if name == 'num_resource_types':
        return "Number of distinct resource types involved"
    
    # 5. Regex mappings for log, event, resource
    import re
    # Match log_log_feature N or log_feature N
    m_log = re.match(r'^(?:log_)?log_feature\s+(\d+)$', name)
    if m_log:
        return f"Log signal #{m_log.group(1)} (elevated volume)"
    
    m_event = re.match(r'^(?:event_)?event_type\s+(\d+)$', name)
    if m_event:
        return f"Event trigger type #{m_event.group(1)}"
        
    m_resource = re.match(r'^(?:resource_)?resource_type\s+(\d+)$', name)
    if m_resource:
        return f"Resource type #{m_resource.group(1)} affected"
        
    return name


def execute_pandas_query(route: dict, df: pd.DataFrame, prompt: str = "") -> str:
    prompt_lower = prompt.lower()
    
    # Check if user query is asking to list specific device types
    if "list" in prompt_lower or "show" in prompt_lower or "display" in prompt_lower or "find" in prompt_lower:
        if "critical" in prompt_lower:
            crit_df = df[df['status'] == 'Critical']
            if crit_df.empty:
                return "There are no critical devices currently loaded in the dashboard."
            device_list = ", ".join([f"#{dev_id} (Health: {score:.1f}%)" for dev_id, score in zip(crit_df['id'], crit_df['health_score'])])
            return f"**Critical Devices in Dashboard:**\n{device_list}"
        elif "warning" in prompt_lower:
            warn_df = df[df['status'] == 'Warning']
            if warn_df.empty:
                return "There are no warning devices currently loaded in the dashboard."
            device_list = ", ".join([f"#{dev_id} (Health: {score:.1f}%)" for dev_id, score in zip(warn_df['id'], warn_df['health_score'])])
            return f"**Warning Devices in Dashboard:**\n{device_list}"
        elif "healthy" in prompt_lower:
            health_df = df[df['status'] == 'Healthy']
            if health_df.empty:
                return "There are no healthy devices currently loaded in the dashboard."
            device_list = ", ".join([f"#{dev_id} (Health: {score:.1f}%)" for dev_id, score in zip(health_df['id'], health_df['health_score'])])
            return f"**Healthy Devices in Dashboard:**\n{device_list}"

    loc = route.get("location")
    agg = route.get("aggregation")
    metric = route.get("metric")
    
    # If location is specified, filter by location
    if loc:
        loc_df = df[df['location'].str.lower() == loc.lower()]
        if loc_df.empty:
            return f"Location '{loc}' was not found in the currently active dashboard samples."
            
        total_incidents = len(loc_df)
        total_faults = int((loc_df['fault_severity'] > 0).sum())
        minor_faults = int((loc_df['fault_severity'] == 1).sum())
        major_faults = int((loc_df['fault_severity'] == 2).sum())
        avg_health = loc_df['health_score'].mean()
        worst_severity = int(loc_df['fault_severity'].max())
        
        return (
            f"**Query Results for {loc}:**\n"
            f"- **Total monitored incidents**: {total_incidents}\n"
            f"- **Total active network faults**: {total_faults} (Minor: {minor_faults}, Major: {major_faults})\n"
            f"- **Worst recorded fault severity level**: {worst_severity}\n"
            f"- **Average device health score**: {avg_health:.1f}%"
        )
    
    # General queries (no specific location)
    total_incidents = len(df)
    total_faults = int((df['fault_severity'] > 0).sum())
    minor_faults = int((df['fault_severity'] == 1).sum())
    major_faults = int((df['fault_severity'] == 2).sum())
    avg_health = df['health_score'].mean()
    avg_severity = df['fault_severity'].mean()
    
    if agg == "sum":
        return (
            f"**Network Aggregates (Sum):**\n"
            f"- **Total monitored incidents**: {total_incidents}\n"
            f"- **Total active network faults**: {total_faults}\n"
            f"  - *Minor faults (Warning)*: {minor_faults}\n"
            f"  - *Major faults (Critical)*: {major_faults}"
        )
    elif agg == "max":
        worst_loc_row = df.groupby('location')['health_score'].mean().idxmin()
        worst_loc_score = df.groupby('location')['health_score'].mean().min()
        return (
            f"**Network Aggregates (Max/Worst):**\n"
            f"- **Max fault severity level**: {int(df['fault_severity'].max())}\n"
            f"- **Worst performing location**: {worst_loc_row} (Avg Health: {worst_loc_score:.1f}%)"
        )
    elif agg == "mean" or agg == "average":
        return (
            f"**Network Aggregates (Average):**\n"
            f"- **Average network health score**: {avg_health:.1f}%\n"
            f"- **Average fault severity index**: {avg_severity:.2f}"
        )
    else:
        return (
            f"**Dashboard Data Summary:**\n"
            f"- **Total monitored incidents**: {total_incidents}\n"
            f"- **Total active network faults**: {total_faults} (Minor: {minor_faults}, Major: {major_faults})\n"
            f"- **Average device health score**: {avg_health:.1f}%\n"
            f"- **Average fault severity index**: {avg_severity:.2f}"
        )


def process_chat_query(prompt: str, active_device_context: str = None) -> str:
    # Always prefer env var if available and valid to sync session state
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    if env_key and env_key != "your_api_key_here" and len(env_key.strip()) > 0:
        st.session_state.openrouter_api_key = env_key
        
    api_key = st.session_state.get("openrouter_api_key", "")
    
    # Initialize LLMService with the active API key
    llm_srv = LLMService(api_key=api_key)
            
    # 1. Route Query
    route = llm_srv.route_query(prompt)
    
    if not route.get("is_network_related", True):
        return "I can only answer questions based on the information available in this dashboard."
        
    if route.get("query_type") == "numerical":
        return execute_pandas_query(route, df_results, prompt)
        
    # Semantic/RAG query
    try:
        rag_srv = RAGService()
    except Exception as e:
        return f"Failed to initialize RAG search service: {e}"
        
    # Decide context based on routing
    if route.get("query_type") == "what_if":
        from src.what_if_engine import WhatIfEngine
        engine = WhatIfEngine()
        
        dev_id = None
        display_id = None
        
        # Extract display format from prompt (e.g. TN1045)
        match_tn = re.search(r'\b(tn\s*#?\s*\d+)\b', prompt, re.IGNORECASE)
        if match_tn:
            display_id = match_tn.group(1).upper()
            
        if route.get("target_device_id"):
            dev_id = route["target_device_id"]
            if not display_id:
                display_id = str(dev_id)
        elif route.get("refers_to_active_device", False) or active_device_context:
            if st.session_state.get("selected_device_id"):
                dev_id = st.session_state.selected_device_id
            elif active_device_context:
                match = re.search(r'Device ID:\s*(\d+)', active_device_context)
                if match:
                    dev_id = int(match.group(1))
            if dev_id and not display_id:
                display_id = f"Device #{dev_id}"
                
        if not dev_id:
            return "Please specify a device ID (e.g. 'What if Device 15086 remains unresolved?') or inspect a device first to ask about it."
            
        what_if_res = engine.analyze(dev_id, df_results=df_results, raw_records=raw_records)
        if not what_if_res:
            return f"Device #{display_id} was not found in the historical network records."
            
        what_if_context = (
            f"Device ID: {display_id}\n"
            f"Current Condition: {what_if_res['current_condition']}\n"
            f"Risk Level: {what_if_res['risk_level']}\n"
            f"Possible Progression: {what_if_res['possible_progression']}\n"
            f"Potential Network & Customer Impact: {what_if_res['potential_impact']}\n"
            f"Recommended Priority & Action: {what_if_res['recommended_priority']}\n"
            f"Evidence & Confidence Level: {what_if_res['confidence_level']}\n"
            f"Estimated Time-to-Impact: {what_if_res['time_to_impact']}"
        )
        
        # Reuse existing device summary from dict lookup (no new FAISS search)
        dict_res = rag_srv.lookup_device_id(dev_id)
        context_records = [dict_res] if dict_res else []
        
        return llm_srv.generate_answer(prompt, context_records, active_device_context, what_if_context=what_if_context)
        
    elif route.get("refers_to_active_device", False):
        if active_device_context:
            context_records = []
            return llm_srv.generate_answer(prompt, context_records, active_device_context)
        else:
            return "You are not currently inspecting a specific device. Please inspect a device first to ask about 'it' or 'this device'."
            
    elif route.get("target_device_id"):
        tgt_id = route["target_device_id"]
        dict_res = rag_srv.lookup_device_id(tgt_id)
        if dict_res:
            context_records = [dict_res]
            return llm_srv.generate_answer(prompt, context_records, active_device_context)
        else:
            # Reconstruct original alphanumeric prefix if present in prompt
            orig_id = str(tgt_id)
            match_tn_p = re.search(r'\b(tn\s*#?\s*' + str(tgt_id) + r')\b', prompt, re.IGNORECASE)
            if match_tn_p:
                orig_id = match_tn_p.group(1).upper()
            return f"Device #{orig_id} was not found in the historical network records."
    else:
        # Standard location-level search
        context_records = rag_srv.retrieve(prompt, k=4)
        return llm_srv.generate_answer(prompt, context_records, active_device_context)


def render_simulator_tab():
    st.markdown(f"""
    <div class="noc-panel" style="margin-bottom: 20px;">
        <div class="noc-panel-title">⚡ Interactive Custom Telemetry & XGBoost Fault Predictor</div>
        <div class="noc-panel-subtitle">Input real-time or hypothetical network telemetry attributes to execute live XGBoost fault severity inference, probability breakdown, and SHAP explainability.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieve pre-computed location map
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
    models_dir = os.path.dirname(model_path)
    loc_map = get_location_freq_map(models_dir=models_dir)
    
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
        
    known_locations = sorted(list(loc_map.keys()), key=natural_sort_key)
    if not known_locations:
        known_locations = [f"location {i}" for i in range(1, 100)]
        
    col_form, col_pred = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("""
        <div style="font-family: 'Geist', sans-serif; font-size: 1.1rem; font-weight: 600; color: #e2e2e8; margin-bottom: 12px; border-bottom: 1px solid #3b494b; padding-bottom: 6px;">
            1. Telemetry Input Configuration
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Location (Selectbox + Custom Textbox Fallback)
        use_custom_loc = st.checkbox("Enter custom / unlisted location name", key="sim_custom_loc_toggle")
        if use_custom_loc:
            selected_location = st.text_input(
                "Custom Network Location", 
                value="location 9999", 
                placeholder="e.g. location 9999", 
                key="sim_loc_text",
                help="Custom or unseen locations will utilize the baseline network frequency prior."
            )
        else:
            default_loc_idx = known_locations.index('location 821') if 'location 821' in known_locations else 0
            selected_location = st.selectbox(
                "Network Location (Monitored Node)", 
                options=known_locations, 
                index=default_loc_idx, 
                key="sim_loc_select",
                help="Select one of 929 monitored network nodes."
            )
            
        # 2. Reported Severity Type (Categorical dropdown)
        severity_options = ["severity_type 1", "severity_type 2", "severity_type 3", "severity_type 4", "severity_type 5"]
        selected_severity = st.selectbox(
            "Reported Severity Type", 
            options=severity_options, 
            index=1, 
            key="sim_severity_select",
            help="Initial log daemon severity tier classification (1 to 5)."
        )
        
        # 3. Resource Types (Multi-categorical dropdown)
        resource_options = [f"resource_type {i}" for i in range(1, 11)]
        selected_resources = st.multiselect(
            "Active Resource Types (Hardware / Software Triggers)",
            options=resource_options,
            default=["resource_type 2"],
            key="sim_resources_select",
            help="Specific infrastructure resource categories triggered by the event."
        )
        
        # 4. Event Types (Multi-categorical dropdown)
        event_options = [f"event_type {i}" for i in range(1, 55)]
        selected_events = st.multiselect(
            "Active Event Types (Log Triggers / Categories)",
            options=event_options,
            default=["event_type 11"],
            key="sim_events_select",
            help="Categorized signal alerts registered during this incident window."
        )
        
        # 5. Log Features & Dynamic Volume Inputs
        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; color: #e2e2e8; margin-top: 16px; margin-bottom: 4px;">
            Diagnostic Log Features & Signal Volumes
        </div>
        """, unsafe_allow_html=True)
        log_feature_options = [f"log_feature {i}" for i in range(1, 387)]
        selected_log_features = st.multiselect(
            "Select Active Log Signals",
            options=log_feature_options,
            default=["log_feature 203", "log_feature 312"],
            key="sim_log_features_select",
            help="Specific diagnostic signal features extracted from log files (up to 386 unique signals)."
        )
        
        log_volumes = {}
        if selected_log_features:
            st.markdown("<div style='font-size: 0.8rem; color: #b9cacb; margin-top: 6px; margin-bottom: 8px;'>Set Volume Intensity for each active log signal:</div>", unsafe_allow_html=True)
            subcols = st.columns(min(len(selected_log_features), 2))
            for idx, lf in enumerate(selected_log_features):
                with subcols[idx % len(subcols)]:
                    default_vol = 5 if lf == "log_feature 203" else 12
                    vol = st.number_input(
                        f"Volume for {lf}", 
                        min_value=1, 
                        max_value=10000, 
                        value=default_vol, 
                        step=1, 
                        key=f"sim_vol_{lf}"
                    )
                    log_volumes[lf] = int(vol)
        else:
            st.info("ℹ️ Select at least one log feature above to simulate telemetry.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("⚡ Refresh / Recalculate Prediction", key="btn_execute_prediction", type="primary", use_container_width=True)

    # Process Input & Predict
    with col_pred:
        st.markdown("""
        <div style="font-family: 'Geist', sans-serif; font-size: 1.1rem; font-weight: 600; color: #e2e2e8; margin-bottom: 12px; border-bottom: 1px solid #3b494b; padding-bottom: 6px;">
            2. Real-Time XGBoost Diagnostic Results
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare input dictionary
        sim_record = {
            'location': str(selected_location).strip() if selected_location else 'location 1',
            'severity_type': selected_severity
        }
        for et in selected_events:
            sim_record[f"event_{et}"] = 1
        for rt in selected_resources:
            sim_record[f"resource_{rt}"] = 1
        for lf, vol in log_volumes.items():
            sim_record[lf] = vol
            
        # Run fast inference
        try:
            import time
            t_start = time.time()
            pred_class, probs = predict(sim_record, model_path=model_path, location_freq_map=loc_map)
            health_score, health_status = compute_health_score(probs)
            shap_factors = explain(sim_record, model_path=model_path, location_freq_map=loc_map) if health_score < 70 else []
            actions = recommend_action(shap_factors, health_status)
            inference_ms = (time.time() - t_start) * 1000.0
            
            # Status color mapping
            if health_status == 'Critical':
                status_color = COLOR_CRITICAL
                status_badge_text = "MAJOR FAULT DETECTED (Critical)"
                status_icon = "🔴"
            elif health_status == 'Warning':
                status_color = COLOR_WARNING
                status_badge_text = "MINOR FAULT DETECTED (Warning)"
                status_icon = "🟡"
            else:
                status_color = COLOR_HEALTHY
                status_badge_text = "NO FAULT (Normal Operation)"
                status_icon = "🟢"
                
            # 1. Prediction Status Banner
            st.markdown(f"""
            <div style="background-color: {status_color}15; border: 1px solid {status_color}55; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {COLOR_TEXT_MUT}; text-transform: uppercase; letter-spacing: 0.08em;">Predicted Severity Outcome</div>
                    <div style="font-family: 'Geist', sans-serif; font-size: 1.25rem; font-weight: 700; color: {status_color}; margin-top: 2px;">
                        {status_icon} Class {pred_class}: {status_badge_text}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {COLOR_TEXT_MUT};">Inference Latency</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; color: {COLOR_HEALTHY};">{inference_ms:.1f} ms</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Composite Health Score Gauge + Probability Breakdown
            col_g, col_prob = st.columns([1, 1])
            
            with col_g:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = health_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLOR_TEXT_MUT},
                        'bar': {'color': status_color},
                        'bgcolor': COLOR_SURFACE,
                        'borderwidth': 2,
                        'bordercolor': COLOR_BORDER,
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(255, 59, 59, 0.08)'},
                            {'range': [40, 70], 'color': 'rgba(255, 186, 32, 0.08)'},
                            {'range': [70, 100], 'color': 'rgba(0, 219, 233, 0.08)'}
                        ],
                    }
                ))
                fig_g.update_layout(
                    paper_bgcolor=COLOR_SURFACE,
                    plot_bgcolor=COLOR_SURFACE,
                    font={'color': COLOR_TEXT_PRI, 'family': "JetBrains Mono"},
                    height=170,
                    margin=dict(l=20, r=20, t=10, b=10)
                )
                st.plotly_chart(fig_g, use_container_width=True)
                st.markdown(f"<div style='text-align: center; font-size: 0.85rem; font-weight: 600; color: {status_color}; margin-top: -10px; margin-bottom: 12px;'>Composite Health Score: {health_score:.1f}% ({health_status})</div>", unsafe_allow_html=True)

            with col_prob:
                st.markdown(f"""
                <div class="noc-panel" style="margin-top: 0px; padding: 12px 16px; height: 195px;">
                    <div style="font-family: 'Geist', sans-serif; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: {COLOR_TEXT_PRI}; margin-bottom: 10px;">
                        Class Probabilities
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px; color: {COLOR_TEXT_MUT};">
                            <span>Class 0 (No Fault)</span>
                            <span style="color: {COLOR_HEALTHY}; font-weight: bold;">{probs[0]*100:.1f}%</span>
                        </div>
                        <div style="background-color: {COLOR_BG_PAGE}; border-radius: 3px; height: 6px; width: 100%;">
                            <div style="background-color: {COLOR_HEALTHY}; width: {probs[0]*100:.1f}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px; color: {COLOR_TEXT_MUT};">
                            <span>Class 1 (Minor Fault)</span>
                            <span style="color: {COLOR_WARNING}; font-weight: bold;">{probs[1]*100:.1f}%</span>
                        </div>
                        <div style="background-color: {COLOR_BG_PAGE}; border-radius: 3px; height: 6px; width: 100%;">
                            <div style="background-color: {COLOR_WARNING}; width: {probs[1]*100:.1f}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px; color: {COLOR_TEXT_MUT};">
                            <span>Class 2 (Major Fault)</span>
                            <span style="color: {COLOR_CRITICAL}; font-weight: bold;">{probs[2]*100:.1f}%</span>
                        </div>
                        <div style="background-color: {COLOR_BG_PAGE}; border-radius: 3px; height: 6px; width: 100%;">
                            <div style="background-color: {COLOR_CRITICAL}; width: {probs[2]*100:.1f}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # 3. Engineered Features Summary Row
            tot_vol = sum(log_volumes.values())
            n_logs = len(log_volumes)
            n_events = len(selected_events)
            n_res = len(selected_resources)
            
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
                <div style="background-color: #1e2024; border: 1px solid #3b494b; border-radius: 6px; padding: 10px 12px; text-align: center;">
                    <div style="font-size: 0.7rem; color: #b9cacb; text-transform: uppercase;">Total Volume</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #00dbe9; font-family: 'JetBrains Mono';">{tot_vol}</div>
                </div>
                <div style="background-color: #1e2024; border: 1px solid #3b494b; border-radius: 6px; padding: 10px 12px; text-align: center;">
                    <div style="font-size: 0.7rem; color: #b9cacb; text-transform: uppercase;">Log Signals</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #e2e2e8; font-family: 'JetBrains Mono';">{n_logs}</div>
                </div>
                <div style="background-color: #1e2024; border: 1px solid #3b494b; border-radius: 6px; padding: 10px 12px; text-align: center;">
                    <div style="font-size: 0.7rem; color: #b9cacb; text-transform: uppercase;">Event Types</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #e2e2e8; font-family: 'JetBrains Mono';">{n_events}</div>
                </div>
                <div style="background-color: #1e2024; border: 1px solid #3b494b; border-radius: 6px; padding: 10px 12px; text-align: center;">
                    <div style="font-size: 0.7rem; color: #b9cacb; text-transform: uppercase;">Resources</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #e2e2e8; font-family: 'JetBrains Mono';">{n_res}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Root Cause Analysis (SHAP XAI)
            st.markdown(f"""
            <div class="noc-panel" style="padding-bottom: 12px; margin-bottom: 16px;">
                <div class="noc-panel-title">Root Cause Analysis (SHAP Explainability)</div>
                <div class="noc-panel-subtitle">Top simulated input parameters contributing to the predicted risk profile.</div>
            """, unsafe_allow_html=True)
            
            if health_score >= 70 or not shap_factors:
                st.markdown(f"""
                <div style="background-color: rgba(0, 219, 233, 0.05); border: 1px solid rgba(0, 219, 233, 0.2); border-radius: 6px; padding: 12px; text-align: center; color: {COLOR_HEALTHY}; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                    Operating within safe baseline thresholds &bull; no critical anomaly triggers.
                </div>
                """, unsafe_allow_html=True)
            else:
                total_shap = sum(val for _, val in shap_factors if val > 0) or 1.0
                for name, val in shap_factors:
                    if val <= 0:
                        continue
                    impact_pct = (val / total_shap) * 100
                    tname = translate_feature_name(name)
                    st.markdown(f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; font-family: 'JetBrains Mono'; color: {COLOR_TEXT_PRI};">
                            <span>{tname}</span>
                            <span style="font-weight: bold; color: {status_color};">+{impact_pct:.1f}%</span>
                        </div>
                        <div style="background-color: {COLOR_BG_PAGE}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; height: 6px; width: 100%;">
                            <div style="background-color: {status_color}; width: {min(100.0, max(0.0, impact_pct))}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 5. Recommended Preventive Actions
            st.markdown("""
            <div class="noc-panel" style="padding-bottom: 16px;">
                <div class="noc-panel-title">Recommended Troubleshooting Actions</div>
                <div class="noc-panel-subtitle">Rule-based operational protocols for this predicted fault category.</div>
                <div style="margin-top: 10px;"></div>
            """, unsafe_allow_html=True)
            for i, act in enumerate(actions, 1):
                st.markdown(f"""
                <div style="background-color: #25282e; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; align-items: flex-start; gap: 10px;">
                        <span style="background-color: {status_color}22; color: {status_color}; border: 1px solid {status_color}44; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-family: 'JetBrains Mono'; font-weight: bold; flex-shrink: 0;">
                            {i}
                        </span>
                        <span style="font-size: 0.82rem; font-family: 'JetBrains Mono'; color: {COLOR_TEXT_PRI}; line-height: 1.4;">{act}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")


def render_overview_page():
    # Calculate Overview Metrics
    global_health_pct = round(df_results['health_score'].mean(), 1)
    total_critical = len(df_results[df_results['status'] == 'Critical'])
    total_warning = len(df_results[df_results['status'] == 'Warning'])
    total_healthy = len(df_results[df_results['status'] == 'Healthy'])
    
    # 2. Header Section
    st.markdown('<div class="noc-header-title">Telecom Network Intelligence</div>', unsafe_allow_html=True)
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f'<div class="noc-header-meta">NOC Command Center Overview &bull; Last updated: {current_time_str}</div>', unsafe_allow_html=True)
    
    tab_overview, tab_simulator, tab_ai = st.tabs(["📡 Fleet Overview", "⚡ Live Telemetry Simulator", "🤖 AI Network Assistant"])
    
    with tab_overview:
        # Compact Status Legend
        st.markdown("""
        <div class="legend-container">
            <div class="legend-item"><span class="dot critical"></span> Critical (&lt; 40)</div>
            <div class="legend-item"><span class="dot warning"></span> Warning (40-69)</div>
            <div class="legend-item"><span class="dot healthy"></span> Healthy (&ge; 70)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Metric Cards Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Global Health %",
                value=f"{global_health_pct}%",
                help="Average network infrastructure health score calculated across all monitored active devices."
            )
        
        with col2:
            st.metric(
                label="Total Critical",
                value=str(total_critical),
                help="Devices with severe health degradation (Health Score < 40), requiring immediate maintenance."
            )
        
        with col3:
            st.metric(
                label="Total Warning",
                value=str(total_warning),
                help="Devices exhibiting moderate fault risk indicators (Health Score between 40 and 69)."
            )
        
        with col4:
            st.metric(
                label="Total Healthy",
                value=str(total_healthy),
                help="Devices operating normally under baseline parameters (Health Score >= 70)."
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4. Risk by Location Panel (Plotly Chart)
        st.markdown("""
        <div class="noc-panel">
            <div class="noc-panel-title">Risk by Location (Location-Based Risk Ranking)</div>
            <div class="noc-panel-subtitle">
                Top 15 locations ranked by worst average health score. Note: Locations are anonymized ID rankings, not geographical coordinates.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Compute average health score per location and pick worst 15
        loc_risk = (
            df_results.groupby('location')['health_score']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_health_score', 'count': 'device_count'})
            .sort_values(by='avg_health_score', ascending=True)
            .head(15)
        )
        
        # Reverse for horizontal bar chart (worst at top)
        loc_risk_chart = loc_risk.sort_values(by='avg_health_score', ascending=False)
        
        # Define color map function using centralized constants
        def get_bar_color(val):
            if val < 40:
                return COLOR_CRITICAL
            elif val < 70:
                return COLOR_WARNING
            else:
                return COLOR_HEALTHY
        
        bar_colors = [get_bar_color(v) for v in loc_risk_chart['avg_health_score']]
        
        fig = go.Figure(go.Bar(
            x=loc_risk_chart['avg_health_score'],
            y=loc_risk_chart['location'],
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(color=COLOR_BORDER, width=1)
            ),
            text=[f"{v:.1f}%" for v in loc_risk_chart['avg_health_score']],
            textposition='outside',
            textfont=dict(family='JetBrains Mono', color=COLOR_TEXT_PRI, size=12)
        ))
        
        fig.update_layout(
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_SURFACE,
            margin=dict(l=20, r=40, t=10, b=30),
            height=420,
            font=dict(family='JetBrains Mono', color=COLOR_TEXT_PRI),
            xaxis=dict(
                title='Average Health Score (%)',
                range=[0, 105],
                color=COLOR_TEXT_MUT,
                gridcolor='#2d3436',
                zerolinecolor=COLOR_BORDER,
                tickfont=dict(family='JetBrains Mono', color=COLOR_TEXT_MUT)
            ),
            yaxis=dict(
                title='',
                color=COLOR_TEXT_PRI,
                autorange="reversed",
                tickfont=dict(family='JetBrains Mono', color=COLOR_TEXT_PRI, size=12)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 4b. Network Operations Map Section ---
        @st.cache_data
        def load_topology_data():
            import re
            data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
            test_path = os.path.join(data_dir, "test.csv")
            summaries_path = os.path.join(os.path.dirname(__file__), "models", "device_summaries.pkl")
            
            test_df = pd.read_csv(test_path)
            dev_sums = joblib.load(summaries_path)
            mapping = dev_sums['mapping']
            
            records = []
            for _, row in test_df.iterrows():
                dev_id = int(row['id'])
                loc = row['location']
                if dev_id in mapping:
                    summary = mapping[dev_id]['summary']
                    
                    # Parse health score
                    m_score = re.search(r'Health Score:\s*([\d\.]+)\s*\(([^\)]+)\)', summary)
                    score = float(m_score.group(1)) if m_score else 100.0
                    
                    # Parse other metrics
                    m_sev = re.search(r'Fault Severity:\s*(\d+)', summary)
                    pred_class = int(m_sev.group(1)) if m_sev else 0
                    
                    m_rep = re.search(r'Reported Severity:\s*([^\n]+)', summary)
                    severity_type = m_rep.group(1) if m_rep else ""
                    
                    m_res = re.search(r'Resource Type:\s*([^\n]+)', summary)
                    resource_type = m_res.group(1) if m_res else ""
                    
                    m_evt = re.search(r'Event Types:\s*([^\n]+)', summary)
                    event_types = m_evt.group(1) if m_evt else ""

                    m_log = re.search(r'Log Features:\s*([^\n]+)', summary)
                    log_features = m_log.group(1) if m_log else ""
                    
                    if score >= 70:
                        status_4 = "Healthy"
                    elif score >= 40:
                        status_4 = "Warning"
                    elif score >= 15:
                        status_4 = "High Risk"
                    else:
                        status_4 = "Critical"
                        
                    records.append({
                        'id': dev_id,
                        'location': loc,
                        'health_score': score,
                        'status': status_4,
                        'predicted_class': pred_class,
                        'severity_type': severity_type,
                        'resource_type': resource_type,
                        'event_types': event_types,
                        'log_features': log_features,
                        'summary_text': summary
                    })
            return pd.DataFrame(records)

        df_topo = load_topology_data()

        # Create location clusters for canvas visualization
        clusters = []
        grouped = df_topo.groupby('location')
        for loc, group in grouped:
            group_sorted = group.sort_values(by='health_score', ascending=True)
            devices = []
            for _, row in group_sorted.iterrows():
                devices.append({
                    'id': int(row['id']),
                    'score': float(row['health_score']),
                    'status': str(row['status']),
                    'fault_severity': int(row['predicted_class'])
                })
            clusters.append({
                'name': str(loc).upper().replace("LOCATION ", "LOC-"),
                'raw_name': str(loc),
                'devices': devices
            })
            
        # Sort clusters by name numerically
        def cluster_sort_key(c):
            m = re.search(r'\d+', c['name'])
            return int(m.group(0)) if m else c['name']

        clusters.sort(key=cluster_sort_key)
        import json
        clusters_json = json.dumps(clusters)

        st.markdown("""
        <div class="noc-panel">
            <div class="noc-panel-title">📡 Network Operations Map</div>
            <div class="noc-panel-subtitle">
                Interactive logical topology mapping network devices to their monitored locations. Pan, scroll to zoom, and hover or search for details.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary Row inside the Map Area
        t_devices = len(df_topo)
        t_healthy = len(df_topo[df_topo['status'] == 'Healthy'])
        t_warning = len(df_topo[df_topo['status'] == 'Warning'])
        t_high_risk = len(df_topo[df_topo['status'] == 'High Risk'])
        t_critical = len(df_topo[df_topo['status'] == 'Critical'])

        st.markdown(f"""
        <style>
        .map-kpi-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            padding: 10px 20px;
            border-radius: 8px;
        }}
        .map-kpi-item {{
            flex: 1;
            min-width: 140px;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            border-right: 1px solid {COLOR_BORDER};
        }}
        .map-kpi-item:last-child {{
            border-right: none;
        }}
        .map-kpi-val {{
            font-size: 1.25rem;
            font-weight: bold;
            color: {COLOR_TEXT_PRI};
        }}
        .map-kpi-lbl {{
            font-size: 0.75rem;
            color: {COLOR_TEXT_MUT};
            text-transform: uppercase;
        }}
        .kpi-h {{ color: #10B981; }}
        .kpi-w {{ color: #FBBF24; }}
        .kpi-hr {{ color: #F97316; }}
        .kpi-c {{ color: #EF4444; }}
        </style>
        <div class="map-kpi-container">
            <div class="map-kpi-item">
                <div class="map-kpi-val">{t_devices:,}</div>
                <div class="map-kpi-lbl">Total Devices</div>
            </div>
            <div class="map-kpi-item">
                <div class="map-kpi-val kpi-h">{t_healthy:,}</div>
                <div class="map-kpi-lbl">Healthy</div>
            </div>
            <div class="map-kpi-item">
                <div class="map-kpi-val kpi-w">{t_warning:,}</div>
                <div class="map-kpi-lbl">Warning</div>
            </div>
            <div class="map-kpi-item">
                <div class="map-kpi-val kpi-hr">{t_high_risk:,}</div>
                <div class="map-kpi-lbl">High Risk</div>
            </div>
            <div class="map-kpi-item">
                <div class="map-kpi-val kpi-c">{t_critical:,}</div>
                <div class="map-kpi-lbl">Critical</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Search & Filter Row
        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            search_query = st.text_input(
                "Search Map (ID or Location)",
                placeholder="e.g. 11066, or location 481",
                key="map_search_query"
            )
        with col_s2:
            status_filter = st.selectbox(
                "Filter Map by Status",
                ["All", "Healthy", "Warning", "High Risk", "Critical"],
                key="map_status_filter"
            )
        with col_s3:
            sorted_locations = sorted(list(df_topo['location'].unique()), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else x)
            location_filter = st.selectbox(
                "Filter Map by Location",
                ["All Locations"] + sorted_locations,
                key="map_location_filter"
            )
            
        # Parse search query
        search_dev_id = None
        if search_query.strip():
            if search_query.strip().isdigit():
                search_val = int(search_query.strip())
                if search_val in df_topo['id'].values:
                    search_dev_id = search_val
                    st.session_state.selected_map_device_id = search_val
                else:
                    st.error(f"Device ID #{search_val} not found in database.")
            elif "location" in search_query.lower() or re.match(r'^\d+$', search_query.strip()):
                m_loc_digits = re.search(r'\d+', search_query)
                if m_loc_digits:
                    loc_target = f"location {m_loc_digits.group(0)}"
                    if loc_target in df_topo['location'].values:
                        location_filter = loc_target
                    else:
                        st.error(f"Location {loc_target} not found in database.")

        # Colors mapping dictionary
        colors_status = {
            'Healthy': '#10B981',
            'Warning': '#FBBF24',
            'High Risk': '#F97316',
            'Critical': '#EF4444'
        }

        # Responsive Columns Layout
        col_map_canvas, col_map_details = st.columns([3, 1])
        
        with col_map_canvas:
            html_code = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {
        margin: 0;
        padding: 0;
        background-color: #111317;
        color: #e2e2e8;
        font-family: 'JetBrains Mono', monospace;
        overflow: hidden;
        user-select: none;
    }
    #canvas-container {
        position: relative;
        width: 100%;
        height: 600px;
        background: radial-gradient(circle, #1a1c22 0%, #111317 100%);
        border: 1px solid #3b494b;
        border-radius: 8px;
    }
    canvas {
        display: block;
        width: 100%;
        height: 100%;
        cursor: grab;
    }
    canvas:active {
        cursor: grabbing;
    }
    .tooltip {
        position: absolute;
        display: none;
        background: rgba(30, 32, 36, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid #3b494b;
        border-radius: 6px;
        padding: 10px;
        font-size: 11px;
        color: #e2e2e8;
        pointer-events: none;
        z-index: 1000;
        min-width: 220px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .tooltip-title {
        font-weight: bold;
        color: #00dbe9;
        border-bottom: 1px solid #3b494b;
        padding-bottom: 4px;
        margin-bottom: 6px;
    }
    .tooltip-row {
        margin: 3px 0;
        display: flex;
        justify-content: space-between;
    }
    .tooltip-label {
        color: #b9cacb;
    }
    .tooltip-val {
        font-weight: bold;
    }
    .controls {
        position: absolute;
        bottom: 15px;
        right: 15px;
        display: flex;
        gap: 8px;
        z-index: 10;
    }
    .control-btn {
        width: 32px;
        height: 32px;
        border-radius: 6px;
        background: rgba(30, 32, 36, 0.85);
        border: 1px solid #3b494b;
        color: #e2e2e8;
        font-size: 16px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .control-btn:hover {
        background: #00dbe9;
        color: #111317;
        border-color: #00dbe9;
        box-shadow: 0 0 10px rgba(0, 219, 233, 0.4);
    }
</style>
</head>
<body>
<div id="canvas-container">
    <canvas id="topoCanvas"></canvas>
    <div class="tooltip" id="tooltip"></div>
    <div class="controls">
        <div class="control-btn" onclick="zoomIn()" title="Zoom In">+</div>
        <div class="control-btn" onclick="zoomOut()" title="Zoom Out">-</div>
        <div class="control-btn" onclick="resetView()" title="Fit View">⊙</div>
    </div>
</div>

<script>
    const canvas = document.getElementById('topoCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');
    
    const clusters = %CLUSTERS_JSON%;
    const selectedDeviceId = %SELECTED_DEVICE_ID%;
    const searchDevId = %SEARCH_DEVICE_ID%;
    const statusFilter = "%STATUS_FILTER%";
    const locationFilter = "%LOCATION_FILTER%";
    
    const colors = {
        'Healthy': '#10B981',
        'Warning': '#FBBF24',
        'High Risk': '#F97316',
        'Critical': '#EF4444',
        'Muted': '#4b5563'
    };
    
    const cols = 20;
    const spacingX = 350;
    const spacingY = 300;
    
    let panX = 0;
    let panY = 0;
    let zoomScale = 0.15;
    
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let dragStartX = 0;
    let dragStartY = 0;
    
    let hoveredDevice = null;
    
    function resizeCanvas() {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
    }
    
    function roundRect(x, y, w, h, r) {
        if (w < 2 * r) r = w / 2;
        if (h < 2 * r) r = h / 2;
        ctx.beginPath();
        ctx.moveTo(x+r, y);
        ctx.arcTo(x+w, y,   x+w, y+h, r);
        ctx.arcTo(x+w, y+h, x,   y+h, r);
        ctx.arcTo(x,   y+h, x,   y,   r);
        ctx.arcTo(x,   y,   x+w, y,   r);
        ctx.closePath();
    }
    
    function initLayout() {
        clusters.forEach((c, index) => {
            const gridX = index % cols;
            const gridY = Math.floor(index / cols);
            c.x = gridX * spacingX + 200;
            c.y = gridY * spacingY + 200;
            
            const dev_cols = Math.ceil(Math.sqrt(c.devices.length));
            const dev_rows = Math.ceil(c.devices.length / dev_cols);
            c.width = Math.max(120, dev_cols * 24 + 20);
            c.height = Math.max(80, dev_rows * 24 + 45);
            c.dev_cols = dev_cols;
            
            c.devices.forEach((d, dIdx) => {
                const dc = dIdx % dev_cols;
                const dr = Math.floor(dIdx / dev_cols);
                d.rx = -c.width/2 + 15 + dc * 24;
                d.ry = -c.height/2 + 45 + dr * 24;
                
                d.x = c.x + d.rx;
                d.y = c.y + d.ry;
            });
        });
        
        let focusTarget = null;
        let zoomTo = 0.15;
        if (searchDevId) {
            focusTarget = findDeviceCluster(searchDevId);
            zoomTo = 1.0;
        } else if (selectedDeviceId) {
            focusTarget = findDeviceCluster(selectedDeviceId);
            zoomTo = 1.0;
        } else if (locationFilter && locationFilter !== 'All Locations') {
            focusTarget = clusters.find(c => c.raw_name.toLowerCase() === locationFilter.toLowerCase());
            zoomTo = 1.0;
        }
        
        if (focusTarget) {
            zoomScale = zoomTo;
            centerCameraOn(focusTarget);
        } else {
            fitAll();
        }
    }
    
    function findDeviceCluster(devId) {
        for (let c of clusters) {
            if (c.devices.some(d => d.id === devId)) {
                return c;
            }
        }
        return null;
    }
    
    function centerCameraOn(target) {
        panX = canvas.width / 2 - target.x * zoomScale;
        panY = canvas.height / 2 - target.y * zoomScale;
    }
    
    function fitAll() {
        if (clusters.length === 0) return;
        panX = 50;
        panY = 50;
        zoomScale = Math.min(canvas.width / 7500, canvas.height / 16000);
        if (zoomScale < 0.05) zoomScale = 0.05;
        if (zoomScale > 0.4) zoomScale = 0.4;
    }
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(panX, panY);
        ctx.scale(zoomScale, zoomScale);
        
        const wLeft = (-panX) / zoomScale;
        const wRight = (canvas.width - panX) / zoomScale;
        const wTop = (-panY) / zoomScale;
        const wBottom = (canvas.height - panY) / zoomScale;
        
        ctx.strokeStyle = '#252932';
        ctx.lineWidth = 2;
        clusters.forEach((c, index) => {
            const rightNeighbor = clusters[index + 1];
            if (rightNeighbor && (index % cols < cols - 1)) {
                ctx.beginPath();
                ctx.moveTo(c.x, c.y);
                ctx.lineTo(rightNeighbor.x, rightNeighbor.y);
                ctx.stroke();
            }
            const downNeighbor = clusters[index + cols];
            if (downNeighbor) {
                ctx.beginPath();
                ctx.moveTo(c.x, c.y);
                ctx.lineTo(downNeighbor.x, downNeighbor.y);
                ctx.stroke();
            }
        });
        
        clusters.forEach(c => {
            const halfW = c.width / 2;
            const halfH = c.height / 2;
            if (c.x + halfW < wLeft || c.x - halfW > wRight || c.y + halfH < wTop || c.y - halfH > wBottom) {
                return;
            }
            
            const isLocMatch = (!locationFilter || locationFilter === 'All Locations' || c.raw_name.toLowerCase() === locationFilter.toLowerCase());
            
            let hasCritical = c.devices.some(d => d.status === 'Critical' && (statusFilter === 'All' || statusFilter === 'Critical'));
            let hasWarning = c.devices.some(d => d.status === 'Warning' && (statusFilter === 'All' || statusFilter === 'Warning'));
            let hasHighRisk = c.devices.some(d => d.status === 'High Risk' && (statusFilter === 'All' || statusFilter === 'High Risk'));
            
            let cardBorderColor = '#3b494b';
            let cardBgColor = 'rgba(30, 32, 36, 0.85)';
            
            if (!isLocMatch) {
                cardBgColor = 'rgba(20, 22, 26, 0.3)';
                cardBorderColor = '#252932';
            } else if (hasCritical) {
                cardBorderColor = 'rgba(239, 68, 68, 0.6)';
            } else if (hasHighRisk) {
                cardBorderColor = 'rgba(249, 115, 22, 0.6)';
            } else if (hasWarning) {
                cardBorderColor = 'rgba(251, 191, 36, 0.6)';
            }
            
            ctx.fillStyle = cardBgColor;
            roundRect(c.x - halfW, c.y - halfH, c.width, c.height, 8);
            ctx.fill();
            
            ctx.strokeStyle = cardBorderColor;
            ctx.lineWidth = isLocMatch ? 1.5 : 1;
            ctx.stroke();
            
            if (zoomScale > 0.12) {
                ctx.fillStyle = isLocMatch ? '#b9cacb' : '#4b5563';
                ctx.font = 'bold 11px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(c.name, c.x, c.y - halfH + 18);
                
                ctx.beginPath();
                ctx.strokeStyle = isLocMatch ? '#3b494b' : '#252932';
                ctx.lineWidth = 1;
                ctx.moveTo(c.x - halfW + 10, c.y - halfH + 25);
                ctx.lineTo(c.x + halfW - 10, c.y - halfH + 25);
                ctx.stroke();
            }
            
            c.devices.forEach(d => {
                const isStatusMatch = (statusFilter === 'All' || d.status === statusFilter);
                const isSelected = (d.id === selectedDeviceId);
                const isSearch = (d.id === searchDevId);
                const isActiveMatch = isLocMatch && isStatusMatch;
                
                let dotColor = colors[d.status];
                let dotRadius = 6;
                
                if (!isActiveMatch) {
                    dotColor = colors['Muted'];
                    dotRadius = 3.5;
                }
                
                if (hoveredDevice && hoveredDevice.id === d.id) {
                    dotRadius = 8;
                }
                
                if (isSelected || isSearch) {
                    ctx.beginPath();
                    ctx.arc(d.x, d.y, dotRadius + 5 + Math.sin(Date.now() / 150) * 2, 0, 2 * Math.PI);
                    ctx.strokeStyle = isSearch ? '#EF4444' : '#00dbe9';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
                
                ctx.beginPath();
                ctx.arc(d.x, d.y, dotRadius, 0, 2 * Math.PI);
                ctx.fillStyle = dotColor;
                ctx.fill();
                
                if (zoomScale > 0.45 && isActiveMatch) {
                    ctx.fillStyle = isSelected ? '#00dbe9' : '#e2e2e8';
                    ctx.font = '9px "JetBrains Mono", monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText(d.id, d.x, d.y - 8);
                }
            });
        });
        
        ctx.restore();
    }
    
    canvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        dragStartX = panX;
        dragStartY = panY;
    });
    
    window.addEventListener('mousemove', (e) => {
        const bounds = canvas.getBoundingClientRect();
        const mouseX = e.clientX - bounds.left;
        const mouseY = e.clientY - bounds.top;
        
        if (isDragging) {
            panX = dragStartX + (e.clientX - startX);
            panY = dragStartY + (e.clientY - startY);
            tooltip.style.display = "none";
            requestAnimationFrame(draw);
            return;
        }
        
        const worldX = (mouseX - panX) / zoomScale;
        const worldY = (mouseY - panY) / zoomScale;
        
        let found = null;
        for (let c of clusters) {
            const halfW = c.width / 2;
            const halfH = c.height / 2;
            if (worldX >= c.x - halfW && worldX <= c.x + halfW && worldY >= c.y - halfH && worldY <= c.y + halfH) {
                for (let d of c.devices) {
                    const dx = worldX - d.x;
                    const dy = worldY - d.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist <= 10) {
                        found = {
                            device: d,
                            location: c.name,
                            raw_location: c.raw_name
                        };
                        break;
                    }
                }
            }
            if (found) break;
        }
        
        if (found) {
            if (!hoveredDevice || hoveredDevice.id !== found.device.id) {
                hoveredDevice = found.device;
                tooltip.style.display = "block";
                tooltip.style.left = (e.clientX + 15) + "px";
                tooltip.style.top = (e.clientY + 15) + "px";
                tooltip.innerHTML = `
                    <div class="tooltip-title">DEVICE #${found.device.id}</div>
                    <div class="tooltip-row"><span class="tooltip-label">Location:</span><span class="tooltip-val">${found.raw_location}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Health Score:</span><span class="tooltip-val" style="color:${colors[found.device.status]}">${found.device.score}%</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Severity:</span><span class="tooltip-val">${found.device.status}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Fault Severity:</span><span class="tooltip-val">${found.device.fault_severity}</span></div>
                    <div style="font-size:8px; color:#4b5563; margin-top:5px; text-align:center;">Click node to select device</div>
                `;
                requestAnimationFrame(draw);
            } else {
                tooltip.style.left = (e.clientX + 15) + "px";
                tooltip.style.top = (e.clientY + 15) + "px";
            }
        } else {
            if (hoveredDevice) {
                hoveredDevice = null;
                tooltip.style.display = "none";
                requestAnimationFrame(draw);
            }
        }
    });
    
    window.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    canvas.addEventListener('click', (e) => {
        if (Math.abs(e.clientX - startX) > 6 || Math.abs(e.clientY - startY) > 6) {
            return;
        }
        if (hoveredDevice) {
            window.parent.location.search = "?selected_map_device_id=" + hoveredDevice.id;
        }
    });
    
    const minZoom = 0.03;
    const maxZoom = 4.0;
    
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomIntensity = 0.1;
        const mouseX = e.clientX - canvas.getBoundingClientRect().left;
        const mouseY = e.clientY - canvas.getBoundingClientRect().top;
        
        const wheel = e.deltaY < 0 ? 1 : -1;
        const zoomFactor = Math.exp(wheel * zoomIntensity);
        
        const newZoom = Math.min(Math.max(zoomScale * zoomFactor, minZoom), maxZoom);
        
        panX = mouseX - (mouseX - panX) * (newZoom / zoomScale);
        panY = mouseY - (mouseY - panY) * (newZoom / zoomScale);
        zoomScale = newZoom;
        
        requestAnimationFrame(draw);
    }, { passive: false });
    
    window.zoomIn = function() {
        const mouseX = canvas.width / 2;
        const mouseY = canvas.height / 2;
        const newZoom = Math.min(zoomScale * 1.3, maxZoom);
        panX = mouseX - (mouseX - panX) * (newZoom / zoomScale);
        panY = mouseY - (mouseY - panY) * (newZoom / zoomScale);
        zoomScale = newZoom;
        requestAnimationFrame(draw);
    }
    
    window.zoomOut = function() {
        const mouseX = canvas.width / 2;
        const mouseY = canvas.height / 2;
        const newZoom = Math.max(zoomScale / 1.3, minZoom);
        panX = mouseX - (mouseX - panX) * (newZoom / zoomScale);
        panY = mouseY - (mouseY - panY) * (newZoom / zoomScale);
        zoomScale = newZoom;
        requestAnimationFrame(draw);
    }
    
    window.resetView = function() {
        fitAll();
        requestAnimationFrame(draw);
    }
    
    function animate() {
        requestAnimationFrame(draw);
    }
    
    resizeCanvas();
    window.addEventListener('resize', () => {
        resizeCanvas();
        requestAnimationFrame(draw);
    });
    
    initLayout();
    animate();
</script>
</body>
</html>
"""

            html_code_filled = (
                html_code
                .replace("%CLUSTERS_JSON%", clusters_json)
                .replace("%SELECTED_DEVICE_ID%", str(st.session_state.get("selected_map_device_id", "null")))
                .replace("%SEARCH_DEVICE_ID%", str(search_dev_id) if search_dev_id else "null")
                .replace("%STATUS_FILTER%", status_filter)
                .replace("%LOCATION_FILTER%", location_filter)
            )
            import streamlit.components.v1 as components
            components.html(html_code_filled, height=620, scrolling=False)
            
        with col_map_details:
            selected_map_dev = st.session_state.get("selected_map_device_id")
            if selected_map_dev and selected_map_dev in df_topo['id'].values:
                dev_row = df_topo[df_topo['id'] == selected_map_dev].iloc[0]
                s_color = colors_status.get(dev_row['status'], COLOR_TEXT_PRI)
                
                st.markdown(f"""
                <div class="noc-panel" style="border-top: 3px solid {s_color}; margin-top:0px;">
                    <div style="font-size:0.8rem; text-transform:uppercase; color:{COLOR_TEXT_MUT};">Device Inspection</div>
                    <div style="font-size:1.4rem; font-weight:bold; color:{COLOR_TEXT_PRI}; margin-top:2px;">ID #{selected_map_dev}</div>
                    <div style="margin-top:10px; border-bottom:1px solid {COLOR_BORDER}; padding-bottom:5px;"></div>
                    <div style="margin-top:10px; display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:{COLOR_TEXT_MUT};">Location:</span>
                        <strong style="color:{COLOR_TEXT_PRI};">{dev_row['location']}</strong>
                    </div>
                    <div style="margin-top:5px; display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:{COLOR_TEXT_MUT};">Status:</span>
                        <strong style="color:{s_color};">{dev_row['status']}</strong>
                    </div>
                    <div style="margin-top:5px; display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:{COLOR_TEXT_MUT};">Health Index:</span>
                        <strong style="color:{s_color};">{dev_row['health_score']:.1f}%</strong>
                    </div>
                    <div style="margin-top:5px; display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:{COLOR_TEXT_MUT};">Fault Class:</span>
                        <strong style="color:{COLOR_TEXT_PRI};">{dev_row['predicted_class']} ({'Hardware Fail' if dev_row['predicted_class'] == 2 else 'Link Degradation' if dev_row['predicted_class'] == 1 else 'Normal'})</strong>
                    </div>
                    <div style="margin-top:5px; display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:{COLOR_TEXT_MUT};">Reported Severity:</span>
                        <strong style="color:{COLOR_TEXT_PRI};">{dev_row['severity_type']}</strong>
                    </div>
                    <div style="margin-top:10px; border-bottom:1px solid {COLOR_BORDER}; padding-bottom:5px;"></div>
                    <div style="margin-top:10px; font-size:0.75rem; color:{COLOR_TEXT_MUT}; font-weight:bold;">RESOURCE & EVENT LOGS</div>
                    <div style="margin-top:5px; font-size:0.75rem; color:{COLOR_TEXT_PRI}; font-family:'JetBrains Mono',monospace; word-break:break-all;">
                        {dev_row['resource_type']}<br>
                        {dev_row['event_types']}<br>
                        {dev_row['log_features']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                
                if st.button("VIEW DETAILS / INSPECT", key="btn_inspect_map_device", use_container_width=True):
                    st.session_state.selected_device_id = int(selected_map_dev)
                    st.rerun()
            else:
                st.markdown(f"""
                <div class="noc-panel" style="border: 1px dashed {COLOR_BORDER}; text-align:center; padding: 40px 10px;">
                    <div style="font-size:2.5rem; color:{COLOR_TEXT_MUT}; margin-bottom:10px;">📡</div>
                    <div style="font-size:0.85rem; color:{COLOR_TEXT_MUT};">Select a device node on the topology map to inspect live network telemetry.</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 5. Device Incident List (Sorted by Health Score Ascending)
        st.markdown("""
        <div class="noc-panel">
            <div class="noc-panel-title">Device Infrastructure Monitor</div>
            <div class="noc-panel-subtitle">
                Real-time monitored devices sorted by health score ascending (worst health score first).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter controls
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All Statuses", "Critical", "Warning", "Healthy"]
            )
        with col_f2:
            search_id = st.text_input(
                "Search by Device ID",
                placeholder="e.g. 15086",
                key="txt_search_id"
            )
        
        filtered_df = df_results.sort_values(by='health_score', ascending=True)
        if status_filter != "All Statuses":
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
        if search_id.strip():
            filtered_df = filtered_df[filtered_df['id'].astype(str).str.contains(search_id.strip(), case=False)]
    
        # Initialize checkbox states in session_state if not present
        for idx, row in filtered_df.iterrows():
            dev_key = f"select_{row['id']}"
            if dev_key not in st.session_state:
                st.session_state[dev_key] = False
    
        # Batch Action controls
        col_sel_all, col_clear, col_zip_dl = st.columns([1, 1, 2])
        visible_ids = filtered_df['id'].tolist()
        
        with col_sel_all:
            st.button("Select All", key="btn_select_all", on_click=select_all_visible, args=(visible_ids,), use_container_width=True)
            
        with col_clear:
            st.button("Clear Selection", key="btn_clear_all", on_click=clear_all_visible, args=(visible_ids,), use_container_width=True)
            
        # Find selected IDs across currently visible devices
        selected_ids = [dev_id for dev_id in visible_ids if st.session_state.get(f"select_{dev_id}", False)]
        num_selected = len(selected_ids)
        
        with col_zip_dl:
            if num_selected == 0:
                st.button("Download Selected Reports", key="btn_zip_dl_disabled", disabled=True, use_container_width=True)
            else:
                try:
                    zip_data = generate_reports_zip(selected_ids, df_results, raw_records)
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    st.download_button(
                        label=f"Download {num_selected} Reports",
                        data=zip_data,
                        file_name=f"network_reports_{current_date}.zip",
                        mime="application/zip",
                        key="btn_zip_dl_active",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error packing zip: {e}")
                    
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        
        # Render device cards
        if filtered_df.empty:
            st.markdown(f"""
            <div style="background-color: rgba(255, 59, 59, 0.05); border: 1px solid rgba(255, 59, 59, 0.2); border-radius: 6px; padding: 18px; text-align: center; color: {COLOR_CRITICAL}; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 500; margin-top: 12px;">
                No devices found matching this ID
            </div>
            """, unsafe_allow_html=True)
        else:
            for idx, row in filtered_df.iterrows():
                dev_id = row['id']
                loc = row['location']
                score = row['health_score']
                status = row['status']
                status_class = status.lower()
                
                # Split row into checkbox, device card info, and inspect button
                col_chk, col_card, col_btn = st.columns([0.4, 4.6, 1])
                
                with col_chk:
                    st.checkbox(f"Select #{dev_id}", key=f"select_{dev_id}", label_visibility="collapsed")
                    
                with col_card:
                    st.markdown(f"""
                    <div class="device-card {status_class}">
                        <div class="device-info">
                            <div class="device-id">Device ID: #{dev_id}</div>
                            <div class="device-loc">Location: {loc} &bull; Reported Severity: {row['severity_type']}</div>
                        </div>
                        <div class="device-score-container">
                            <div class="device-score {status_class}">{score:.1f}</div>
                            <div class="status-pill {status_class}">{status}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_btn:
                    st.markdown('<div class="inspect-btn-container">', unsafe_allow_html=True)
                    if st.button("Inspect 📡", key=f"inspect_{dev_id}", use_container_width=True):
                        st.session_state.selected_device_id = dev_id
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
    with tab_simulator:
        render_simulator_tab()
        
    with tab_ai:
        st.markdown(f"""
        <div class="noc-panel">
            <div class="noc-panel-title">NOC AI Operations Assistant</div>
            <div class="noc-panel-subtitle">Ask questions about network status, location alerts, and fault explanations.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Retrieve API Key from env or session state
        env_key = os.getenv("OPENROUTER_API_KEY", "")
        if env_key and env_key != "your_api_key_here" and len(env_key.strip()) > 0:
            st.session_state.openrouter_api_key = env_key
        elif 'openrouter_api_key' not in st.session_state:
            st.session_state.openrouter_api_key = ""
            
        api_key = st.session_state.openrouter_api_key
        is_degraded = not api_key or api_key == "your_api_key_here" or len(api_key.strip()) == 0
        
        if is_degraded:
            st.warning("⚠️ OpenRouter API Key is missing. The AI Assistant is running in local degraded mode (regex-based routing and raw context matching).")
            new_key = st.text_input("Enter OpenRouter API Key to unlock full RAG intelligence:", type="password")
            if new_key:
                st.session_state.openrouter_api_key = new_key
                st.success("API Key updated. Please refresh or enter your question.")
                st.rerun()
                
        # Initialize services
        @st.cache_resource
        def get_rag_service():
            return RAGService()
            
        llm_srv = LLMService(api_key=api_key)
                
        # Render the chat messages history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Welcome to NOC Command Center Assistant. Ask me anything about the network status or location-specific disruptions."}
            ]
            
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Handle user input
        if prompt := st.chat_input("Ask a question (e.g. 'How many total faults are there?' or 'Explain location 515'):"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.spinner("Analyzing request..."):
                response = process_chat_query(prompt, active_device_context=None)
                    
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()


def select_all_visible(visible_ids):
    for dev_id in visible_ids:
        st.session_state[f"select_{dev_id}"] = True

def clear_all_visible(visible_ids):
    for dev_id in visible_ids:
        st.session_state[f"select_{dev_id}"] = False

def generate_reports_zip(selected_ids, df_results, raw_records):
    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for dev_id in selected_ids:
            # Find the row in df_results
            dev_results = df_results[df_results['id'] == dev_id]
            if dev_results.empty:
                continue
            row = dev_results.iloc[0]
            status = row['status']
            score = float(row['health_score'])
            
            # Dynamic status colors
            if status == 'Critical':
                status_color = COLOR_CRITICAL
            elif status == 'Warning':
                status_color = COLOR_WARNING
            else:
                status_color = COLOR_HEALTHY
                
            raw_dict = raw_records.get(dev_id, {})
            if raw_dict:
                raw_dict['id'] = dev_id
                
            if score >= 70:
                shap_factors = []
            else:
                shap_factors = explain(raw_dict)
                
            actions = recommend_action(shap_factors, status)
            
            # Generate individual PDF report
            pdf_bytes = generate_device_pdf(dev_id, row, status_color, shap_factors, actions)
            
            # Add to zip file
            filename = f"device_{dev_id}_report.pdf"
            zip_file.writestr(filename, bytes(pdf_bytes))
            
    return zip_buffer.getvalue()


def generate_device_pdf(device_id, row, status_color, shap_factors, actions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 20, 15)
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(17, 19, 23)
    pdf.cell(0, 10, "Telecom Network Intelligence Report", ln=True, align="C")
    
    # Subtitle with timestamp & ID
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 110, 120)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 8, f"Device ID: #{device_id}  |  Generated: {current_time}", ln=True, align="C")
    pdf.ln(6)
    
    # Divider
    pdf.set_draw_color(185, 202, 203)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)
    
    # Section 1: Device Info
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(17, 19, 23)
    pdf.cell(0, 8, "1. Device Information", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 6, "Device ID:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"#{device_id}", border=0, ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(55, 6, "Location:", border=0)
    pdf.cell(0, 6, str(row['location']), border=0, ln=True)
    
    pdf.cell(55, 6, "Reported Severity Type:", border=0)
    pdf.cell(0, 6, str(row['severity_type']), border=0, ln=True)
    
    pdf.cell(55, 6, "Predicted Fault Severity:", border=0)
    pdf.cell(0, 6, f"Class {row['predicted_class']}", border=0, ln=True)
    pdf.ln(6)
    
    # Section 2: Health Score & Status
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Health Score & Status", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 6, "Composite Health Score:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"{float(row['health_score']):.1f} / 100.0", border=0, ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(55, 6, "Infrastructure Status:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    
    status_str = str(row['status'])
    if status_str == 'Critical':
        pdf.set_text_color(255, 59, 59)
    elif status_str == 'Warning':
        pdf.set_text_color(255, 186, 32)
    else:
        pdf.set_text_color(0, 219, 233)
    pdf.cell(0, 6, status_str, border=0, ln=True)
    pdf.set_text_color(17, 19, 23)
    pdf.ln(6)
    
    # Section 3: Root Cause Analysis (XAI)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. Root Cause Analysis (XAI)", ln=True)
    pdf.ln(2)
    
    if float(row['health_score']) >= 70:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 110, 120)
        pdf.cell(0, 6, "Operating normally - no significant risk factors detected.", ln=True)
        pdf.set_text_color(17, 19, 23)
    else:
        pdf.set_font("Helvetica", "", 10)
        total_shap = sum(val for _, val in shap_factors if val > 0)
        if total_shap <= 0:
            total_shap = 1.0
            
        has_positive = False
        for name, val in shap_factors:
            if val > 0:
                has_positive = True
                impact_pct = (val / total_shap) * 100
                translated = translate_feature_name(name)
                translated_ascii = translated.encode('latin-1', errors='replace').decode('latin-1')
                pdf.cell(100, 6, f"- {translated_ascii}:", border=0)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, f"+{impact_pct:.1f}% impact weight", border=0, ln=True)
                pdf.set_font("Helvetica", "", 10)
                
        if not has_positive:
            pdf.cell(0, 6, "No positive anomaly signals detected.", ln=True)
            
    pdf.ln(6)
    
    # Section 4: Recommended Actions
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "4. Recommended Preventive Actions", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    for i, act in enumerate(actions, 1):
        act_ascii = act.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 6, f"{i}. {act_ascii}")
        pdf.ln(1)
        
    return pdf.output()


def render_detail_page(selected_device_id):
    # Retrieve result record from df_results
    dev_results = df_results[df_results['id'] == selected_device_id]
    if dev_results.empty:
        st.error(f"Device ID #{selected_device_id} not found.")
        if st.button("Back to NOC Overview"):
            st.session_state.selected_device_id = None
            st.rerun()
        return
        
    row = dev_results.iloc[0]
    loc = row['location']
    status = row['status']
    score = float(row['health_score'])
    
    # Dynamic status colors
    if status == 'Critical':
        status_color = COLOR_CRITICAL
    elif status == 'Warning':
        status_color = COLOR_WARNING
    else:
        status_color = COLOR_HEALTHY

    # Precompute SHAP factors and actions for UI and PDF
    raw_dict = raw_records.get(selected_device_id, {})
    if raw_dict:
        raw_dict['id'] = selected_device_id

    if score >= 70:
        shap_factors = []
    else:
        shap_factors = explain(raw_dict)
    actions = recommend_action(shap_factors, status)

    # 1. Action Row (Back button & Download Report button)
    col_back, col_download = st.columns([1, 1])
    with col_back:
        if st.button("Back to NOC Overview", key="back_to_overview"):
            st.session_state.selected_device_id = None
            st.rerun()
            
    with col_download:
        try:
            pdf_bytes = generate_device_pdf(selected_device_id, row, status_color, shap_factors, actions)
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            st.download_button(
                label="Download Report (PDF)",
                data=bytes(pdf_bytes),
                file_name=f"device_{selected_device_id}_report_{current_date}.pdf",
                mime="application/pdf",
                key="btn_download_report",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
        
    # 2. Header Panel
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: 16px; margin-bottom: 24px; margin-top: 12px;">
        <div>
            <h1 style="margin: 0; font-family: 'Geist', sans-serif; font-size: 2rem;">Device ID: #{selected_device_id}</h1>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: {COLOR_TEXT_MUT}; margin-top: 4px;">
                Location: {loc} &bull; Reported Severity: {row['severity_type']}
            </div>
        </div>
        <div>
            <span class="status-pill {status.lower()}">{status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout: Left (Gauge), Right (Diagnostics Placeholders)
    col_gauge, col_diag = st.columns([2, 3])
    
    with col_gauge:
        st.markdown(f"""
        <div class="noc-panel" style="margin-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; padding-bottom: 0px;">
            <div class="noc-panel-title">Device Health Status</div>
            <div class="noc-panel-subtitle">Real-time composite health index.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create gauge figure
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLOR_TEXT_MUT},
                'bar': {'color': status_color},
                'bgcolor': COLOR_SURFACE,
                'borderwidth': 2,
                'bordercolor': COLOR_BORDER,
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 59, 59, 0.05)'},
                    {'range': [40, 70], 'color': 'rgba(255, 186, 32, 0.05)'},
                    {'range': [70, 100], 'color': 'rgba(0, 219, 233, 0.05)'}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_SURFACE,
            font={'color': COLOR_TEXT_PRI, 'family': "JetBrains Mono"},
            height=200,
            margin=dict(l=30, r=30, t=10, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class="noc-panel" style="margin-top: 0px; border-top: none; border-top-left-radius: 0px; border-top-right-radius: 0px; padding-top: 0px; text-align: center;">
            <div style="margin-top: -10px; padding-bottom: 10px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: bold; color: {status_color};">{status.upper()}</span>
                {"<span style='background-color: rgba(255, 59, 59, 0.15); color: " + COLOR_CRITICAL + "; border: 1px solid rgba(255, 59, 59, 0.3); padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-family: \"JetBrains Mono\", monospace; margin-left: 8px;'>Health declining</span>" if status in ['Critical', 'Warning'] else "<span style='background-color: rgba(0, 219, 233, 0.15); color: " + COLOR_HEALTHY + "; border: 1px solid rgba(0, 219, 233, 0.3); padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-family: \"JetBrains Mono\", monospace; margin-left: 8px;'>Stable</span>"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_diag:
        # 1. Root Cause Analysis (XAI) Panel
        if score >= 70:
            # Healthy Device: reassuring message banner
            st.markdown(f"""
            <div class="noc-panel">
                <div class="noc-panel-title">Root Cause Analysis (XAI)</div>
                <div class="noc-panel-subtitle">Top anomaly triggers contributing to predicted status.</div>
                <div style="background-color: rgba(0, 219, 233, 0.05); border: 1px solid rgba(0, 219, 233, 0.2); border-radius: 6px; padding: 18px; text-align: center; color: {COLOR_HEALTHY}; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 500; margin-top: 12px;">
                    Operating normally &bull; no significant risk factors detected.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Warning or Critical Device: compute SHAP factors
            # Compute sum of positive SHAP factors
            total_shap = sum(val for _, val in shap_factors if val > 0)
            if total_shap <= 0:
                total_shap = 1.0
                
            st.markdown(f"""
            <div class="noc-panel" style="padding-bottom: 12px;">
                <div class="noc-panel-title">Root Cause Analysis (XAI)</div>
                <div class="noc-panel-subtitle">Top anomaly triggers ranked by impact weight (AI-powered SHAP).</div>
                <div style="margin-top: 12px;"></div>
            """, unsafe_allow_html=True)
            
            has_positive_factors = False
            for name, val in shap_factors:
                if val <= 0:
                    continue
                has_positive_factors = True
                impact_pct = (val / total_shap) * 100
                translated_name = translate_feature_name(name)
                
                st.markdown(f"""
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; color: {COLOR_TEXT_PRI};">
                        <span>{translated_name}</span>
                        <span style="font-weight: bold; color: {status_color};">+{impact_pct:.1f}%</span>
                    </div>
                    <div style="background-color: {COLOR_BG_PAGE}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; height: 8px; width: 100%;">
                        <div style="background-color: {status_color}; width: {min(100.0, max(0.0, impact_pct))}%; height: 100%; border-radius: 3px; box-shadow: 0 0 6px {status_color}55;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            if not has_positive_factors:
                st.markdown(f"""
                <div style="color: {COLOR_TEXT_MUT}; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; text-align: center; padding: 12px;">
                    No positive anomaly signals detected.
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        # 2. Recommended Preventive Actions Panel
        st.markdown(f"""
        <div class="noc-panel" style="padding-bottom: 16px;">
            <div class="noc-panel-title">Recommended Preventive Actions</div>
            <div class="noc-panel-subtitle">Rule-based network troubleshooting guidelines.</div>
            <div style="margin-top: 12px;"></div>
        """, unsafe_allow_html=True)
        
        for i, act in enumerate(actions, 1):
            st.markdown(f"""
            <div style="background-color: #25282e; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;">
                <div style="display: flex; align-items: flex-start; gap: 12px;">
                    <span style="background-color: {status_color}22; color: {status_color}; border: 1px solid {status_color}44; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; font-weight: bold; flex-shrink: 0;">
                        {i}
                    </span>
                    <span style="font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; color: {COLOR_TEXT_PRI}; line-height: 1.4;">{act}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. What-If Scenario Analysis Expander
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        with st.expander("🔍 Run What-If Impact Scenario"):
            from src.what_if_engine import WhatIfEngine
            engine = WhatIfEngine()
            what_if_res = engine.analyze(selected_device_id, df_results=df_results, raw_records=raw_records)
            if what_if_res:
                st.markdown(f"""
                <div style="background-color: #25282e; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 12px 16px; margin-top: 5px;">
                    <div style="font-family: 'Geist', sans-serif; font-size: 0.9rem; font-weight: bold; text-transform: uppercase; color: {COLOR_TEXT_PRI}; margin-bottom: 8px;">
                        What-If Impact Assessment
                    </div>
                    <div style="font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; line-height: 1.5; color: {COLOR_TEXT_MUT};">
                        <strong>Current Condition:</strong> {what_if_res['current_condition']}<br><br>
                        <strong>Risk Level:</strong> <span style="color: {status_color}; font-weight: bold;">{what_if_res['risk_level']}</span><br><br>
                        <strong>Possible Progression:</strong> {what_if_res['possible_progression']}<br><br>
                        <strong>Potential Impact:</strong> {what_if_res['potential_impact']}<br><br>
                        <strong>Recommended Priority:</strong> {what_if_res['recommended_priority']}<br><br>
                        <strong>Confidence Level:</strong> {what_if_res['confidence_level']}<br><br>
                        <strong>Time to Impact:</strong> {what_if_res['time_to_impact']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("No telemetry data found for analysis.")
            
        if status == 'Critical':
            st.markdown('<div class="dispatch-container">', unsafe_allow_html=True)
            if st.button("Dispatch Tech", key="btn_dispatch_tech", use_container_width=True):
                st.toast(f"Dispatch instruction issued for location {loc} (Device #{selected_device_id})!")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # --- AI Operations Assistant Chatbot Integration (State-Aware) ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="noc-panel">
        <div class="noc-panel-title">NOC AI Device Assistant (State-Aware)</div>
        <div class="noc-panel-subtitle">Ask questions about Device #{selected_device_id} or analyze anomaly metrics in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    # Compile active device context block
    total_shap_val = sum(val for _, val in shap_factors if val > 0)
    if total_shap_val <= 0:
        total_shap_val = 1.0
    shap_strs = []
    for name, val in shap_factors:
        if val > 0:
            impact_pct = (val / total_shap_val) * 100
            translated_name = translate_feature_name(name)
            shap_strs.append(f"- {translated_name}: +{impact_pct:.1f}% impact weight")
    shap_str = "\n".join(shap_strs) if shap_strs else "No positive anomaly signals detected."
    
    actions_str = "\n".join([f"{i}. {act}" for i, act in enumerate(actions, 1)])
    
    active_device_context = (
        f"Device ID: {selected_device_id}\n"
        f"Location: {loc}\n"
        f"Reported Severity: {row['severity_type']}\n"
        f"Predicted Fault Severity Class: {row['predicted_class']}\n"
        f"Composite Health Score: {score:.1f} ({status})\n"
        f"Top Anomaly Triggers (SHAP):\n{shap_str}\n"
        f"Recommended Preventive Actions:\n{actions_str}"
    )

    # Render the chat messages history for this specific device
    chat_key = f"chat_history_{selected_device_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {"role": "assistant", "content": f"I have loaded the telemetry details for Device #{selected_device_id}. Ask me anything about its health score, anomaly triggers, or actions."}
        ]
        
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle user input
    if prompt := st.chat_input("Ask a question about this device (e.g. 'Why is this device critical?' or 'What action should I take?'):", key="detail_chat_input"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing request..."):
            response = process_chat_query(prompt, active_device_context=active_device_context)
            
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state[chat_key].append({"role": "assistant", "content": response})
        st.rerun()


# ROUTING & EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    if 'selected_device_id' not in st.session_state:
        st.session_state.selected_device_id = None
    if 'selected_map_device_id' not in st.session_state:
        st.session_state.selected_map_device_id = None

    # Check query parameters for node selections from iframe visualization
    try:
        q_params = st.query_params
        if "inspect_device_id" in q_params:
            dev_id_str = q_params["inspect_device_id"]
            if dev_id_str:
                st.session_state.selected_device_id = int(dev_id_str)
                st.query_params.clear()
                st.rerun()
        if "selected_map_device_id" in q_params:
            dev_id_str = q_params["selected_map_device_id"]
            if dev_id_str:
                st.session_state.selected_map_device_id = int(dev_id_str)
                st.query_params.clear()
                st.rerun()
    except Exception as e:
        pass

    if st.session_state.selected_device_id is not None:
        render_detail_page(st.session_state.selected_device_id)
    else:
        render_overview_page()

