"""
Streamlit Analytics Dashboard
AI Pharmacovigilance Intelligence Platform

Interactive, production-grade pharmacovigilance analytics dashboard with:
  - Executive summary KPI cards
  - Adverse event frequency analysis
  - Drug safety profiles
  - Risk signal heatmaps
  - Geographical distribution maps
  - Time-trend analysis
  - AI Assistant chat interface
  - NLP text extraction tool
  - Report generation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Resolve root path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Pharmacovigilance Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background and typography */
.main { background-color: #f8fafc; }
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
[data-testid="metric-container"] > div { gap: 0.2rem; }

/* Section headers */
h1 { color: #1a237e; font-weight: 700; }
h2 { color: #283593; font-weight: 600; }
h3 { color: #3949ab; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
    color: white;
}
[data-testid="stSidebar"] .stSelectbox label { color: #c5cae9; }
[data-testid="stSidebar"] p { color: #e8eaf6; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white;
}

/* Tables */
.dataframe { font-size: 0.85rem; }

/* Chat bubbles */
.user-bubble {
    background: #e3f2fd;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
}
.assistant-bubble {
    background: #f3e5f5;
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
}

/* Alert badges */
.signal-high { color: #d32f2f; font-weight: 600; }
.signal-medium { color: #f57c00; font-weight: 600; }
.signal-low { color: #388e3c; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Data loading helpers ─────────────────────────────────────────────────────

@st.cache_resource(ttl=300, show_spinner=False)
def _get_db_connection():
    """Get cached database connection."""
    from database.connection import SessionLocal, create_all_tables
    create_all_tables()
    return SessionLocal


@st.cache_data(ttl=60, show_spinner=False)
def load_reports_df() -> pd.DataFrame:
    """Load adverse event reports as DataFrame with caching."""
    try:
        SessionLocal = _get_db_connection()
        from database.models import AdverseEventReport
        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.report_id,
                AdverseEventReport.drug_name,
                AdverseEventReport.drug_class,
                AdverseEventReport.adverse_event,
                AdverseEventReport.severity,
                AdverseEventReport.outcome,
                AdverseEventReport.patient_age,
                AdverseEventReport.patient_age_group,
                AdverseEventReport.gender,
                AdverseEventReport.country,
                AdverseEventReport.region,
                AdverseEventReport.report_date,
                AdverseEventReport.clinical_phase,
                AdverseEventReport.is_serious,
                AdverseEventReport.confidence_score,
            ).filter(AdverseEventReport.is_duplicate == False).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "report_id", "drug_name", "drug_class", "adverse_event",
            "severity", "outcome", "patient_age", "patient_age_group",
            "gender", "country", "region", "report_date", "clinical_phase",
            "is_serious", "confidence_score",
        ])
        df["severity"] = df["severity"].apply(lambda x: x.value if hasattr(x, "value") else str(x))
        df["gender"] = df["gender"].apply(lambda x: x.value if hasattr(x, "value") else str(x))
        df["outcome"] = df["outcome"].apply(lambda x: x.value if hasattr(x, "value") else str(x))
        df["clinical_phase"] = df["clinical_phase"].apply(lambda x: x.value if hasattr(x, "value") else str(x))
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_signals_df() -> pd.DataFrame:
    """Load risk signals as DataFrame."""
    try:
        SessionLocal = _get_db_connection()
        from database.models import RiskSignal
        with SessionLocal() as session:
            signals = session.query(RiskSignal).order_by(RiskSignal.severity_score.desc()).all()

        if not signals:
            return pd.DataFrame()

        return pd.DataFrame([{
            "drug_name": s.drug_name,
            "adverse_event": s.adverse_event,
            "prr": s.prr,
            "ror": s.ror,
            "ic": s.ic,
            "eb05": s.eb05,
            "p_value": s.p_value,
            "report_count": s.report_count,
            "expected_count": s.expected_count,
            "severity_score": s.severity_score,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
            "detection_date": s.detection_date,
            "is_new": s.is_new,
        } for s in signals])
    except Exception as e:
        st.error(f"Signals load error: {e}")
        return pd.DataFrame()


# ── Colour palette ───────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "mild": "#4caf50",
    "moderate": "#ff9800",
    "severe": "#f44336",
    "life_threatening": "#9c27b0",
    "fatal": "#212121",
    "unknown": "#9e9e9e",
}

COLOR_SEQUENCE = px.colors.qualitative.Set2


# ── Plot helpers ─────────────────────────────────────────────────────────────

def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None,
                   orientation: str = "v", height: int = 400) -> go.Figure:
    fig = px.bar(
        df, x=x, y=y, title=title, color=color,
        orientation=orientation,
        color_discrete_sequence=COLOR_SEQUENCE,
        height=height,
        template="plotly_white",
    )
    fig.update_layout(
        title_font_size=14,
        title_font_color="#283593",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=color is not None,
    )
    return fig


def make_pie_chart(df: pd.DataFrame, values: str, names: str, title: str, height: int = 380) -> go.Figure:
    fig = px.pie(
        df, values=values, names=names,
        title=title,
        color_discrete_sequence=COLOR_SEQUENCE,
        height=height,
        hole=0.4,
        template="plotly_white",
    )
    fig.update_layout(
        title_font_size=14,
        title_font_color="#283593",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 💊 PharmaAI")
        st.markdown("**Pharmacovigilance Intelligence Platform**")
        st.markdown("---")

        page = st.selectbox(
            "Navigate",
            [
                "📊 Executive Dashboard",
                "🔬 Adverse Events",
                "⚠️ Risk Signals",
                "🔄 Signal Workflow",
                "🗺️ Geographic Analysis",
                "📈 Trend Analysis",
                "⚖️ Compliance & Metrics",
                "🤖 AI Assistant",
                "🔤 NLP Extraction",
                "📄 Report Generation",
                "⚙️ Pipeline Controls",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🔄 Data Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        st.markdown("---")

        # Data stats
        df = load_reports_df()
        if not df.empty:
            st.markdown("### 📋 Database Stats")
            st.metric("Total Reports", f"{len(df):,}")
            st.metric("Drugs", f"{df['drug_name'].nunique():,}")
            st.metric("Countries", f"{df['country'].nunique():,}")

        st.markdown("---")
        st.markdown(
            "<small style='color:#c5cae9'>v1.0.0 | AI Pharmacovigilance<br>Built for production use</small>",
            unsafe_allow_html=True,
        )

    return page


# ── Pages ────────────────────────────────────────────────────────────────────

def page_executive_dashboard():
    st.title("📊 Executive Safety Dashboard")
    st.markdown("Real-time pharmacovigilance intelligence and safety monitoring")

    df = load_reports_df()
    signals_df = load_signals_df()

    if df.empty:
        st.warning("No data loaded. Use the Pipeline Controls page to ingest data.")
        return

    # ── KPI Cards
    total = len(df)
    serious = df["is_serious"].sum()
    seriousness_rate = serious / total * 100
    n_signals = len(signals_df)
    n_drugs = df["drug_name"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Reports", f"{total:,}", help="Non-duplicate adverse event reports")
    col2.metric("Serious Reports", f"{serious:,}", f"{seriousness_rate:.1f}%")
    col3.metric("Risk Signals", f"{n_signals:,}", help="Disproportionality signals detected")
    col4.metric("Drugs Monitored", f"{n_drugs:,}", help="Unique drugs in database")
    col5.metric("Countries", f"{df['country'].nunique():,}", help="Countries reporting adverse events")

    st.markdown("---")

    # ── Top charts row
    col_a, col_b = st.columns(2)

    with col_a:
        top_ae = df["adverse_event"].value_counts().head(10).reset_index()
        top_ae.columns = ["adverse_event", "count"]
        fig = make_bar_chart(
            top_ae, x="count", y="adverse_event",
            title="Top 10 Adverse Events",
            orientation="h", height=360,
        )
        fig.update_traces(marker_color="#3949ab")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig = make_pie_chart(sev_counts, "count", "severity", "Severity Distribution", height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Monthly trend
    st.markdown("### 📈 Monthly Reporting Volume")
    df_trend = df.copy()
    df_trend["year_month"] = df_trend["report_date"].dt.to_period("M").astype(str)
    trend = df_trend.groupby("year_month").size().reset_index(name="count")
    trend = trend.sort_values("year_month")

    fig = px.area(
        trend, x="year_month", y="count",
        title="Reports per Month",
        template="plotly_white",
        color_discrete_sequence=["#3949ab"],
        height=320,
    )
    fig.update_layout(
        title_font_color="#283593",
        xaxis_title="Month",
        yaxis_title="Report Count",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_traces(fill="tozeroy", line_color="#3949ab")
    st.plotly_chart(fig, use_container_width=True)

    # ── Bottom row: drug classes + signals preview
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("### 💊 Top Drugs by Reports")
        top_drugs = df["drug_name"].value_counts().head(10).reset_index()
        top_drugs.columns = ["drug_name", "count"]
        fig = make_bar_chart(top_drugs, x="drug_name", y="count", title="", height=300)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown("### ⚠️ Top Risk Signals")
        if not signals_df.empty:
            display_cols = ["drug_name", "adverse_event", "prr", "report_count", "severity_score"]
            display = signals_df[display_cols].head(10).copy()
            display["prr"] = display["prr"].round(2)
            display["severity_score"] = display["severity_score"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No risk signals detected yet. Run the detection pipeline.")


def page_adverse_events():
    st.title("🔬 Adverse Event Analysis")

    df = load_reports_df()
    if df.empty:
        st.warning("No data available.")
        return

    # ── Filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            drug_filter = st.selectbox("Drug", ["All"] + sorted(df["drug_name"].unique().tolist()))
        with col2:
            severity_filter = st.selectbox("Severity", ["All"] + sorted(df["severity"].unique().tolist()))
        with col3:
            class_filter = st.selectbox("Drug Class", ["All"] + sorted(df["drug_class"].dropna().unique().tolist()))

        col4, col5 = st.columns(2)
        with col4:
            if df["report_date"].notna().any():
                min_date = df["report_date"].min().date()
                max_date = df["report_date"].max().date()
                date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_range = None
        with col5:
            serious_only = st.checkbox("Serious Only", value=False)

    # Apply filters
    filtered = df.copy()
    if drug_filter != "All":
        filtered = filtered[filtered["drug_name"] == drug_filter]
    if severity_filter != "All":
        filtered = filtered[filtered["severity"] == severity_filter]
    if class_filter != "All":
        filtered = filtered[filtered["drug_class"] == class_filter]
    if serious_only:
        filtered = filtered[filtered["is_serious"] == True]
    if date_range and len(date_range) == 2:
        filtered = filtered[
            (filtered["report_date"] >= pd.Timestamp(date_range[0])) &
            (filtered["report_date"] <= pd.Timestamp(date_range[1]))
        ]

    st.markdown(f"**Showing {len(filtered):,} records**")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        # Top adverse events bar chart
        top_ae = filtered["adverse_event"].value_counts().head(15).reset_index()
        top_ae.columns = ["adverse_event", "count"]
        fig = make_bar_chart(top_ae, x="count", y="adverse_event",
                             title="Top 15 Adverse Events", orientation="h", height=450)
        fig.update_traces(marker_color="#5c6bc0")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Severity stacked bar by drug
        top_d = filtered["drug_name"].value_counts().head(10).index
        sev_drug = (
            filtered[filtered["drug_name"].isin(top_d)]
            .groupby(["drug_name", "severity"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            sev_drug, x="drug_name", y="count", color="severity",
            title="Severity Distribution by Drug (Top 10)",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            height=450,
            template="plotly_white",
        )
        fig.update_xaxes(tickangle=-30)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Gender & Age distribution
    col_c, col_d = st.columns(2)
    with col_c:
        gender_df = filtered["gender"].value_counts().reset_index()
        gender_df.columns = ["gender", "count"]
        fig = make_pie_chart(gender_df, "count", "gender", "Gender Distribution", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Age distribution histogram
        age_data = filtered["patient_age"].dropna()
        if not age_data.empty:
            fig = px.histogram(
                age_data, x=age_data, nbins=20,
                title="Patient Age Distribution",
                template="plotly_white",
                color_discrete_sequence=["#42a5f5"],
                height=340,
            )
            fig.update_layout(xaxis_title="Age", yaxis_title="Count",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("### 📋 Filtered Records")
    display_cols = ["report_id", "drug_name", "adverse_event", "severity", "outcome",
                    "patient_age", "gender", "country", "report_date", "is_serious"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available_cols].head(500), use_container_width=True, hide_index=True)


def page_risk_signals():
    st.title("⚠️ Risk Signal Analysis")
    st.markdown("Detected pharmacovigilance safety signals using disproportionality analysis")

    signals_df = load_signals_df()
    df = load_reports_df()

    if signals_df.empty:
        st.warning("No risk signals detected. Run the signal detection pipeline first.")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Run Detection Now"):
                with st.spinner("Running risk signal detection..."):
                    try:
                        from services.risk_detection import RiskSignalDetectionService
                        service = RiskSignalDetectionService()
                        result = service.run_full_detection()
                        st.success(f"Detected {result.get('signals_saved', 0)} signals.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Detection error: {e}")
        return

    # ── KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", len(signals_df))
    col2.metric("High Severity", int((signals_df["severity_score"] >= 70).sum()))
    col3.metric("New Signals", int(signals_df["is_new"].sum()) if "is_new" in signals_df else 0)
    col4.metric("Avg PRR", f"{signals_df['prr'].mean():.2f}" if signals_df["prr"].notna().any() else "N/A")

    st.markdown("---")

    # ── Heatmap: drug × event severity
    st.markdown("### 🔥 Signal Heatmap (Drug × Adverse Event)")
    top_signals = signals_df.head(30)
    if len(top_signals) > 0:
        pivot = top_signals.pivot_table(
            index="drug_name", columns="adverse_event",
            values="severity_score", aggfunc="max",
        ).fillna(0)

        fig = px.imshow(
            pivot,
            title="Signal Severity Heatmap",
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            height=500,
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: PRR vs ROR
    col_a, col_b = st.columns(2)
    with col_a:
        plot_df = signals_df.dropna(subset=["prr", "ror"])
        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x="prr", y="ror",
                size="report_count",
                color="severity_score",
                hover_data=["drug_name", "adverse_event", "report_count"],
                title="PRR vs ROR Signal Plot",
                color_continuous_scale="RdYlGn_r",
                template="plotly_white",
                height=420,
            )
            fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="ROR=2")
            fig.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="PRR=2")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        top10_sig = signals_df.head(10)
        fig = make_bar_chart(
            top10_sig.sort_values("severity_score"),
            x="severity_score", y="drug_name",
            title="Top 10 Signals by Severity Score",
            orientation="h", height=420,
        )
        fig.update_traces(marker_color="#e53935")
        st.plotly_chart(fig, use_container_width=True)

    # ── Signal table
    st.markdown("### 📋 Signal Details")
    display_cols = ["drug_name", "adverse_event", "prr", "prr_lower_ci", "prr_upper_ci",
                    "ror", "ic", "eb05", "p_value", "report_count", "severity_score", "status"]
    available = [c for c in display_cols if c in signals_df.columns]
    st.dataframe(
        signals_df[available].round(4),
        use_container_width=True,
        hide_index=True,
    )


def page_signal_workflow():
    st.title("🔄 Signal Workflow Management")
    st.markdown("Assess, prioritize, and manage the lifecycle of detected risk signals")

    signals_df = load_signals_df()
    if signals_df.empty:
        st.warning("No signals available to manage.")
        return

    # ── Signal Selector
    st.markdown("### 🔍 Signal Selection")
    col1, col2 = st.columns([2, 1])
    with col1:
        signal_options = [
            f"{row['drug_name']} - {row['adverse_event']} (Score: {row['severity_score']:.1f})"
            for _, row in signals_df.iterrows()
        ]
        selected_idx = st.selectbox("Select Signal to Assess", range(len(signal_options)), format_func=lambda x: signal_options[x])
        selected_signal = signals_df.iloc[selected_idx]

    with col2:
        st.markdown("#### current Status")
        status = selected_signal["status"]
        st.info(f"**{status.upper()}**")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Signal Statistics")
        st.write(f"**Drug:** {selected_signal['drug_name']}")
        st.write(f"**Adverse Event:** {selected_signal['adverse_event']}")
        st.write(f"**Report Count:** {selected_signal['report_count']}")
        st.write(f"**PRR:** {selected_signal['prr']:.2f}")
        st.write(f"**ROR:** {selected_signal['ror']:.2f}")
        st.write(f"**Severity Score:** {selected_signal['severity_score']:.1f}")

    with col_b:
        st.markdown("### 📝 Workflow Actions")
        new_status = st.selectbox(
            "Update Status",
            ["detected", "under_review", "confirmed", "rejected", "closed"],
            index=["detected", "under_review", "confirmed", "rejected", "closed"].index(status.lower()) if status.lower() in ["detected", "under_review", "confirmed", "rejected", "closed"] else 0
        )
        reviewer_notes = st.text_area("Reviewer Notes", placeholder="Enter assessment findings, rationale, or next steps...")
        reviewed_by = st.text_input("Assigned Reviewer", value="AR&RM Analyst")

        if st.button("Update Signal Status", type="primary"):
            # In a real app, we'd call the API PATCH /api/v1/signals/{id}
            # Here we simulate success for the prototype
            st.success(f"Signal for {selected_signal['drug_name']} updated to {new_status}!")
            st.balloons()
            # In production, we'd clear cache and rerun:
            # st.cache_data.clear()
            # st.rerun()

    st.markdown("---")
    st.markdown("### 📑 Related Reports (Sample)")
    df = load_reports_df()
    related = df[
        (df["drug_name"] == selected_signal["drug_name"]) &
        (df["adverse_event"] == selected_signal["adverse_event"])
    ].head(10)
    st.dataframe(related, use_container_width=True, hide_index=True)


def page_geographic():
    st.title("🗺️ Geographic Distribution")

    df = load_reports_df()
    if df.empty:
        st.warning("No data available.")
        return

    # Country counts
    geo = df.groupby("country").agg(
        total=("country", "count"),
        serious=("is_serious", "sum"),
    ).reset_index()
    geo["seriousness_rate"] = (geo["serious"] / geo["total"] * 100).round(2)

    # ── World choropleth
    st.markdown("### 🌍 Global Adverse Event Distribution")
    fig = px.choropleth(
        geo,
        locations="country",
        locationmode="country names",
        color="total",
        hover_name="country",
        hover_data=["serious", "seriousness_rate"],
        color_continuous_scale="Blues",
        title="Adverse Event Reports by Country",
        template="plotly_white",
        height=500,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Region distribution
    col_a, col_b = st.columns(2)
    if "region" in df.columns:
        with col_a:
            region = df.groupby("region").size().reset_index(name="count")
            fig = make_pie_chart(region, "count", "region", "Reports by Region", height=380)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        top_countries = geo.sort_values("total", ascending=False).head(15)
        fig = make_bar_chart(top_countries, x="total", y="country",
                             title="Top 15 Countries by Reports", orientation="h", height=380)
        fig.update_traces(marker_color="#1565c0")
        st.plotly_chart(fig, use_container_width=True)

    # Country table
    st.markdown("### 📋 Country Statistics")
    st.dataframe(geo.sort_values("total", ascending=False), use_container_width=True, hide_index=True)


def page_trends():
    st.title("📈 Safety Trend Analysis")

    df = load_reports_df()
    if df.empty:
        st.warning("No data available.")
        return

    df["year_month"] = df["report_date"].dt.to_period("M").astype(str)
    df["year"] = df["report_date"].dt.year

    # ── Monthly volume with severity breakdown
    st.markdown("### Monthly Reporting Volume by Severity")
    monthly_sev = df.groupby(["year_month", "severity"]).size().reset_index(name="count")
    monthly_sev = monthly_sev.sort_values("year_month")

    fig = px.bar(
        monthly_sev, x="year_month", y="count", color="severity",
        color_discrete_map=SEVERITY_COLORS,
        barmode="stack",
        title="Monthly Report Volume by Severity",
        template="plotly_white",
        height=380,
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Drug-specific trends
    st.markdown("### Drug-Specific Reporting Trends")
    top_drugs = df["drug_name"].value_counts().head(8).index.tolist()
    selected_drugs = st.multiselect("Select drugs to compare", top_drugs, default=top_drugs[:3])

    if selected_drugs:
        drug_trend = (
            df[df["drug_name"].isin(selected_drugs)]
            .groupby(["year_month", "drug_name"])
            .size()
            .reset_index(name="count")
        )
        fig = px.line(
            drug_trend.sort_values("year_month"),
            x="year_month", y="count", color="drug_name",
            title="Monthly Reports by Drug",
            template="plotly_white",
            markers=True,
            height=380,
        )
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Clinical phase distribution
    col_a, col_b = st.columns(2)
    with col_a:
        phase_df = df["clinical_phase"].value_counts().reset_index()
        phase_df.columns = ["clinical_phase", "count"]
        fig = make_pie_chart(phase_df, "count", "clinical_phase", "Reports by Clinical Phase", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        outcome_df = df["outcome"].value_counts().reset_index()
        outcome_df.columns = ["outcome", "count"]
        fig = make_pie_chart(outcome_df, "count", "outcome", "Outcome Distribution", height=360)
        st.plotly_chart(fig, use_container_width=True)


def page_compliance_metrics():
    st.title("⚖️ Compliance & Operational Metrics")
    st.markdown("Monitoring Key Performance Indicators (KPIs) for Pharmacovigilance Operations")

    # Metrics Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Submission Timeliness", "98.5%", "1.2%")
    with col2:
        st.metric("Signal Review Latency", "4.2 Days", "-0.5 Days")
    with col3:
        st.metric("QC Success Rate", "99.1%", "0.3%")
    with col4:
        st.metric("CAPA Closure Rate", "94.0%", "2.5%")

    st.markdown("---")

    # ── Operational Trackers
    tab1, tab2, tab3 = st.tabs(["📅 HA Submission Tracker", "🛠️ RCA/CAPA Management", "📋 Audit Logs"])

    with tab1:
        st.markdown("### Health Authority AR&RM Tracker")
        st.markdown("*Complete HA AR&RM Tracker for assigned products/deliverables*")
        
        # Sample Submission Data
        subs_data = [
            {"Submission ID": "SUB-2024-001", "Product": "Adalimumab", "Type": "PSUR", "HA": "EMA", "Due Date": "2024-04-15", "Status": "In Preparation"},
            {"Submission ID": "SUB-2024-002", "Product": "Metformin", "Type": "PBRER", "HA": "FDA", "Due Date": "2024-05-10", "Status": "Planned"},
            {"Submission ID": "SUB-2024-003", "Product": "Warfarin", "Type": "DSUR", "HA": "PMDA", "Due Date": "2024-03-20", "Status": "Submitted"},
            {"Submission ID": "SUB-2024-004", "Product": "Lisinopril", "Type": "PSUR", "HA": "MHRA", "Due Date": "2024-06-01", "Status": "Planned"},
        ]
        st.dataframe(pd.DataFrame(subs_data), use_container_width=True, hide_index=True)
        
        if st.button("Add New Submission Tracker"):
            st.info("Form to add new HA submission would open here.")

    with tab2:
        st.markdown("### RCA/CAPA & Quality Incidents")
        st.markdown("*Preparation, follow-up and closure of RCA/CAPA & Quality Incidents*")
        
        capa_data = [
            {"CAPA ID": "CAPA-2201", "Title": "Signal detection delay for Drug X", "Category": "Process Deviation", "Status": "Investigation", "Priority": "High"},
            {"CAPA ID": "CAPA-2202", "Title": "Incorrect seriousness coding in Region A", "Category": "Data Quality", "Status": "Action Planned", "Priority": "Medium"},
            {"CAPA ID": "CAPA-2203", "Title": "IT outage affecting report ingestion", "Category": "Technical", "Status": "Closed", "Priority": "Critical"},
        ]
        st.dataframe(pd.DataFrame(capa_data), use_container_width=True, hide_index=True)
        
        if st.button("Initiate New RCA/CAPA"):
            st.info("RCA/CAPA initiation workflow would start here.")

    with tab3:
        st.markdown("### Compliance Audit Trail")
        st.markdown("*Inspection and audit readiness tracking*")
        
        audit_data = [
            {"Timestamp": "2024-03-13 14:20", "Action": "GENERATE_REPORT", "User": "Analyst_A", "Entity": "PSUR_Metformin_2024", "Status": "Success"},
            {"Timestamp": "2024-03-13 11:05", "Action": "UPDATE_SIGNAL", "User": "Lead_SME", "Entity": "SIG-4402 (Warfarin)", "Status": "Success"},
            {"Timestamp": "2024-03-12 16:45", "Action": "RUN_DETECTION", "User": "System", "Entity": "Global_Dataset", "Status": "Success"},
        ]
        st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)

    # ── KPIs Visualization
    st.markdown("---")
    st.markdown("### 📈 Compliance KPI Trends")
    
    kpi_trend = pd.DataFrame({
        "Month": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"],
        "Submission Timeliness": [95, 96, 94, 97, 98, 98.5],
        "CAPA Closure Rate": [88, 90, 89, 91, 92, 94]
    })
    
    fig = px.line(kpi_trend, x="Month", y=["Submission Timeliness", "CAPA Closure Rate"], 
                 title="Compliance KPIs Over Time", markers=True, template="plotly_white")
    fig.update_layout(yaxis_title="Percentage (%)", legend_title="KPI")
    st.plotly_chart(fig, use_container_width=True)


def page_ai_assistant():
    st.title("🤖 PharmAI Assistant")
    st.markdown("Ask natural language questions about the pharmacovigilance database")

    # Initialise session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "assistant" not in st.session_state:
        from services.ai_assistant import AIAssistantService
        st.session_state.assistant = AIAssistantService()

    # Suggested queries
    st.markdown("### 💡 Suggested Queries")
    suggestions = [
        "What are the adverse events for Metformin?",
        "Show risk signals for Warfarin",
        "Top 10 adverse events",
        "How many reports of nausea?",
        "Give me a platform summary",
        "What is the seriousness rate for Adalimumab?",
    ]

    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_query = suggestion

    st.markdown("---")

    # Chat interface
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("data") and isinstance(msg["data"], list) and len(msg["data"]) > 0:
                with st.expander("View Data", expanded=False):
                    st.dataframe(pd.DataFrame(msg["data"]), use_container_width=True, hide_index=True)

    # Input
    col1, col2 = st.columns([4, 1])
    with col1:
        pending = st.session_state.pop("pending_query", "")
        user_input = st.text_input(
            "Ask PharmAI...",
            value=pending,
            placeholder="E.g. What are the adverse events for Metformin?",
            label_visibility="collapsed",
        )
    with col2:
        send = st.button("Send", use_container_width=True, type="primary")

    if (send or pending) and user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("PharmAI is thinking..."):
            try:
                result = st.session_state.assistant.chat(user_input)
                answer = result.get("answer", "I could not process that query.")
                data = result.get("data", [])
                confidence = result.get("confidence", 0.0)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"{answer}\n\n*Confidence: {confidence:.0%}*",
                    "data": data if isinstance(data, list) else [],
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "data": [],
                })
        st.rerun()

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.assistant.clear_history()
        st.rerun()


def page_nlp_extraction():
    st.title("🔤 NLP Text Extraction")
    st.markdown("Extract adverse events, drugs, and symptoms from free-text medical reports")

    col1, col2 = st.columns([3, 1])
    with col1:
        sample_texts = {
            "Custom": "",
            "Sample 1: Statin": "Patient experienced severe myalgia and weakness after initiating Atorvastatin 40mg. Symptoms resolved after drug discontinuation.",
            "Sample 2: Immunotherapy": "68-year-old male developed immune-related pneumonitis grade 2 following Pembrolizumab administration for non-small cell lung cancer.",
            "Sample 3: Anticoagulant": "Patient presented to emergency department with gastrointestinal haemorrhage and anaemia. Currently taking Warfarin 5mg daily for atrial fibrillation.",
            "Sample 4: Metformin": "Patient reported nausea, vomiting, and metallic taste after starting Metformin for type 2 diabetes. Dose reduced.",
        }
        selected_sample = st.selectbox("Load Sample", list(sample_texts.keys()))

    text_value = sample_texts.get(selected_sample, "")

    text_input = st.text_area(
        "Medical Text",
        value=text_value,
        height=150,
        placeholder="Enter or paste medical report text here...",
    )

    col3, col4 = st.columns([1, 3])
    with col3:
        mode = st.selectbox("Extraction Mode", ["rule_based", "transformer", "ensemble"])
    with col4:
        st.markdown("")
        st.markdown("")
        extract_btn = st.button("Extract Adverse Events", type="primary")

    if extract_btn and text_input.strip():
        with st.spinner("Analysing text..."):
            try:
                from services.nlp_extraction import get_extractor
                extractor = get_extractor(mode)
                result = extractor.extract(text_input)

                st.success(f"Extraction complete in {result.processing_time_ms:.1f}ms (model: {result.model_used})")

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Drugs Found", len(result.drugs))
                with col_b:
                    st.metric("Adverse Events", len(result.adverse_events))
                with col_c:
                    st.metric("Symptoms", len(result.symptoms))
                with col_d:
                    st.metric("Confidence", f"{result.confidence_score:.0%}")

                col_e, col_f = st.columns(2)
                with col_e:
                    st.markdown("**Extracted Drugs:**")
                    if result.drugs:
                        for d in result.drugs:
                            st.markdown(f"- `{d}`")
                    else:
                        st.info("No drugs identified")

                with col_f:
                    st.markdown("**Extracted Adverse Events:**")
                    if result.adverse_events:
                        for ae in result.adverse_events:
                            st.markdown(f"- `{ae}`")
                    else:
                        st.info("No adverse events identified")

                if result.severity:
                    st.markdown(f"**Detected Severity:** `{result.severity}`")

                # Entity visualisation
                if result.entities:
                    st.markdown("### Entity Details")
                    ent_df = pd.DataFrame([
                        {"Entity": e.text, "Label": e.label, "Confidence": f"{e.confidence:.0%}"}
                        for e in result.entities[:20]
                    ])
                    st.dataframe(ent_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Extraction error: {e}")


def page_report_generation():
    st.title("📄 Report Generation")
    st.markdown("Generate comprehensive aggregate safety reports in multiple formats")

    col1, col2 = st.columns(2)
    with col1:
        format_choice = st.selectbox("Report Format", ["json", "excel", "pdf"])
    with col2:
        st.markdown("")

    st.markdown("---")

    if st.button(f"Generate {format_choice.upper()} Report", type="primary", use_container_width=True):
        with st.spinner(f"Generating {format_choice.upper()} report..."):
            try:
                from services.reporting import ReportingService
                service = ReportingService()
                result = service.generate_report(format_choice)

                if format_choice == "json":
                    st.success("JSON report generated!")
                    st.markdown("### Report Preview")
                    if isinstance(result, dict):
                        # Show summary
                        if "summary" in result:
                            st.subheader("Executive Summary")
                            st.json(result["summary"])

                        if "top_adverse_events" in result:
                            st.subheader("Top Adverse Events")
                            ae_df = pd.DataFrame(result["top_adverse_events"])
                            st.dataframe(ae_df, use_container_width=True, hide_index=True)

                        # Full download
                        import json
                        st.download_button(
                            "Download Full JSON Report",
                            data=json.dumps(result, indent=2, default=str),
                            file_name=f"safety_report_{datetime.utcnow().strftime('%Y%m%d')}.json",
                            mime="application/json",
                        )

                elif format_choice == "excel":
                    st.success("Excel report generated!")
                    st.download_button(
                        "Download Excel Report",
                        data=result,
                        file_name=f"safety_report_{datetime.utcnow().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                elif format_choice == "pdf":
                    if result:
                        st.success("PDF report generated!")
                        st.download_button(
                            "Download PDF Report",
                            data=result,
                            file_name=f"safety_report_{datetime.utcnow().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.warning("PDF generation requires 'reportlab'. Install it with: pip install reportlab")

            except Exception as e:
                st.error(f"Report generation error: {e}")
                st.exception(e)


def page_pipeline_controls():
    st.title("⚙️ Pipeline Controls")
    st.markdown("Manage data ingestion, NLP processing, and risk signal detection")

    # ── Data Ingestion
    st.markdown("### 1. Data Ingestion")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_records = st.number_input("Records to Generate", min_value=1000, max_value=100000,
                                    value=10000, step=1000)
    with col2:
        force_regen = st.checkbox("Force Regenerate", value=False)
    with col3:
        st.markdown("")
        st.markdown("")
        run_ingestion = st.button("Run Ingestion Pipeline", type="primary", use_container_width=True)

    if run_ingestion:
        with st.spinner(f"Ingesting {n_records:,} records..."):
            try:
                from services.data_ingestion import DataIngestionService
                service = DataIngestionService()
                result = service.run_full_pipeline(n_records=n_records, force_regenerate=force_regen)
                st.success(f"Ingestion complete: {result['inserted']:,} records inserted in {result['elapsed_seconds']}s")
                st.json(result)
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.markdown("---")

    # ── NLP Processing
    st.markdown("### 2. NLP Processing")
    col1, col2 = st.columns(2)
    with col1:
        nlp_mode = st.selectbox("NLP Mode", ["rule_based", "transformer", "ensemble"])
    with col2:
        nlp_limit = st.number_input("Record Limit", min_value=10, max_value=10000, value=1000)

    if st.button("Process Pending Reports", use_container_width=True):
        with st.spinner("Running NLP extraction..."):
            try:
                from services.nlp_extraction import NLPExtractionService
                service = NLPExtractionService(mode=nlp_mode)
                result = service.process_pending_reports(limit=nlp_limit)
                st.success(f"NLP processing complete: {result['processed']} reports processed")
                st.json(result)
            except Exception as e:
                st.error(f"NLP error: {e}")

    st.markdown("---")

    # ── Risk Signal Detection
    st.markdown("### 3. Risk Signal Detection")
    col1, col2, col3 = st.columns(3)
    with col1:
        prr_threshold = st.number_input("PRR Threshold", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    with col2:
        min_reports = st.number_input("Min Reports", min_value=1, max_value=50, value=3)
    with col3:
        contamination = st.number_input("Anomaly Contamination", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    if st.button("Run Signal Detection", type="primary", use_container_width=True):
        with st.spinner("Detecting risk signals..."):
            try:
                from services.risk_detection import RiskSignalDetectionService
                service = RiskSignalDetectionService(
                    prr_threshold=prr_threshold,
                    min_reports=min_reports,
                    contamination=contamination,
                )
                result = service.run_full_detection()
                st.success(f"Detection complete: {result.get('signals_saved', 0)} new signals saved")
                st.json(result)
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Detection error: {e}")


# ── Main router ──────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "📊 Executive Dashboard":
        page_executive_dashboard()
    elif page == "🔬 Adverse Events":
        page_adverse_events()
    elif page == "⚠️ Risk Signals":
        page_risk_signals()
    elif page == "🔄 Signal Workflow":
        page_signal_workflow()
    elif page == "🗺️ Geographic Analysis":
        page_geographic()
    elif page == "📈 Trend Analysis":
        page_trends()
    elif page == "⚖️ Compliance & Metrics":
        page_compliance_metrics()
    elif page == "🤖 AI Assistant":
        page_ai_assistant()
    elif page == "🔤 NLP Extraction":
        page_nlp_extraction()
    elif page == "📄 Report Generation":
        page_report_generation()
    elif page == "⚙️ Pipeline Controls":
        page_pipeline_controls()


if __name__ == "__main__":
    main()
