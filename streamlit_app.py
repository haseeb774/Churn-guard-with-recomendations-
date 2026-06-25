"""
streamlit_app.py
------------------
Churn Intelligence Dashboard — the customer-facing SaaS UI.

Tabs:
    1. Overview          - KPI cards, churn distribution, top risk drivers across the base
    2. At-Risk Customers  - filterable/sortable table with risk flags
    3. Customer Detail     - deep dive on one customer: profile, score, drivers, chart
    4. Simulate Activity   - generate "live" new signups / usage changes on demand

Run locally:
    streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline import ChurnPipeline
from src.simulate import simulate_batch

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◆",
    layout="wide",
)

# ----------------------------------------------------------------------
# Theming — quiet, data-dense, one accent color used deliberately
# ----------------------------------------------------------------------
ACCENT = "#D14E3A"        # terracotta-red alert accent (used ONLY for risk/danger)
SAFE = "#2E7D5B"          # muted green for low risk
NEUTRAL = "#5B6470"       # slate for neutral text/bars
BG_CARD = "#F7F5F1"

st.markdown(f"""
<style>
.metric-card {{
    background-color: {BG_CARD};
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #E5E1D8;
}}
.risk-critical {{ color: {ACCENT}; font-weight: 700; }}
.risk-high {{ color: #C97A2B; font-weight: 700; }}
.risk-medium {{ color: #B59A1F; font-weight: 600; }}
.risk-low {{ color: {SAFE}; font-weight: 600; }}
div.block-container {{ padding-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return ChurnPipeline(artifact_path="artifacts/best_model.pkl")


@st.cache_data
def load_raw():
    return pd.read_csv("data/raw/churn.csv")


@st.cache_data(show_spinner="Scoring customers...")
def score_customers(raw_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = load_pipeline()
    return pipeline.run(raw_df.drop(columns=["Churn"], errors="ignore"))


def risk_badge(tier: str) -> str:
    cls = {
        "Critical": "risk-critical",
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(tier, "risk-low")
    return f'<span class="{cls}">{tier}</span>'


# ----------------------------------------------------------------------
# Session state for simulated/live data
# ----------------------------------------------------------------------
if "live_events" not in st.session_state:
    st.session_state.live_events = pd.DataFrame()

raw = load_raw()
base_scored = score_customers(raw)

if not st.session_state.live_events.empty:
    full_table = pd.concat(
        [base_scored, st.session_state.live_events], ignore_index=True
    )
else:
    full_table = base_scored

st.title("Churn Intelligence")
st.caption("Predictive churn scoring, risk alerts, and retention recommendations")

tab_overview, tab_atrisk, tab_detail, tab_simulate = st.tabs(
    ["Overview", "At-Risk Customers", "Customer Detail", "Simulate Activity"]
)

# ----------------------------------------------------------------------
# TAB 1 — Overview
# ----------------------------------------------------------------------
with tab_overview:
    total = len(full_table)
    at_risk = (full_table["risk_tier"].isin(["High", "Critical"])).sum()
    avg_prob = full_table["churn_probability"].mean()
    critical = (full_table["risk_tier"] == "Critical").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div style="font-size:13px;color:{NEUTRAL}">Total Customers</div>'
            f'<div style="font-size:28px;font-weight:700">{total:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div style="font-size:13px;color:{NEUTRAL}">At Risk (High/Critical)</div>'
            f'<div style="font-size:28px;font-weight:700;color:{ACCENT}">{at_risk:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div style="font-size:13px;color:{NEUTRAL}">Avg Churn Probability</div>'
            f'<div style="font-size:28px;font-weight:700">{avg_prob:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div style="font-size:13px;color:{NEUTRAL}">Critical Alerts</div>'
            f'<div style="font-size:28px;font-weight:700;color:{ACCENT}">{critical:,}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        tier_counts = full_table["risk_tier"].value_counts().reindex(
            ["Low", "Medium", "High", "Critical"]
        ).fillna(0)
        fig = px.bar(
            x=tier_counts.index, y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map={
                "Low": SAFE, "Medium": "#B59A1F", "High": "#C97A2B", "Critical": ACCENT
            },
            labels={"x": "Risk Tier", "y": "Customers"},
            title="Customer Base by Risk Tier",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        all_drivers = full_table["risk_drivers"].explode()
        all_drivers = all_drivers[all_drivers != "No major individual risk drivers detected"]
        top_drivers = all_drivers.value_counts().head(8).sort_values()
        fig2 = px.bar(
            x=top_drivers.values, y=top_drivers.index, orientation="h",
            labels={"x": "Customers affected", "y": ""},
            title="Most Common Risk Drivers Across Customer Base",
        )
        fig2.update_traces(marker_color=NEUTRAL)
        fig2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2 — At-Risk Customers
# ----------------------------------------------------------------------
with tab_atrisk:
    st.subheader("At-Risk Customers")

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        tier_filter = st.multiselect(
            "Risk tier", ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with f2:
        min_prob = st.slider("Min churn probability", 0.0, 1.0, 0.0, 0.05)
    with f3:
        search_id = st.text_input("Search customer ID")

    view = full_table.copy()
    if tier_filter:
        view = view[view["risk_tier"].isin(tier_filter)]
    view = view[view["churn_probability"] >= min_prob]
    if search_id:
        view = view[view["customerID"].astype(str).str.contains(search_id, case=False)]

    view = view.sort_values("churn_probability", ascending=False)

    display_cols = ["customerID", "risk_tier", "churn_probability", "tenure",
                     "MonthlyCharges", "Contract"]
    display_cols = [c for c in display_cols if c in view.columns]

    st.write(f"{len(view)} customer(s) match filters")
    st.dataframe(
        view[display_cols].style.format({"churn_probability": "{:.1%}"}),
        use_container_width=True,
        height=450,
    )

# ----------------------------------------------------------------------
# TAB 3 — Customer Detail
# ----------------------------------------------------------------------
with tab_detail:
    st.subheader("Customer Detail")

    customer_options = full_table["customerID"].astype(str).tolist()
    selected_id = st.selectbox("Select a customer", customer_options)

    row = full_table[full_table["customerID"].astype(str) == selected_id].iloc[0]

    colL, colR = st.columns([1, 1.4])

    with colL:
        prob = row["churn_probability"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT if prob >= 0.5 else NEUTRAL},
                "steps": [
                    {"range": [0, 25], "color": "#EAF3EE"},
                    {"range": [25, 50], "color": "#FBF4DC"},
                    {"range": [50, 75], "color": "#FBE6D6"},
                    {"range": [75, 100], "color": "#F8DCD6"},
                ],
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"**Risk Tier:** {risk_badge(row['risk_tier'])}", unsafe_allow_html=True)

        st.write("**Profile**")
        profile_fields = ["tenure", "Contract", "MonthlyCharges", "TotalCharges",
                           "PaymentMethod", "InternetService"]
        for field in profile_fields:
            if field in row.index:
                st.write(f"- **{field}:** {row[field]}")

    with colR:
        st.write("**Why this customer is at risk**")
        drivers = row["risk_drivers"]
        if isinstance(drivers, list) and drivers:
            for d in drivers:
                st.markdown(f"- {d}")
        else:
            st.write("No major risk drivers detected.")

        st.write("**Recommended retention actions**")
        recs = row["recommendations"]
        if isinstance(recs, list) and recs:
            for r in recs:
                st.success(r, icon="✅")
        else:
            st.write("No action needed — monitor at next billing cycle.")

        # Comparison chart: this customer vs cohort average
        numeric_compare = [c for c in ["tenure", "MonthlyCharges", "NumServices", "RiskScore"]
                            if c in full_table.columns]
        if numeric_compare:
            cohort_avg = full_table[numeric_compare].mean()
            customer_vals = row[numeric_compare]
            compare_df = pd.DataFrame({
                "Metric": numeric_compare,
                "This Customer": customer_vals.values,
                "Cohort Average": cohort_avg.values,
            }).melt(id_vars="Metric", var_name="Group", value_name="Value")

            fig3 = px.bar(
                compare_df, x="Metric", y="Value", color="Group",
                barmode="group",
                color_discrete_map={"This Customer": ACCENT, "Cohort Average": NEUTRAL},
                title="This Customer vs. Cohort Average",
            )
            fig3.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4 — Simulate Activity
# ----------------------------------------------------------------------
with tab_simulate:
    st.subheader("Simulate Live Customer Activity")
    st.caption(
        "Generates new signups and usage changes sampled from real customer "
        "profiles, scores them, and adds them to the dashboard for this session."
    )

    s1, s2 = st.columns(2)
    with s1:
        n_new = st.number_input("New signups to simulate", 0, 20, 3)
    with s2:
        n_updates = st.number_input("Existing customer updates to simulate", 0, 20, 3)

    if st.button("Run Simulation", type="primary"):
        batch = simulate_batch(raw, n_new=n_new, n_updates=n_updates)
        scored_batch = score_customers(batch.drop(columns=["_event_type"], errors="ignore"))
        scored_batch["_event_type"] = batch["_event_type"].values[:len(scored_batch)]

        st.session_state.live_events = pd.concat(
            [st.session_state.live_events, scored_batch], ignore_index=True
        )
        st.success(f"Added {len(scored_batch)} simulated customer event(s). See Overview / At-Risk tabs.")

    if not st.session_state.live_events.empty:
        st.write("**Simulated events this session**")
        cols = [c for c in ["customerID", "_event_type", "risk_tier", "churn_probability"]
                if c in st.session_state.live_events.columns]
        st.dataframe(st.session_state.live_events[cols], use_container_width=True)

        if st.button("Clear simulated events"):
            st.session_state.live_events = pd.DataFrame()
            st.rerun()