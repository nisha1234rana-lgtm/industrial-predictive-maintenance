from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

st.set_page_config(
    page_title="Industrial Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Remove Streamlit top white header completely */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    .stAppToolbar {
        display: none !important;
    }

    /* Main page background */
    .stApp {
        background-color: #F4F1EA;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #F4F1EA;
    }

    [data-testid="stMain"] {
        background-color: #F4F1EA;
    }

    /* Main content */
    .block-container {
        padding-top: 1.3rem !important;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #243642;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DDD7CC;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #5B6570;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E8E2D8;
    }

    [data-testid="stSidebarContent"] {
        background-color: #E8E2D8;
    }

    /* Custom information card */
    .status-card {
        background-color: #FFFFFF;
        border: 1px solid #DDD7CC;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .small-note {
        color: #6B7280;
        font-size: 0.88rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_data():

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    maintenance = con.execute(
        """
        SELECT *
        FROM maintenance_decisions
        ORDER BY minute_timestamp
        """
    ).fetchdf()

    episodes = con.execute(
        """
        SELECT *
        FROM maintenance_alert_episodes
        ORDER BY episode_start
        """
    ).fetchdf()

    costs = con.execute(
        """
        SELECT *
        FROM maintenance_cost_simulation
        """
    ).fetchdf()

    drift = con.execute(
        """
        SELECT *
        FROM feature_drift_monitoring
        ORDER BY psi DESC
        """
    ).fetchdf()

    weekly = con.execute(
        """
        SELECT *
        FROM weekly_model_monitoring
        ORDER BY week_start
        """
    ).fetchdf()

    models = con.execute(
        """
        SELECT *
        FROM model_comparison
        ORDER BY horizon, pr_auc DESC
        """
    ).fetchdf()

    failures = con.execute(
        """
        SELECT *
        FROM failure_events
        ORDER BY event_id
        """
    ).fetchdf()

    con.close()

    maintenance["minute_timestamp"] = pd.to_datetime(
        maintenance["minute_timestamp"]
    )

    episodes["episode_start"] = pd.to_datetime(
        episodes["episode_start"]
    )

    episodes["episode_end"] = pd.to_datetime(
        episodes["episode_end"]
    )

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"]
    )

    failures["start_time"] = pd.to_datetime(
        failures["start_time"]
    )

    failures["end_time"] = pd.to_datetime(
        failures["end_time"]
    )

    return (
        maintenance,
        episodes,
        costs,
        drift,
        weekly,
        models,
        failures,
    )


(
    maintenance,
    episodes,
    costs,
    drift,
    weekly,
    models,
    failures,
) = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Maintenance Console")

st.sidebar.caption(
    "MetroPT2 industrial compressor system"
)

view = st.sidebar.radio(
    "Time view",
    [
        "Second failure warning window",
        "Full production period",
        "Latest 48 hours",
    ],
)


SECOND_FAILURE = failures.loc[
    failures["event_id"] == 2,
    "start_time",
].iloc[0]


if view == "Second failure warning window":

    start_time = (
        SECOND_FAILURE
        - pd.Timedelta(hours=24)
    )

    end_time = SECOND_FAILURE

elif view == "Latest 48 hours":

    end_time = maintenance[
        "minute_timestamp"
    ].max()

    start_time = (
        end_time
        - pd.Timedelta(hours=48)
    )

else:

    start_time = maintenance[
        "minute_timestamp"
    ].min()

    end_time = maintenance[
        "minute_timestamp"
    ].max()


filtered = maintenance[
    (
        maintenance["minute_timestamp"]
        >= start_time
    )
    &
    (
        maintenance["minute_timestamp"]
        <= end_time
    )
].copy()


st.sidebar.markdown("---")

st.sidebar.caption(
    f"""
    Showing

    {start_time:%Y-%m-%d %H:%M}

    to

    {end_time:%Y-%m-%d %H:%M}
    """
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "Industrial Equipment Predictive Maintenance"
)

st.caption(
    "IoT telemetry • anomaly detection • failure prediction • "
    "maintenance prioritization • model monitoring"
)


# ============================================================
# EXECUTIVE KPIs
# ============================================================

failure_related = int(
    episodes["failure_related"].sum()
)

false_episodes = int(
    (
        episodes["failure_related"] == 0
    ).sum()
)

reactive_cost = float(
    costs.loc[
        costs["scenario"]
        == "Reactive maintenance",
        "total_cost",
    ].iloc[0]
)

predictive_cost = float(
    costs.loc[
        costs["scenario"]
        == "Predictive maintenance",
        "total_cost",
    ].iloc[0]
)

avoided_cost = (
    reactive_cost
    - predictive_cost
)


warning_rows = maintenance[
    (
        maintenance["minute_timestamp"]
        >= SECOND_FAILURE
        - pd.Timedelta(hours=24)
    )
    &
    (
        maintenance["minute_timestamp"]
        < SECOND_FAILURE
    )
    &
    (
        maintenance["risk_level"]
        .isin(["High", "Critical"])
    )
]


if len(warning_rows) > 0:

    first_warning = (
        warning_rows[
            "minute_timestamp"
        ].min()
    )

    warning_hours = (
        SECOND_FAILURE
        - first_warning
    ).total_seconds() / 3600

else:

    warning_hours = 0


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Earliest warning",
    f"{warning_hours:.2f} h",
)

col2.metric(
    "High / Critical episodes",
    f"{len(episodes):,}",
)

col3.metric(
    "Failure-related episodes",
    f"{failure_related}",
)

col4.metric(
    "False alert episodes",
    f"{false_episodes}",
)

col5.metric(
    "Modeled avoided cost",
    f"${avoided_cost:,.0f}",
)


st.markdown("")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Equipment Health",
        "Failure Intelligence",
        "Model Performance",
        "Monitoring",
        "Business Impact",
    ]
)


# ============================================================
# TAB 1 — EQUIPMENT HEALTH
# ============================================================

with tab1:

    st.subheader(
        "Equipment Health Overview"
    )

    if len(filtered) == 0:

        st.warning(
            "No observations in this time range."
        )

    else:

        latest = filtered.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Risk level",
            latest["risk_level"],
        )

        c2.metric(
            "Maintenance priority",
            f"{latest['maintenance_priority_score']:.1f}/100",
        )

        c3.metric(
            "24h failure score",
            f"{latest['early_warning_score']:.1%}",
        )

        c4.metric(
            "Anomaly percentile",
            f"{latest['anomaly_percentile']:.1f}",
        )


        st.markdown(
            f"""
            <div class="status-card">

            <b>Primary warning signal</b><br>
            {latest['primary_warning_signal']}

            <br><br>

            <b>Recommended action</b><br>
            {latest['recommended_action']}

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # PRIORITY TIMELINE
        # ----------------------------------------------------

        st.markdown(
            "#### Maintenance priority timeline"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=filtered[
                    "minute_timestamp"
                ],
                y=filtered[
                    "maintenance_priority_score"
                ],
                mode="lines",
                name="Priority score",
            )
        )

        fig.add_hline(
            y=45,
            line_dash="dash",
            annotation_text="Elevated risk",
        )

        fig.update_layout(
            yaxis_title="Priority score",
            xaxis_title="",
            height=380,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # SENSOR CHARTS
        # ----------------------------------------------------

        st.markdown(
            "#### Core sensor behavior"
        )

        sensor1, sensor2 = st.columns(2)

        with sensor1:

            fig = px.line(
                filtered,
                x="minute_timestamp",
                y=[
                    "oil_temperature_mean",
                ],
                labels={
                    "value":
                        "Temperature",
                    "minute_timestamp":
                        "",
                },
                title="Oil temperature",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        with sensor2:

            fig = px.line(
                filtered,
                x="minute_timestamp",
                y=[
                    "motor_current_mean",
                    "motor_current_avg_6h",
                ],
                labels={
                    "value":
                        "Motor current",
                    "minute_timestamp":
                        "",
                },
                title="Motor current vs 6-hour baseline",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        sensor3, sensor4 = st.columns(2)

        with sensor3:

            fig = px.line(
                filtered,
                x="minute_timestamp",
                y=[
                    "tp3_mean",
                    "tp3_avg_6h",
                ],
                title="TP3 pressure vs 6-hour baseline",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        with sensor4:

            fig = px.line(
                filtered,
                x="minute_timestamp",
                y="flow_deviation_6h",
                title="Airflow deviation from 6-hour baseline",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# TAB 2 — FAILURE INTELLIGENCE
# ============================================================

with tab2:

    st.subheader(
        "Failure Risk and Warning Signals"
    )


    failure_window = maintenance[
        (
            maintenance[
                "minute_timestamp"
            ]
            >= SECOND_FAILURE
            - pd.Timedelta(hours=24)
        )
        &
        (
            maintenance[
                "minute_timestamp"
            ]
            < SECOND_FAILURE
        )
    ].copy()


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=failure_window[
                "minute_timestamp"
            ],
            y=failure_window[
                "early_warning_score"
            ],
            mode="lines",
            name="24h early-warning model",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=failure_window[
                "minute_timestamp"
            ],
            y=failure_window[
                "near_failure_score"
            ],
            mode="lines",
            name="Near-failure detector",
        )
    )

    fig.add_hline(
        y=float(
            failure_window[
                "early_warning_threshold"
            ].iloc[0]
        ),
        line_dash="dash",
        annotation_text="24h alert threshold",
    )

    fig.update_layout(
        title=(
            "Model risk scores during the "
            "24 hours before failure"
        ),
        yaxis_title="Failure probability",
        xaxis_title="",
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "#### 24-hour early warning"
        )

        st.metric(
            "First warning",
            "22.84 hours before failure",
        )

        st.write(
            "The Logistic Regression model "
            "generated its first High-risk "
            "warning at 2022-07-10 11:20."
        )

        st.write(
            "The first-alert probability was "
            "**91.73%**, above the calibrated "
            "49.90% alert threshold."
        )

        st.markdown(
            """
            Main model drivers included:

            - 6-hour motor-current behavior
            - TP3 pressure baseline
            - reservoir pressure baseline
            - motor-current instability
            - anomaly score
            """
        )


    with c2:

        st.markdown(
            "#### Near-failure detection"
        )

        st.metric(
            "First near-failure warning",
            "0.71 hours before failure",
        )

        st.write(
            "The Random Forest detected every "
            "usable observation in the final "
            "43-minute pre-failure period."
        )

        st.markdown(
            """
            Strong warning signals included:

            - airflow deviation
            - motor-current behavior
            - compressor activity
            - pressure-switch state
            - oil-level state
            """
        )


    st.markdown(
        "#### High and Critical maintenance episodes"
    )

    episode_display = episodes.copy()

    episode_display[
        "failure_related"
    ] = episode_display[
        "failure_related"
    ].map(
        {
            1: "Yes",
            0: "No",
        }
    )

    st.dataframe(
        episode_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab3:

    st.subheader(
        "Failure Prediction Model Comparison"
    )

    st.info(
        "Models were evaluated chronologically. "
        "The July oil-leak failure was kept in "
        "the final test period."
    )


    display_models = models[
        [
            "horizon",
            "model",
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "false_alert_rate_percent",
            "failure_window_alert_percent",
            "first_warning_hours",
        ]
    ].copy()


    numeric_cols = display_models.select_dtypes(
        include="number"
    ).columns

    display_models[
        numeric_cols
    ] = display_models[
        numeric_cols
    ].round(3)


    st.dataframe(
        display_models,
        use_container_width=True,
        hide_index=True,
    )


    col1, col2 = st.columns(2)


    with col1:

        fig = px.bar(
            models,
            x="model",
            y="pr_auc",
            facet_col="horizon",
            title="PR-AUC by model and prediction horizon",
        )

        fig.update_layout(
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with col2:

        fig = px.scatter(
            models,
            x="false_alert_rate_percent",
            y="failure_window_alert_percent",
            size="pr_auc",
            hover_name="model",
            color="horizon",
            title=(
                "Failure detection vs false-alert rate"
            ),
            labels={
                "false_alert_rate_percent":
                    "False alert rate (%)",

                "failure_window_alert_percent":
                    "Failure-window detection (%)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    st.caption(
        "The 6-hour Random Forest result should be "
        "interpreted as near-failure detection. "
        "Telemetry coverage provided only 43 usable "
        "minutes immediately before the July failure."
    )


# ============================================================
# TAB 4 — MONITORING
# ============================================================

with tab4:

    st.subheader(
        "Model and Sensor Monitoring"
    )


    high_drift = int(
        (
            drift["drift_level"]
            == "High"
        ).sum()
    )

    moderate_drift = int(
        (
            drift["drift_level"]
            == "Moderate"
        ).sum()
    )

    stable = int(
        (
            drift["drift_level"]
            == "Stable"
        ).sum()
    )


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "High drift features",
        high_drift,
    )

    c2.metric(
        "Moderate drift",
        moderate_drift,
    )

    c3.metric(
        "Stable features",
        stable,
    )

    c4.metric(
        "Monitoring status",
        "Retraining review",
    )


    st.caption(
        "Drift indicates that July sensor distributions "
        "differ from the healthy baseline. Because July "
        "also contains a documented failure and changing "
        "operating behavior, this is a retraining-review "
        "signal rather than proof of model degradation."
    )


    fig = px.bar(
        drift.sort_values(
            "psi",
            ascending=True,
        ),
        x="psi",
        y="feature",
        color="drift_level",
        orientation="h",
        title="Population Stability Index by feature",
        labels={
            "psi":
                "PSI",

            "feature":
                "",
        },
    )

    fig.update_layout(
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.markdown(
        "#### Weekly production monitoring"
    )


    weekly_fig = go.Figure()

    weekly_fig.add_trace(
        go.Scatter(
            x=weekly[
                "week_start"
            ],
            y=weekly[
                "mean_failure_probability"
            ],
            mode="lines+markers",
            name="Mean failure probability",
        )
    )

    weekly_fig.add_trace(
        go.Scatter(
            x=weekly[
                "week_start"
            ],
            y=weekly[
                "alert_rate_percent"
            ] / 100,
            mode="lines+markers",
            name="Alert rate",
        )
    )

    weekly_fig.update_layout(
        yaxis_title="Rate",
        xaxis_title="",
        height=400,
    )

    st.plotly_chart(
        weekly_fig,
        use_container_width=True,
    )


    drift_display = drift[
        [
            "feature",
            "baseline_mean",
            "production_mean",
            "psi",
            "drift_level",
            "ks_statistic",
        ]
    ].copy()


    st.dataframe(
        drift_display.round(4),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 5 — BUSINESS IMPACT
# ============================================================

with tab5:

    st.subheader(
        "Maintenance Cost Simulation"
    )

    st.caption(
        "Cost values below are scenario assumptions "
        "for business analysis. They are not observed "
        "MetroPT2 maintenance costs."
    )


    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Reactive maintenance",
        f"${reactive_cost:,.0f}",
    )

    c2.metric(
        "Predictive maintenance",
        f"${predictive_cost:,.0f}",
    )

    c3.metric(
        "Modeled savings",
        f"${avoided_cost:,.0f}",
    )


    fig = px.bar(
        costs,
        x="scenario",
        y="total_cost",
        text="total_cost",
        title="Reactive vs predictive maintenance cost",
        labels={
            "total_cost":
                "Modeled cost ($)",

            "scenario":
                "",
        },
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.markdown(
        """
        #### Scenario assumptions

        Reactive maintenance assumes:

        - 12 hours of downtime
        - $2,500 downtime cost per hour
        - $12,000 emergency repair

        Predictive maintenance assumes:

        - 4 hours of controlled downtime
        - $5,000 planned repair
        - $500 inspection cost per High/Critical alert episode

        Under these assumptions, predictive maintenance reduces
        modeled downtime by **8 hours** and total cost by
        **$20,000** for the simulated failure response.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dataset: MetroPT2 industrial air-production telemetry | "
    "7.1M raw sensor readings | April–July 2022"
)