"""
Streamlit BI Dashboard for FloodMAS visualization and analysis.
Displays simulation results, metrics comparison, and interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="FloodMAS Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 FloodMAS - Multi-Agent Flood Detection Dashboard")

DATA_DIR = Path("outputs")
LOGS_DIR = DATA_DIR / "logs"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
MODELS_DIR = DATA_DIR / "models"


@st.cache_data
def load_simulation_logs(log_path: str) -> pd.DataFrame:
    """Load simulation logs from parquet file."""
    try:
        return pd.read_parquet(log_path)
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        return pd.DataFrame()


@st.cache_data
def load_experiment_results(results_path: str) -> dict:
    """Load experiment results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return {}


@st.cache_data
def load_training_report(report_path: str) -> dict:
    """Load ML training report."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}


def render_sidebar():
    """Render sidebar with data source selection."""
    st.sidebar.header("📁 Data Selection")
    
    log_files = list(LOGS_DIR.glob("*.parquet")) if LOGS_DIR.exists() else []
    exp_files = list(EXPERIMENTS_DIR.glob("*.json")) if EXPERIMENTS_DIR.exists() else []
    
    selected_log = None
    selected_exp = None
    
    if log_files:
        log_options = ["None"] + [f.name for f in log_files]
        selected_log_name = st.sidebar.selectbox("Simulation Log", log_options)
        if selected_log_name != "None":
            selected_log = LOGS_DIR / selected_log_name
    else:
        st.sidebar.info("No simulation logs found in outputs/logs/")
    
    if exp_files:
        exp_options = ["None"] + [f.name for f in exp_files]
        selected_exp_name = st.sidebar.selectbox("Experiment Results", exp_options)
        if selected_exp_name != "None":
            selected_exp = EXPERIMENTS_DIR / selected_exp_name
    else:
        st.sidebar.info("No experiment results found in outputs/experiments/")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Filters")
    
    return selected_log, selected_exp


def render_timeline(logs: pd.DataFrame):
    """Render timeline visualization of risk and state."""
    st.subheader("📈 Timeline Analysis")
    
    if logs.empty:
        st.warning("No data available for timeline visualization.")
        return
    
    zones = sorted(logs['zone_id'].unique())
    selected_zone = st.selectbox("Select Zone", zones, key="timeline_zone")
    
    zone_data = logs[logs['zone_id'] == selected_zone].sort_values('step')
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Risk Score', 'Alert State', 'Ground Truth vs Prediction'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    fig.add_trace(
        go.Scatter(
            x=zone_data['step'],
            y=zone_data['risk'],
            mode='lines',
            name='Risk',
            line=dict(color='#FF6B6B', width=2)
        ),
        row=1, col=1
    )
    
    if 'water_mean_5' in zone_data.columns:
        fig.add_trace(
            go.Scatter(
                x=zone_data['step'],
                y=zone_data['water_mean_5'],
                mode='lines',
                name='Water Level',
                line=dict(color='#4ECDC4', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    state_map = {'NORMAL': 0, 'SUSPECTED': 1, 'ALERT': 2, 'COOLDOWN': 3}
    state_colors = {'NORMAL': '#2ECC71', 'SUSPECTED': '#F39C12', 'ALERT': '#E74C3C', 'COOLDOWN': '#3498DB'}
    
    zone_data['state_numeric'] = zone_data['state'].map(state_map)
    
    for state, color in state_colors.items():
        mask = zone_data['state'] == state
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=zone_data.loc[mask, 'step'],
                    y=zone_data.loc[mask, 'state_numeric'],
                    mode='markers',
                    name=state,
                    marker=dict(color=color, size=8)
                ),
                row=2, col=1
            )
    
    if 'ground_truth_flooded' in zone_data.columns:
        fig.add_trace(
            go.Scatter(
                x=zone_data['step'],
                y=zone_data['ground_truth_flooded'].astype(int),
                mode='lines',
                name='Ground Truth',
                line=dict(color='#E74C3C', width=2)
            ),
            row=3, col=1
        )
        
        predicted = zone_data['state'].isin(['ALERT', 'SUSPECTED']).astype(int)
        fig.add_trace(
            go.Scatter(
                x=zone_data['step'],
                y=predicted,
                mode='lines',
                name='Predicted',
                line=dict(color='#3498DB', width=2, dash='dash')
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Risk", row=1, col=1)
    fig.update_yaxes(title_text="State", ticktext=['NORMAL', 'SUSPECTED', 'ALERT', 'COOLDOWN'],
                     tickvals=[0, 1, 2, 3], row=2, col=1)
    fig.update_yaxes(title_text="Flood", ticktext=['No', 'Yes'], tickvals=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text="Step", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(logs: pd.DataFrame):
    """Render confusion matrix visualization."""
    st.subheader("📊 Confusion Matrix")
    
    if logs.empty or 'ground_truth_flooded' not in logs.columns:
        st.warning("No ground truth data available.")
        return
    
    y_true = logs['ground_truth_flooded'].astype(int)
    y_pred = logs['state'].isin(['ALERT', 'SUSPECTED']).astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Flood', 'Flood'],
        y=['No Flood', 'Flood'],
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(height=400)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1 Score", f"{f1:.3f}")
        st.metric("False Positive Rate", f"{fp / (fp + tn):.3f}" if (fp + tn) > 0 else "N/A")


def render_lead_time_distribution(logs: pd.DataFrame):
    """Render lead time distribution analysis."""
    st.subheader("⏱️ Lead Time Distribution")
    
    if logs.empty or 'ground_truth_flooded' not in logs.columns:
        st.warning("No data available for lead time analysis.")
        return
    
    lead_times = []
    
    for zone_id in logs['zone_id'].unique():
        zone_data = logs[logs['zone_id'] == zone_id].sort_values('step')
        
        alert_starts = []
        flood_starts = []
        
        prev_state = 'NORMAL'
        prev_flood = False
        
        for _, row in zone_data.iterrows():
            if row['state'] == 'ALERT' and prev_state != 'ALERT':
                alert_starts.append(row['step'])
            if row['ground_truth_flooded'] and not prev_flood:
                flood_starts.append(row['step'])
            prev_state = row['state']
            prev_flood = row['ground_truth_flooded']
        
        for flood_time in flood_starts:
            prior_alerts = [a for a in alert_starts if a < flood_time]
            if prior_alerts:
                lead_times.append(flood_time - max(prior_alerts))
    
    if lead_times:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                x=lead_times,
                nbins=20,
                labels={'x': 'Lead Time (steps)', 'y': 'Count'},
                title='Lead Time Distribution'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Mean Lead Time", f"{np.mean(lead_times):.1f} steps")
            st.metric("Median Lead Time", f"{np.median(lead_times):.1f} steps")
            st.metric("Min Lead Time", f"{np.min(lead_times)} steps")
            st.metric("Max Lead Time", f"{np.max(lead_times)} steps")
            st.metric("Total Detections", len(lead_times))
    else:
        st.info("No lead time data available (no alerts before floods detected).")


def render_experiment_comparison(results: dict):
    """Render experiment comparison visualization."""
    st.subheader("🔬 Experiment Comparison: MAS vs Baseline")
    
    if not results or 'scenarios' not in results:
        st.warning("No experiment results available.")
        return
    
    scenarios = results['scenarios']
    
    comparison_data = []
    for scenario in scenarios:
        name = scenario['scenario_name']
        agg = scenario['aggregated']
        
        mas_f1 = agg['mas'].get('detection', {}).get('f1', {}).get('mean', 0)
        baseline_f1 = agg['baseline'].get('detection', {}).get('f1', {}).get('mean', 0)
        mas_stability = agg['mas'].get('stability', {}).get('total_state_changes', {}).get('mean', 0)
        baseline_stability = agg['baseline'].get('stability', {}).get('total_state_changes', {}).get('mean', 0)
        
        comparison_data.append({
            'Scenario': name,
            'MAS F1': mas_f1,
            'Baseline F1': baseline_f1,
            'MAS State Changes': mas_stability,
            'Baseline State Changes': baseline_stability
        })
    
    df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='MAS', x=df['Scenario'], y=df['MAS F1'], marker_color='#3498DB'))
        fig.add_trace(go.Bar(name='Baseline', x=df['Scenario'], y=df['Baseline F1'], marker_color='#E74C3C'))
        fig.update_layout(
            title='F1 Score Comparison',
            barmode='group',
            yaxis_title='F1 Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='MAS', x=df['Scenario'], y=df['MAS State Changes'], marker_color='#3498DB'))
        fig.add_trace(go.Bar(name='Baseline', x=df['Scenario'], y=df['Baseline State Changes'], marker_color='#E74C3C'))
        fig.update_layout(
            title='Stability Comparison (State Changes)',
            barmode='group',
            yaxis_title='State Changes',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if 'summary' in results:
        st.markdown("### Summary")
        summary = results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAS Avg F1", f"{summary.get('mas_avg_f1', 0):.3f}")
        with col2:
            st.metric("Baseline Avg F1", f"{summary.get('baseline_avg_f1', 0):.3f}")
        with col3:
            improvement = summary.get('f1_improvement', 0)
            st.metric("F1 Improvement", f"{improvement:+.3f}", 
                     delta=f"{improvement*100:.1f}%")
        with col4:
            stability_imp = summary.get('stability_improvement', 0)
            st.metric("Stability Improvement", f"{stability_imp:+.1f}",
                     delta="fewer state changes" if stability_imp > 0 else "more changes")


def render_feature_importance(report: dict):
    """Render ML model feature importance."""
    st.subheader("🎯 Feature Importance")
    
    if not report or 'feature_importance' not in report:
        st.warning("No feature importance data available.")
        return
    
    importance = report['feature_importance']
    df = pd.DataFrame([
        {'Feature': k, 'Importance': v}
        for k, v in sorted(importance.items(), key=lambda x: -x[1])
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (ML Model)'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Model Type", report.get('model_type', 'N/A').upper())
        st.metric("AUC-ROC", f"{report.get('auc_roc', 0):.4f}")
        st.metric("F1 Score", f"{report.get('f1', 0):.4f}")
        st.metric("Calibrated", "Yes" if report.get('calibrated', False) else "No")


def render_robustness_analysis(results: dict):
    """Render robustness analysis vs dropout rate."""
    st.subheader("🛡️ Robustness Analysis")
    
    if not results or 'scenarios' not in results:
        st.warning("No experiment results available.")
        return
    
    dropout_data = []
    for scenario in results['scenarios']:
        config = scenario['scenario_config']
        dropout = config.get('dropout_rate', 0)
        agg = scenario['aggregated']
        mas_f1 = agg['mas'].get('detection', {}).get('f1', {}).get('mean', 0)
        baseline_f1 = agg['baseline'].get('detection', {}).get('f1', {}).get('mean', 0)
        
        dropout_data.append({
            'Dropout Rate': dropout,
            'MAS F1': mas_f1,
            'Baseline F1': baseline_f1
        })
    
    df = pd.DataFrame(dropout_data)
    df = df.groupby('Dropout Rate').mean().reset_index()
    
    if len(df) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Dropout Rate'],
            y=df['MAS F1'],
            mode='lines+markers',
            name='MAS',
            line=dict(color='#3498DB', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df['Dropout Rate'],
            y=df['Baseline F1'],
            mode='lines+markers',
            name='Baseline',
            line=dict(color='#E74C3C', width=3)
        ))
        fig.update_layout(
            title='Performance vs Sensor Dropout Rate',
            xaxis_title='Dropout Rate',
            yaxis_title='F1 Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough dropout variation in experiments for robustness analysis.")


def main():
    """Main dashboard entry point."""
    
    selected_log, selected_exp = render_sidebar()
    
    tabs = st.tabs(["📈 Simulation", "🔬 Experiments", "🤖 ML Model", "📋 Raw Data"])
    
    with tabs[0]:
        if selected_log:
            logs = load_simulation_logs(str(selected_log))
            if not logs.empty:
                st.success(f"Loaded {len(logs)} log entries from {selected_log.name}")
                
                render_timeline(logs)
                
                col1, col2 = st.columns(2)
                with col1:
                    render_confusion_matrix(logs)
                with col2:
                    render_lead_time_distribution(logs)
            else:
                st.warning("Log file is empty or could not be loaded.")
        else:
            st.info("👈 Select a simulation log from the sidebar to view results.")
            st.markdown("""
            ### Getting Started
            
            1. **Generate data**: `python -m ml.generate_data --episodes 500 --steps 300`
            2. **Train model**: `python -m ml.train --data outputs/datasets/sim.parquet`
            3. **Run simulation**: `python -m sim.model --model outputs/models/risk_model.pkl`
            4. **View results here!**
            """)
    
    with tabs[1]:
        if selected_exp:
            results = load_experiment_results(str(selected_exp))
            if results:
                st.success(f"Loaded experiment results from {selected_exp.name}")
                
                render_experiment_comparison(results)
                render_robustness_analysis(results)
            else:
                st.warning("Experiment file is empty or could not be loaded.")
        else:
            st.info("👈 Select experiment results from the sidebar to view comparison.")
            st.markdown("""
            ### Running Experiments
            
            ```bash
            python -m eval.run_experiments --config configs/scenarios.yaml \\
                --model outputs/models/risk_model.pkl \\
                --out outputs/experiments/results.json
            ```
            """)
    
    with tabs[2]:
        report_path = MODELS_DIR / "train_report.json"
        if report_path.exists():
            report = load_training_report(str(report_path))
            if report:
                st.success("Loaded ML training report")
                render_feature_importance(report)
                
                st.markdown("### Model Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision", f"{report.get('precision', 0):.4f}")
                with col2:
                    st.metric("Recall", f"{report.get('recall', 0):.4f}")
                with col3:
                    st.metric("Accuracy", f"{report.get('accuracy', 0):.4f}")
                with col4:
                    cv_auc = report.get('cv_auc_mean', 0)
                    cv_std = report.get('cv_auc_std', 0)
                    st.metric("CV AUC", f"{cv_auc:.4f} ± {cv_std:.4f}")
        else:
            st.info("No training report found. Train a model first:")
            st.code("python -m ml.train --data outputs/datasets/sim.parquet")
    
    with tabs[3]:
        if selected_log:
            logs = load_simulation_logs(str(selected_log))
            if not logs.empty:
                st.subheader("Simulation Logs")
                
                zone_filter = st.multiselect(
                    "Filter by Zone",
                    options=sorted(logs['zone_id'].unique()),
                    default=sorted(logs['zone_id'].unique())
                )
                
                filtered = logs[logs['zone_id'].isin(zone_filter)]
                st.dataframe(filtered, use_container_width=True)
                
                csv = filtered.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="simulation_logs.csv",
                    mime="text/csv"
                )
        else:
            st.info("Select a log file to view raw data.")


if __name__ == "__main__":
    main()
