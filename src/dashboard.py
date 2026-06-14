import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

st.set_page_config(layout="wide", page_title="Delhivery Graph Network", page_icon=":material/hub:")

st.title(":blue[:material/local_shipping: Delhivery Network Intelligence Dashboard]")
st.markdown("Monitor real-time delay risks, explore bottleneck hubs, compare model performance, and analyze corridor profiles using the graph-based framework.")

@st.cache_data
def load_data():
    edges_df = pd.read_csv("../data/processed/graph_edges.csv")
    nodes_df = pd.read_csv("../data/processed/graph_nodes.csv")
    metrics_df = pd.read_csv("../data/processed/node_metrics.csv")
    ftl_df = pd.read_csv("../data/processed/ftl_carting_framework.csv")
    return edges_df, nodes_df, metrics_df, ftl_df

try:
    edges_df, nodes_df, metrics_df, ftl_df = load_data()

    # ─── Sidebar Filters ───────────────────────────────────────────────────────
    st.sidebar.title("Filters")
    route_type = st.sidebar.selectbox("Route Type", ["All", "FTL", "Carting"])
    time_of_day = st.sidebar.selectbox("Time of Day", ["All", "Morning", "Afternoon", "Evening", "Night"])

    filtered_edges = edges_df.copy()
    if route_type != "All":
        filtered_edges = filtered_edges[filtered_edges['route_type'] == route_type]
    if time_of_day != "All":
        filtered_edges = filtered_edges[filtered_edges['time_of_day'] == time_of_day]

    # ─── Section 1: Network Overview ──────────────────────────────────────────
    st.markdown("---")
    st.subheader(f":blue[:material/router: Network Overview ({len(filtered_edges)} corridors)]")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Corridors", len(filtered_edges))
    with col2:
        avg_delay = filtered_edges['median_segment_factor'].mean()
        st.metric("Avg Delay Factor", f"{avg_delay:.2f}x")
    with col3:
        breach_pct = (filtered_edges['median_segment_factor'] > 1.2).mean() * 100
        st.metric("Corridors Breaching SLA", f"{breach_pct:.1f}%")

    # ─── Section 2: Top Bottleneck Hubs ───────────────────────────────────────
    st.markdown("---")
    st.subheader(":red[:material/warning: Top Bottleneck Hubs (SLA Breach Risk)]")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        top_hubs = (
            metrics_df[['node_name', 'betweenness_centrality', 'delayed_trips', 'sla_breach_score']]
            .sort_values('sla_breach_score', ascending=False)
            .head(10)
        )
        st.dataframe(
            top_hubs,
            column_config={
                "node_name": st.column_config.TextColumn("Hub Name", width="large"),
                "betweenness_centrality": st.column_config.NumberColumn("Betweenness Centrality", format="%.4f"),
                "delayed_trips": st.column_config.NumberColumn("Delayed Trips"),
                "sla_breach_score": st.column_config.ProgressColumn(
                    "SLA Breach Score", format="%.0f",
                    min_value=0, max_value=float(top_hubs['sla_breach_score'].max())
                ),
            },
            hide_index=True,
            use_container_width=True
        )

    with col_right:
        fig_bar = px.bar(
            top_hubs.head(5).sort_values('sla_breach_score'),
            x='sla_breach_score', y='node_name', orientation='h',
            title="Top 5 Hubs by SLA Breach Score",
            labels={'sla_breach_score': 'SLA Breach Score', 'node_name': 'Hub'},
            color='sla_breach_score', color_continuous_scale='Reds'
        )
        fig_bar.update_layout(template="plotly_white", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─── Section 2.5: Interactive Network Topology ───────────────────────────
    st.markdown("---")
    st.subheader(":violet[:material/hub: Interactive Network Topology]")
    st.markdown("Visualizing the top 150 most critical corridors by SLA breach risk.")
    
    # We build a subgraph to keep the visualization responsive in Streamlit
    risk_edges = filtered_edges[filtered_edges['median_segment_factor'] > 1.2].sort_values('trip_count', ascending=False).head(150)
    
    if len(risk_edges) > 0:
        G_vis = nx.from_pandas_edgelist(risk_edges, 'source_center', 'destination_center', ['median_segment_factor', 'trip_count'])
        pos = nx.spring_layout(G_vis, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in G_vis.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
            
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # Build node metadata lookup
        node_meta = metrics_df.set_index('node_id')
        
        for node in G_vis.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            try:
                score = node_meta.loc[node, 'sla_breach_score']
                name = node_meta.loc[node, 'node_name']
            except KeyError:
                score = 0
                name = str(node)
                
            node_text.append(f"{name}<br>SLA Score: {score:.1f}")
            node_size.append(10 + min(score / 200, 40) if score > 0 else 10)  # scale node size safely
            node_color.append(score)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Reds',
                reversescale=False,
                color=node_color,
                size=node_size,
                colorbar=dict(title='SLA Risk'),
                line_width=2))
                
        fig_net = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("No breached corridors found for the selected filters to visualize.")

    # ─── Section 3: Corridor Delay Heatmap ────────────────────────────────────
    st.markdown("---")
    st.subheader(":orange[:material/bar_chart: Corridor Delay Distribution]")

    if len(filtered_edges) > 0:
        fig_hist = px.histogram(
            filtered_edges,
            x='median_segment_factor', nbins=50,
            title="Distribution of Corridor Delay Factors",
            labels={'median_segment_factor': 'Median Segment Delay Factor'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig_hist.add_vline(x=1.2, line_dash="dash", line_color="red",
                           annotation_text="SLA Breach Threshold (>1.2x)")
        fig_hist.update_layout(bargap=0.1, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ─── Section 4: Model Performance Comparison ──────────────────────────────
    st.markdown("---")
    st.subheader(":green[:material/model_training: ETA Model Performance Comparison]")
    st.markdown("Graph-enhanced model vs. Baseline LightGBM — trained on 144,846 cleaned trips.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Baseline MAE", "54.58 mins")
    col_m2.metric("Graph-Enhanced MAE", "43.25 mins", delta="-11.33 mins", delta_color="normal")
    col_m3.metric("Baseline Accuracy@15%", "43.56%")
    col_m4.metric("Graph-Enhanced Accuracy@15%", "50.98%", delta="+7.42%", delta_color="normal")

    fig_model = go.Figure()
    fig_model.add_trace(go.Bar(name='Baseline LightGBM', x=['MAE (mins)', 'Accuracy@15% (%)'],
                                y=[54.58, 43.56], marker_color='#74B9FF'))
    fig_model.add_trace(go.Bar(name='Graph-Enhanced LightGBM', x=['MAE (mins)', 'Accuracy@15% (%)'],
                                y=[43.25, 50.98], marker_color='#00B894'))
    fig_model.update_layout(barmode='group', title="Baseline vs Graph-Enhanced Model Metrics",
                             template="plotly_white", yaxis_title="Score")
    st.plotly_chart(fig_model, use_container_width=True)

    # ─── Section 5: FTL vs Carting Strategy Framework ─────────────────────────
    st.markdown("---")
    st.subheader(":blue[:material/local_shipping: FTL vs. Carting Strategy Framework]")
    st.markdown("Data-driven corridor profile matrix quantifying time-cost trade-offs by distance and time of day.")

    # Reload raw edges for this section (unaffected by sidebar filter)
    strategy_df = edges_df.copy()
    strategy_df['dist_bucket'] = pd.qcut(strategy_df['median_segment_osrm_distance'], q=3, labels=['Short', 'Medium', 'Long'])
    strategy_grouped = strategy_df.groupby(['dist_bucket', 'time_of_day', 'route_type']).agg(
        avg_delay_factor=('median_segment_factor', 'mean')
    ).reset_index()

    fig_strat = px.bar(
        strategy_grouped,
        x='time_of_day', y='avg_delay_factor', color='route_type', barmode='group',
        facet_col='dist_bucket',
        title="Average Delay Factor: FTL vs. Carting by Distance and Time of Day",
        labels={'avg_delay_factor': 'Avg Delay Factor', 'time_of_day': 'Time of Day', 'route_type': 'Route Type'},
        color_discrete_map={'FTL': '#0984E3', 'Carting': '#E17055'}
    )
    fig_strat.add_hline(y=1.2, line_dash="dot", line_color="red",
                         annotation_text="SLA Threshold")
    fig_strat.update_layout(template="plotly_white")
    st.plotly_chart(fig_strat, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Processed data not found ({e}). Please run the data pipeline and graph analysis scripts first.")
