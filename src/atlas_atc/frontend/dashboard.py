"""ATLAS Dashboard for visualizing flight trajectories and predictions."""

import json
import logging
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from atlas_atc.config import settings
from atlas_atc.data.loader import SCATDataLoader

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ATLAS - Air Traffic Visualization",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_flight_list(data_path: Path, max_files: int = 100) -> list[tuple[str, str]]:
    """Load list of available flights."""
    loader = SCATDataLoader(data_path)
    flight_files = loader.trajectory_files[:max_files]

    flight_list = []
    for file_path in flight_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                fpl_base = data.get('fpl', {}).get('fpl_base', [{}])[0]
                callsign = fpl_base.get('callsign', 'Unknown')
                flight_id = data.get('id', file_path.stem)
                flight_list.append((str(flight_id), f"{flight_id} - {callsign}"))
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")

    return flight_list


@st.cache_data
def load_flight_data(data_path: Path, flight_id: str) -> dict[str, Any]:
    """Load data for a specific flight."""
    file_path = data_path / f"scat20161015_20161021/{flight_id}.json"
    with open(file_path) as f:
        return json.load(f)


def extract_trajectory_df(flight_data: dict[str, Any]) -> pd.DataFrame:
    """Extract trajectory data into a DataFrame."""
    plots = flight_data.get('plots', [])

    records = []
    for plot in plots:
        if 'I062/105' in plot:
            record = {
                'time': pd.to_datetime(plot.get('time_of_track')),
                'lat': plot['I062/105']['lat'],
                'lon': plot['I062/105']['lon'],
                'flight_level': plot.get('I062/136', {}).get('measured_flight_level', 0),
                'altitude_ft': plot.get('I062/136', {}).get('measured_flight_level', 0) * 100,
                'vx': plot.get('I062/185', {}).get('vx', 0),
                'vy': plot.get('I062/185', {}).get('vy', 0),
                'rocd': plot.get('I062/220', {}).get('rocd', 0),
                'heading': 0,
                'ias': 0,
                'mach': 0,
            }
            
            # Safely extract I062/380 fields
            i062_380 = plot.get('I062/380', {})
            if i062_380:
                # Heading
                subitem3 = i062_380.get('subitem3', {})
                if isinstance(subitem3, dict):
                    record['heading'] = subitem3.get('mag_hdg', 0)
                # IAS
                subitem26 = i062_380.get('subitem26', {})
                if isinstance(subitem26, dict):
                    record['ias'] = subitem26.get('ias', 0)
                # Mach
                subitem27 = i062_380.get('subitem27', {})
                if isinstance(subitem27, dict):
                    record['mach'] = subitem27.get('mach', 0)
            
            records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df['ground_speed_kts'] = np.sqrt(df['vx']**2 + df['vy']**2) * 1.94384  # m/s to knots
        df['time_elapsed'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds() / 60  # minutes

    return df


def extract_predicted_trajectory(flight_data: dict[str, Any]) -> pd.DataFrame:
    """Extract predicted trajectory waypoints."""
    pred_traj = flight_data.get('predicted_trajectory', [])

    records = []
    for traj in pred_traj:
        for waypoint in traj.get('route', []):
            record = {
                'time': pd.to_datetime(waypoint.get('eto')),
                'fix_name': waypoint.get('fix_name'),
                'fix_kind': waypoint.get('fix_kind'),
                'lat': waypoint.get('lat'),
                'lon': waypoint.get('lon'),
                'flight_level': waypoint.get('afl_value', 0),
                'altitude_ft': waypoint.get('afl_value', 0) * 100,
                'is_ato': waypoint.get('is_ato', False)
            }
            records.append(record)

    return pd.DataFrame(records)


def create_trajectory_map(actual_df: pd.DataFrame, predicted_df: pd.DataFrame) -> folium.Map:
    """Create a folium map with actual and predicted trajectories."""
    if actual_df.empty:
        return folium.Map(location=[59.0, 18.0], zoom_start=6)

    # Center map on trajectory
    center_lat = actual_df['lat'].mean()
    center_lon = actual_df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Add actual trajectory
    actual_coords = actual_df[['lat', 'lon']].values.tolist()
    folium.PolyLine(
        actual_coords,
        color='blue',
        weight=3,
        opacity=0.8,
        popup='Actual trajectory'
    ).add_to(m)

    # Add start and end markers
    folium.Marker(
        actual_coords[0],
        popup=f"Start: {actual_df['time'].iloc[0]}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    folium.Marker(
        actual_coords[-1],
        popup=f"End: {actual_df['time'].iloc[-1]}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

    # Add predicted trajectory if available
    if not predicted_df.empty:
        pred_coords = predicted_df[['lat', 'lon']].values.tolist()
        folium.PolyLine(
            pred_coords,
            color='orange',
            weight=2,
            opacity=0.6,
            dash_array='5, 5',
            popup='Predicted trajectory'
        ).add_to(m)

        # Add waypoints
        for _, waypoint in predicted_df.iterrows():
            if waypoint['is_ato']:
                folium.CircleMarker(
                    [waypoint['lat'], waypoint['lon']],
                    radius=3,
                    popup=f"{waypoint['fix_name']} ({waypoint['fix_kind']})",
                    color='orange',
                    fill=True
                ).add_to(m)

    return m


def main():
    """Main dashboard application."""
    st.title("✈️ ATLAS - Air Traffic Visualization Dashboard")
    st.markdown("Visualize and analyze aircraft trajectories from the SCAT dataset")

    # Sidebar
    with st.sidebar:
        st.header("Flight Selection")

        # Load flight list
        data_path = settings.DATA_DIR / "scat"

        if not data_path.exists():
            st.error(f"Data path {data_path} does not exist!")
            return

        flight_list = load_flight_list(data_path)

        if not flight_list:
            st.error("No flights found in dataset!")
            return

        # Flight selector
        selected_flight = st.selectbox(
            "Select a flight:",
            options=[f[0] for f in flight_list],
            format_func=lambda x: next(f[1] for f in flight_list if f[0] == x)
        )

        # Load flight data
        flight_data = load_flight_data(data_path, selected_flight)

        # Display flight info
        st.subheader("Flight Information")
        fpl_base = flight_data.get('fpl', {}).get('fpl_base', [{}])[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Callsign", fpl_base.get('callsign', 'N/A'))
            st.metric("Departure", fpl_base.get('adep', 'N/A'))
            st.metric("Aircraft", fpl_base.get('aircraft_type', 'N/A'))

        with col2:
            st.metric("Flight ID", flight_data.get('id', 'N/A'))
            st.metric("Destination", fpl_base.get('ades', 'N/A'))
            st.metric("WTC", fpl_base.get('wtc', 'N/A'))

    # Extract trajectory data
    actual_df = extract_trajectory_df(flight_data)
    predicted_df = extract_predicted_trajectory(flight_data)

    if actual_df.empty:
        st.error("No trajectory data available for this flight!")
        return

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🗺️ Trajectory Map", "📊 Altitude Profile", "🚀 Performance", "📍 Waypoints", "📈 Predictions"]
    )

    with tab1:
        st.subheader("Flight Trajectory")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Distance", f"{len(actual_df)} points")
        with col2:
            st.metric("Duration", f"{actual_df['time_elapsed'].max():.1f} min")
        with col3:
            st.metric("Max Altitude", f"{actual_df['altitude_ft'].max():,.0f} ft")

        # Create and display map
        trajectory_map = create_trajectory_map(actual_df, predicted_df)
        st_folium(trajectory_map, width=None, height=600)

    with tab2:
        st.subheader("Altitude Profile")

        fig = go.Figure()

        # Actual altitude
        fig.add_trace(go.Scatter(
            x=actual_df['time_elapsed'],
            y=actual_df['altitude_ft'],
            mode='lines',
            name='Actual Altitude',
            line=dict(color='blue', width=2)
        ))

        # Vertical rate on secondary axis
        fig.add_trace(go.Scatter(
            x=actual_df['time_elapsed'],
            y=actual_df['rocd'],
            mode='lines',
            name='Vertical Rate',
            line=dict(color='red', width=1, dash='dot'),
            yaxis='y2'
        ))

        fig.update_layout(
            title="Altitude and Vertical Rate Profile",
            xaxis_title="Time Elapsed (minutes)",
            yaxis_title="Altitude (feet)",
            yaxis2=dict(
                title="Vertical Rate (ft/min)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Altitude", f"{actual_df['altitude_ft'].mean():,.0f} ft")
        with col2:
            st.metric("Max Climb Rate", f"{actual_df['rocd'].max():,.0f} ft/min")
        with col3:
            st.metric("Max Descent Rate", f"{actual_df['rocd'].min():,.0f} ft/min")
        with col4:
            st.metric("Altitude Change", f"{actual_df['altitude_ft'].iloc[-1] - actual_df['altitude_ft'].iloc[0]:+,.0f} ft")

    with tab3:
        st.subheader("Aircraft Performance")

        # Speed and heading plot
        fig = go.Figure()

        # Ground speed
        fig.add_trace(go.Scatter(
            x=actual_df['time_elapsed'],
            y=actual_df['ground_speed_kts'],
            mode='lines',
            name='Ground Speed',
            line=dict(color='green', width=2)
        ))

        # IAS if available
        if actual_df['ias'].any():
            fig.add_trace(go.Scatter(
                x=actual_df['time_elapsed'],
                y=actual_df['ias'],
                mode='lines',
                name='Indicated Airspeed',
                line=dict(color='orange', width=2)
            ))

        # Heading on secondary axis
        fig.add_trace(go.Scatter(
            x=actual_df['time_elapsed'],
            y=actual_df['heading'],
            mode='lines',
            name='Heading',
            line=dict(color='purple', width=1),
            yaxis='y2'
        ))

        fig.update_layout(
            title="Speed and Heading",
            xaxis_title="Time Elapsed (minutes)",
            yaxis_title="Speed (knots)",
            yaxis2=dict(
                title="Heading (degrees)",
                overlaying='y',
                side='right',
                range=[0, 360]
            ),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Ground Speed", f"{actual_df['ground_speed_kts'].mean():.0f} kts")
        with col2:
            st.metric("Max Ground Speed", f"{actual_df['ground_speed_kts'].max():.0f} kts")
        with col3:
            if actual_df['mach'].any():
                st.metric("Max Mach", f"{actual_df['mach'].max():.2f}")
            else:
                st.metric("Max Mach", "N/A")
        with col4:
            heading_change = abs(actual_df['heading'].diff()).sum()
            st.metric("Total Heading Change", f"{heading_change:.0f}°")

    with tab4:
        st.subheader("Waypoints and Route")

        if not predicted_df.empty:
            # Filter for active waypoints
            active_waypoints = predicted_df[predicted_df['is_ato']]

            # Display waypoints table
            st.dataframe(
                active_waypoints[['time', 'fix_name', 'fix_kind', 'flight_level', 'lat', 'lon']],
                use_container_width=True,
                hide_index=True
            )

            # Plot altitude at waypoints
            fig = px.line(
                active_waypoints,
                x='fix_name',
                y='altitude_ft',
                title='Planned Altitude at Waypoints',
                markers=True
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No waypoint data available for this flight")

    with tab5:
        st.subheader("Trajectory Predictions")

        # Compare actual vs predicted if we have both
        if not predicted_df.empty and not actual_df.empty:
            # Create comparison visualization
            fig = go.Figure()

            # Actual trajectory
            fig.add_trace(go.Scattergeo(
                lat=actual_df['lat'],
                lon=actual_df['lon'],
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=3)
            ))

            # Predicted trajectory
            fig.add_trace(go.Scattergeo(
                lat=predicted_df['lat'],
                lon=predicted_df['lon'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title="Actual vs Predicted Trajectory",
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    coastlinecolor='rgb(204, 204, 204)',
                    showlakes=True,
                    lakecolor='rgb(255, 255, 255)',
                    showcountries=True,
                    countrycolor='rgb(204, 204, 204)',
                    center=dict(
                        lat=actual_df['lat'].mean(),
                        lon=actual_df['lon'].mean()
                    ),
                    projection_scale=3
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate prediction accuracy metrics if possible
            st.subheader("Prediction Analysis")
            st.info("Prediction accuracy analysis would require matching time points between actual and predicted trajectories")
        else:
            st.info("No prediction data available for comparison")

    # Footer
    st.markdown("---")
    st.markdown("**ATLAS** - Air Traffic Learning & Analytics System | SCAT Dataset Visualization")


if __name__ == "__main__":
    main()
