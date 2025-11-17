import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
import os

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Metro Manila Traffic Incident Monitor",
    page_icon="🚦",
    layout="wide"
)

# DATA LOADING & CLEANING

@st.cache_data
def load_and_clean_data():
    # --- DOWNLOAD VIA KAGGLEHUB ---
    with st.spinner("Downloading and combining all data files..."):
        try:
            # Download the MMDA dataset
            path = kagglehub.dataset_download("esparko/mmda-traffic-incident-data")
            
            # Find ALL CSV files in the folder (handles multiple years)
            all_csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
            
            if not all_csv_files:
                st.error("No CSV files found in the downloaded dataset.")
                st.stop()
            
            # Read and combine all CSVs into one DataFrame
            df_list = []
            for file in all_csv_files:
                # Unicode escape handles encoding errors common in this dataset
                try:
                    temp_df = pd.read_csv(file, encoding='unicode_escape')
                    df_list.append(temp_df)
                except Exception as e:
                    st.warning(f"Could not read file {file}: {e}")
            
            if not df_list:
                st.error("No valid data files could be read.")
                st.stop()

            df = pd.concat(df_list, ignore_index=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    # --- DATA CLEANING ---
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Fix City Names (Force String to prevent sorting errors)
    if 'City' in df.columns:
        df['City'] = df['City'].fillna('Unknown').astype(str).str.title().str.strip()
    
    # Fix Date (Handle mixed formats)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Fix Time and Hour extraction
    df['Time'] = pd.to_datetime(df['Time'], format='%I:%M %p', errors='coerce').dt.time
    df['Hour'] = pd.to_datetime(df['Time'].astype(str), errors='coerce').dt.hour
    
    # Fix Vehicle Types
    if 'Involved' in df.columns:
        df['Involved'] = df['Involved'].fillna("Unknown").astype(str)
    
    # Ensure Coordinates exist for the map
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Create Month_Year for trend lines
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)

    return df

# Load Data
df = load_and_clean_data()

# SIDEBAR FILTERS

st.sidebar.header("Filter Traffic Data")

# Date Filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# City Filter
cities = sorted(df['City'].unique().tolist())
selected_cities = st.sidebar.multiselect("Select City", cities, default=cities[:5])

# --- SMART DATE HANDLING (Fixes Index Error) ---
if len(date_range) == 2:
    start_date = date_range[0]
    end_date = date_range[1]
elif len(date_range) == 1:
    # If user picked only one date so far, treat it as Start=End
    start_date = date_range[0]
    end_date = date_range[0]
else:
    start_date = min_date
    end_date = max_date

# Apply Filters
mask = (
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date) &
    (df['City'].isin(selected_cities))
)
filtered_df = df.loc[mask]


# MAIN DASHBOARD

st.title("🚦 Metro Manila Accident Dashboard")
st.markdown("Analyzed from MMDA Traffic Incident Data via KaggleHub")

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)

total_incidents = len(filtered_df)
top_city = filtered_df['City'].mode()[0] if not filtered_df.empty else "N/A"
top_incident = filtered_df['Type'].mode()[0] if not filtered_df.empty and 'Type' in filtered_df.columns else "N/A"
peak_hour_val = int(filtered_df['Hour'].mode()[0]) if not filtered_df.empty else "N/A"
peak_hour_str = f"{peak_hour_val}:00" if peak_hour_val != "N/A" else "N/A"

# Metric 1
col1.metric("Total Incidents", f"{total_incidents:,}")

# Metric 2
col2.metric("Most Dangerous City", top_city)

# Metric 3: 
with col3:
    # Calculate font size based on text length
    text_len = len(str(top_incident))
    if text_len > 20:
        font_size = "18px"
    elif text_len > 12:
        font_size = "24px"
    else:
        font_size = "32px" 

    st.markdown(
        f"""
        <div style="margin-bottom: 0;">
            <p style="font-size: 14px; margin-bottom: 0; opacity: 0.7;">Most Common Incident</p>
            <p style="font-size: {font_size}; font-weight: 700; margin: 0; line-height: 1.2;">
                {top_incident}
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Metric 4
col4.metric("Peak Hour", peak_hour_str)

st.markdown("---")

# --- ROW 1: MAP & TIME SERIES ---
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("📍 Incident Hotspots")
    if not filtered_df.empty:
        # Using Plotly Density Mapbox 
        fig_map = px.density_mapbox(
            filtered_df,
            lat='Latitude',
            lon='Longitude',
            z=None, # Heatmap based on density of points
            radius=15,
            center=dict(lat=filtered_df['Latitude'].mean(), lon=filtered_df['Longitude'].mean()),
            zoom=10,
            mapbox_style="carto-positron",
            height=450
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

with row1_col2:
    st.subheader("📈 Trend Over Time")
    if not filtered_df.empty:
        # Convert period to string for plotting compatibility
        trend_data = filtered_df.groupby('Month_Year').size().reset_index(name='Count')
        trend_data['Month_Year'] = trend_data['Month_Year'].astype(str)
        
        fig_trend = px.line(trend_data, x='Month_Year', y='Count', markers=True, template="plotly_dark")
        st.plotly_chart(fig_trend, use_container_width=True)

# --- ROW 2: HEATMAP & VEHICLES ---
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("⏰ The 'Danger Hour' Matrix")
    if not filtered_df.empty:
        filtered_df['Day_Name'] = filtered_df['Date'].dt.day_name()
        heatmap_data = filtered_df.groupby(['Day_Name', 'Hour']).size().reset_index(name='Count')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig_heat = px.density_heatmap(
            heatmap_data, 
            x='Hour', 
            y='Day_Name', 
            z='Count', 
            nbinsx=24,
            category_orders={'Day_Name': days_order},
            color_continuous_scale='Reds',
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

with row2_col2:
    st.subheader("🚗 Vehicles Involved")
    if not filtered_df.empty:
        # Get top 10 most common vehicle involvements
        veh_counts = filtered_df['Involved'].value_counts().head(10).reset_index()
        veh_counts.columns = ['Vehicle/s', 'Count']
        
        fig_bar = px.bar(veh_counts, x='Count', y='Vehicle/s', orientation='h', color='Count', template="plotly_dark")
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- RAW DATA EXPANDER ---
with st.expander("📂 View Raw Data Table"):
    st.dataframe(filtered_df)