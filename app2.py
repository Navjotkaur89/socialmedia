import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Pie


# Set page config
st.set_page_config(layout="wide", page_title="Global AI Content Impact Analysis")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Global_AI_Content_Impact_Dataset.csv")

df = load_data()

# Title
st.title("🌍 Global AI Content Impact Analysis")
st.markdown("Exploring the influence of AI-generated content across industries and countries")

# Sidebar filters
st.sidebar.header("Filters")

# Get unique values
years = sorted(df['Year'].unique())
countries = sorted(df['Country'].unique())
industries = sorted(df['Industry'].unique())

# Select all as default
default_years = years
default_countries = countries
default_industries = industries

selected_years = st.sidebar.multiselect(
    "Select Years", 
    options=years,
    default=default_years,
    key="years_multiselect"
)

selected_countries = st.sidebar.multiselect(
    "Select Countries", 
    options=countries,
    default=default_countries,
    key="countries_multiselect"
)

selected_industries = st.sidebar.multiselect(
    "Select Industries", 
    options=industries,
    default=default_industries,
    key="industries_multiselect"
)

selected_industries = st.sidebar.multiselect(
    "Select Industries", 
    options=industries,
    default=default_industries
)

# Apply filters
filtered_df = df[
    (df['Year'].isin(selected_years)) &
    (df['Country'].isin(selected_countries)) &
    (df['Industry'].isin(selected_industries))
]

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🌐 Geographical", 
    "🏭 Industry", 
    "📈 Trends", 
    "🔍 Deep Dive"
])

with tab1:
    st.header("Dataset Overview")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_adoption = filtered_df['AI Adoption Rate (%)'].mean()
        st.metric("Average AI Adoption", f"{avg_adoption:.1f}%")
    with col2:
        avg_job_loss = filtered_df['Job Loss Due to AI (%)'].mean()
        st.metric("Average Job Impact", f"{avg_job_loss:.1f}%")
    with col3:
        avg_revenue = filtered_df['Revenue Increase Due to AI (%)'].mean()
        st.metric("Average Revenue Impact", f"{avg_revenue:.1f}%")
    with col4:
        avg_trust = filtered_df['Consumer Trust in AI (%)'].mean()
        st.metric("Average Consumer Trust", f"{avg_trust:.1f}%")
    
    st.markdown("---") 
    
    # Second row KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_content = filtered_df['AI-Generated Content Volume (TBs per year)'].sum()
        st.metric("Total AI Content Volume", f"{total_content:,.0f} TB/year")
    with col2:
        avg_collab = filtered_df['Human-AI Collaboration Rate (%)'].mean()
        st.metric("Avg Collaboration Rate", f"{avg_collab:.1f}%")
    with col3:
        num_countries = filtered_df['Country'].nunique()
        st.metric("Countries Covered", num_countries)
    with col4:
        num_industries = filtered_df['Industry'].nunique()
        st.metric("Industries Covered", num_industries)
    
    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(filtered_df, height=300)
    st.markdown("---")
    
    # Data distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df, 
                          x="AI Adoption Rate (%)",
                          title="AI Adoption Rate Distribution",
                          color_discrete_sequence=["#1292ED"],
                          nbins=20,  
                          opacity=0.8,
                          barmode='overlay')  
        fig.update_layout(bargap=0.2)  
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = px.histogram(filtered_df,
                          x="Job Loss Due to AI (%)",
                          title="Job Loss Due to AI Distribution",
                          color_discrete_sequence=["#ff7f0e"],
                          nbins=15, 
                          opacity=0.8,
                          barmode='overlay')  
        fig.update_layout(bargap=0.3)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Geographical Analysis")
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    country_means = filtered_df.groupby('Country')[numeric_cols].mean().reset_index()

    fig = px.choropleth(
        country_means,
        locations="Country",
        locationmode="country names",
        color="AI Adoption Rate (%)",
        title="Average AI Adoption Rate by Country",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=700, width=1000)
    st.plotly_chart(fig, use_container_width=True)

    content_volume = filtered_df.groupby('Country')['AI-Generated Content Volume (TBs per year)'].mean().reset_index()
    fig = px.bar(content_volume.sort_values('AI-Generated Content Volume (TBs per year)', ascending=False),
                x="Country", y="AI-Generated Content Volume (TBs per year)",
                title="AI-Generated Content Volume by Country",
                color_discrete_sequence=["#dfed1f"])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Country Comparison")
    metric = st.selectbox("Select metric to compare", 
                         ["AI Adoption Rate (%)", 
                          "Job Loss Due to AI (%)",
                          "Revenue Increase Due to AI (%)",
                          "Human-AI Collaboration Rate (%)",
                          "Consumer Trust in AI (%)"])
    
    country_metric = filtered_df.groupby('Country')[metric].mean().reset_index()
    fig = px.bar(country_metric.sort_values(metric, ascending=False),
                x="Country", y=metric,
                title=f"{metric} by Country",
                color_discrete_sequence=["#855bef"])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Industry Analysis")
    
    fig = px.sunburst(filtered_df, path=['Industry', 'Country'], 
                     values='AI-Generated Content Volume (TBs per year)',
                     title="AI Content Volume by Industry and Country",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=600, width=700)
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(filtered_df, x="Industry", y="AI Adoption Rate (%)",
                title="AI Adoption Rate Distribution by Industry",
                color_discrete_sequence=["#9467bd"])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Industry Comparison")
    metric = st.selectbox("Select metric for industry comparison", 
                         ["AI Adoption Rate (%)", 
                          "Job Loss Due to AI (%)",
                          "Revenue Increase Due to AI (%)",
                          "Human-AI Collaboration Rate (%)",
                          "Consumer Trust in AI (%)"])
    
    industry_metric = filtered_df.groupby('Industry')[metric].mean().reset_index()
    fig = px.bar(industry_metric.sort_values(metric, ascending=False),
                x="Industry", y=metric,
                title=f"{metric} by Industry",
                color_discrete_sequence=["#a2f088"])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Temporal Trends")
    
    country_trend = filtered_df.groupby(['Year', 'Country'])['AI Adoption Rate (%)'].mean().reset_index()
    fig = px.line(country_trend,
                 x="Year", y="AI Adoption Rate (%)", color="Country",
                 title="AI Adoption Rate Over Time by Country",
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig, use_container_width=True)
        
    industry_trend = filtered_df.groupby(['Year', 'Industry'])['AI Adoption Rate (%)'].mean().reset_index()
    fig = px.line(industry_trend,
                 x="Year", y="AI Adoption Rate (%)", color="Industry",
                 title="AI Adoption Rate Over Time by Industry",
                 color_discrete_sequence=px.colors.qualitative.Light24)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Metric Trends Over Time")
    metric = st.selectbox("Select metric to analyze over time", 
                         ["AI Adoption Rate (%)", 
                          "Job Loss Due to AI (%)",
                          "Revenue Increase Due to AI (%)",
                          "Human-AI Collaboration Rate (%)",
                          "Consumer Trust in AI (%)"])
    
    group_by = st.radio("Group by", ["Country", "Industry"])
    time_trend = filtered_df.groupby(['Year', group_by])[metric].mean().reset_index()
    fig = px.line(time_trend,
                 x="Year", y=metric, color=group_by,
                 title=f"{metric} Over Time by {group_by}",
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Deep Dive Analysis")
    
    st.subheader("Correlation Analysis")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis variable", 
                             ["AI Adoption Rate (%)", 
                              "AI-Generated Content Volume (TBs per year)",
                              "Job Loss Due to AI (%)",
                              "Revenue Increase Due to AI (%)",
                              "Human-AI Collaboration Rate (%)",
                              "Consumer Trust in AI (%)"])
    with col2:
        y_axis = st.selectbox("Y-axis variable", 
                             ["AI Adoption Rate (%)", 
                              "AI-Generated Content Volume (TBs per year)",
                              "Job Loss Due to AI (%)",
                              "Revenue Increase Due to AI (%)",
                              "Human-AI Collaboration Rate (%)",
                              "Consumer Trust in AI (%)"])
    
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color="Industry",
                    hover_data=["Country", "Year"],
                    title=f"Relationship between {x_axis} and {y_axis}",
                    color_discrete_sequence=px.colors.qualitative.G10)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("AI Tool Popularity")
    tool_counts = filtered_df['Top AI Tools Used'].value_counts().reset_index()
    tool_counts.columns = ['Tool', 'Count']
    fig = px.bar(tool_counts, x='Tool', y='Count',
                title="Most Used AI Tools Overall",
                color_discrete_sequence=["#e8bbdb"])
    st.plotly_chart(fig, use_container_width=True)
    
    # col1, col2 = st.columns(2)
  

# with col1:
    selected_industry = st.selectbox(
        "Select industry for tool analysis", 
        filtered_df['Industry'].unique()
    )
    
    # Prepare the data
    industry_tools = filtered_df[filtered_df['Industry'] == selected_industry]
    tool_counts = industry_tools['Top AI Tools Used'].value_counts().reset_index()
    tool_counts.columns = ['Tool', 'Count']
    
    # Convert to ECharts format (list of (name, value) tuples)
    data = [(row['Tool'], row['Count']) for _, row in tool_counts.iterrows()]
    
    # Create ECharts pie chart
    pie = (
        Pie()
        .add(
            series_name="Tools",
            data_pair=data,
            radius=["30%", "70%"],  # Makes it a donut chart
            label_opts=opts.LabelOpts(
                formatter="{b}: {c} ({d}%)",  # Shows name, count, and percentage
                position="outside"
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Tool Usage in {selected_industry} Industry",
                pos_left="center"
            ),
            legend_opts=opts.LegendOpts(
                orient="vertical",
                pos_top="15%",
                pos_left="2%"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{a} <br/>{b}: {c} ({d}%)"
            ),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=True),
            itemstyle_opts={
                "borderRadius": 6,
                "borderColor": "#fff",
                "borderWidth": 2
            }
        )
    )
    
    # Display in Streamlit
    st_pyecharts(pie, height="500px")
        
    # with col2:
    selected_country = st.selectbox(
        "Select country for tool analysis", 
        filtered_df['Country'].unique()
    )
    
    # Prepare the data
    country_tools = filtered_df[filtered_df['Country'] == selected_country]
    tool_counts = country_tools['Top AI Tools Used'].value_counts().reset_index()
    tool_counts.columns = ['Tool', 'Count']
    
    # Convert to ECharts format (list of (name, value) tuples)
    data = [(row['Tool'], row['Count']) for _, row in tool_counts.iterrows()]
    
    # Create ECharts pie chart with Set3-like colors
    pie = (
        Pie()
        .add(
            series_name="Tools",
            data_pair=data,
            radius=["30%", "70%"],
            label_opts=opts.LabelOpts(
                formatter="{b}: {c} ({d}%)",
                position="outside"
            ),
            # Using colors similar to Plotly's Set3
            color=[
                "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", 
                "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
                "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"
            ]
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Tool Usage in {selected_country}",
                pos_left="center"
            ),
            legend_opts=opts.LegendOpts(
                orient="vertical",
                pos_top="15%",
                pos_left="2%"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{a} <br/>{b}: {c} ({d}%)"
            ),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=True),
            itemstyle_opts={
                "borderRadius": 6,
                "borderColor": "#fff",
                "borderWidth": 2
            }
        )
    )
    
    # Display in Streamlit
    st_pyecharts(pie, height="500px")
    st.subheader("Regulation Impact")
    fig = px.box(filtered_df, x="Regulation Status", y="AI Adoption Rate (%)",
                title="AI Adoption Rate by Regulation Strictness",
                color_discrete_sequence=["#17becf"])
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(filtered_df, x="Regulation Status", y="Job Loss Due to AI (%)",
                title="Job Loss Due to AI by Regulation Strictness",
                color_discrete_sequence=["#bcbd22"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap (All Metrics)")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    st.pyplot(plt)