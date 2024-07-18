

import openai
import streamlit as st
import pandas as pd
#import pandas_profiling  
import plotly.express as px
import numpy as npS
from jira import JIRA
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from similarity import preprocess_data, calculate_similarity
import requests
import subprocess
#from neo4j import GraphDatabase, basic_auth
import plotly.graph_objects as go
#import streamlit_pandas_profiling
#from streamlit_pandas_profiling import st_profile_report
#import neo4j
#from neo4j_integration import Neo4jManager
import io  
import csv
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from io import BytesIO
import os
import streamlit as st
import pandas as pd
import base64
from io import BytesIO

# Download stopwords if not already downloaded
nltk.download('stopwords')

def to_json(df):
    """
    Converts a DataFrame to a JSON string.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to convert.
    
    Returns:
    - str: The JSON string.
    """
    return df.to_json(orient='records', lines=True)


def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue().decode()
    return processed_data




def read_csv_files(uploaded_file_iterative, uploaded_file_eigen):
    try:
        Iterative = pd.read_csv(uploaded_file_iterative, encoding='iso-8859-1')
        Eigen = pd.read_csv(uploaded_file_eigen, encoding='iso-8859-1')
        return Iterative, Eigen
    except Exception as e:
        st.warning(f"Error reading the files: {e}")
        return None, None

sprint_bins = [
    ('09 Nov 2022', '30 Nov 2022'),
    ('01 Dec 2022', '21 Dec 2022'),
    ('05 Jan 2023', '25 Jan 2023'),
    ('26 Jan 2023', '15 Feb 2023'),
    ('16 Feb 2023', '08 Mar 2023'),
    ('09 Mar 2023', '29 Mar 2023'),
    ('30 Mar 2023', '19 Apr 2023'),
    ('20 Apr 2023', '10 May 2023'),
    ('11 May 2023', '31 May 2023'),
    ('01 June 2023', '21 June 2023'),
    ('22 June 2023', '12 July 2023'),
    ('13 July 2023', '02 August 2023'),
    ('03 August 2023', '23 August 2023'),
    ('24 August 2023', '13 September 2023'),
    ('14 September 2023', '04 Oct 2023'),
    ('05 October 2023', '25 October 2023'),
    ('26 October 2023', '15 November 2023'),
    ('16 Nov 2023', '06 Dec 2023'),
    ('07 Dec 2023', '20 Dec 2023'),
    ('21 Dec 2023', '10 Jan 2024'),
    ('11 Jan 2024', '31 Jan 2024'),
    ('1 Feb 2024', '21 Feb 2024'),
    ('22 Feb 2024', '13 Mar 2024'),
    ('14 Mar 2024', '3 Apr 2024'),
    ('4 Apr 2024', '24 Apr 2024'),
    ('25 Apr 2024', '15 May 2024'),
    ('16 May 2024', '05 June 2024'),
    ('06 June 2024', '26 June 2024'),
    ('27 June 2024', '18 July 2024'),



    

    
]



# Convert the date strings to datetime objects
sprint_bins = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in sprint_bins]

def get_sprint(date, sprint_bins):
    for i, (start_date, end_date) in enumerate(sprint_bins):
        if start_date <= date <= end_date:
            return f"Sprint {80 + i:03d}"
    return None





def extract_key_ID(df):
    df = df.copy()  # Make a copy to work with
    
    # Extracting the pattern after F# inside []
    df['Issue key'] = df['Issue summary'].str.extract('\[F#(\d+)\]', expand=False)
    
    # For rows where 'Issue key' is NaN, extracting the pattern inside []
    mask = df['Issue key'].isna()
    df.loc[mask, 'Issue key'] = df.loc[mask, 'Issue summary'].str.extract('\[(.*?)\]', expand=False)
    
    df = df.drop('Issue summary', axis=1)
    return df


def similarity_func(df):
    with st.expander("Similarity Functionality"):
        st.subheader("Similarity Results")

        # Use session state to retrieve columns, remove the 'Save Columns' button
        text_column = st.session_state.get('text_column', df.columns[0])
        identifier_column = st.session_state.get('identifier_column', df.columns[0])
        additional_columns = st.session_state.get('additional_columns', df.columns[0])

        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)

        if st.button('Start Similarity Analysis'):
            # Check if columns exist
            if set([st.session_state.text_column, st.session_state.identifier_column] + st.session_state.additional_columns).issubset(set(df.columns)):
                try:
                    # Ensure text_column is of string type
                    df[st.session_state.text_column] = df[st.session_state.text_column].astype(str)
                    
                    # Preprocess and calculate similarity
                    preprocessed_data = preprocess_data(df, st.session_state.text_column)
                    similar_pairs = calculate_similarity(df, threshold, st.session_state.identifier_column, st.session_state.text_column, st.session_state.additional_columns)

                    # Diagnostic outputs
                    st.write(f"Number of rows in the original data: {len(preprocessed_data)}")
                    st.write(f"Number of similar pairs found: {len(similar_pairs)}")

                    # Display similarity results
                    st.subheader(f"Similarity Threshold: {threshold}")
                    st.dataframe(similar_pairs)
                except Exception as e:
                    st.error(f"Error running similarity analysis. Error: {str(e)}")
            else:
                st.error("Selected columns are not present in the data. Please check the column names and try again.")





def preprocess___data(Iterative, Eigen):
    # For Iterative dataframe
    if 'Issue summary' in Iterative.columns:
        Iterative = Iterative[['Issue summary', 'Hours', 'Full name', 'Work date', 'CoreTime', 'Issue Type', 'Issue Status']]
        Iterative = extract_key_ID(Iterative)
        Iterative = Iterative.rename(columns={'Full name': 'Assignee'})
    else:
        st.warning("Issue summary not found in Iterative file.")
        return None, None

    # For Eigen dataframe
    eigen_columns = ['Summary', 'Issue key', 'Status', 'Resolved', 'Epic Link Summary', 'Labels', 'Priority', 'Issue Type', 
                    'Assignee', 'Creator', 'Sprint', 'Created', 'Resolution', 'Custom field (Story Points)',
                    'Custom field (CoreTimeActivity)', 'Custom field (CoreTimeClient)', 'Custom field (CoreTimePhase)', 
                    'Custom field (CoreTimeProject)', 'Project name']

    if all(col in Eigen for col in eigen_columns):
        Eigen = Eigen[eigen_columns]
        Eigen = Eigen.rename(columns={
            'Custom field (CoreTimeActivity)': 'CoreTimeActivity', 
            'Custom field (CoreTimeClient)': 'CoreTimeClient',
            'Custom field (CoreTimePhase)': 'CoreTimePhase', 
            'Custom field (CoreTimeProject)': 'CoreTimeProject', 
            'Epic Link Summary': 'Epic', 
            'Custom field (Story Points)': 'Story Points'
        })
    else:
        st.warning("Some necessary columns are missing in Eigen file.")
        return None, None

    return Iterative, Eigen







def merge_data(Iterative, Eigen, sprint_bins):
    merged_df = pd.merge(Iterative, Eigen, on='Issue key', how='outer')  # Changed 'left' to 'outer' here
    merged_df = pd.merge(Iterative, Eigen, on='Issue key', how='left')  # Using 'left' merge here
    merged_df = merged_df.rename(columns={'Assignee_x': 'Assignee', 'Issue Type_x': 'Issue Type'})
    merged_df = merged_df.drop('Issue Type_y', axis=1)
    merged_df = merged_df[['Issue key', 'Work date', 'Assignee', 'Sprint', 'Status', 'Priority', 'Project name', 'Issue Type',
                           'Creator', 'Created', 'Resolution', 'Story Points', 'Hours', 'Resolved', 'Epic', 'CoreTimeActivity', 
                           'CoreTimeClient', 'CoreTimePhase', 'CoreTimeProject', 'Labels']]
    

    grouped_df = merged_df.groupby('Issue key').agg({'Hours': 'sum', 'Story Points': 'first', 'Assignee': 'first', 
                                                     'Sprint': 'first', 'Resolution': 'first', 'Resolved': 'first', 
                                                     'Created': 'first', 'Priority': 'first', 'Creator': 'first', 
                                                     'CoreTimeActivity': 'first', 'CoreTimeClient': 'first', 
                                                     'CoreTimePhase': 'first', 'Epic': 'first', 'CoreTimeProject': 'first', 
                                                     'Project name': 'first', 'Status': 'first', 'Labels': 'first', 
                                                     'Work date': 'first', 'Issue Type': 'first'}).reset_index()
    
    grouped_df['days'] = grouped_df['Hours'] / 8
    grouped_df['Avg_Ratio'] = grouped_df['Story Points'] / grouped_df['days']
    grouped_df['Work date'] = pd.to_datetime(grouped_df['Work date'])
    grouped_df['Created'] = pd.to_datetime(grouped_df['Created'])
    grouped_df['Sprint'] = grouped_df['Created'].apply(lambda x: get_sprint(x, sprint_bins))
    return grouped_df

def read_json_files(uploaded_file_iterative, uploaded_file_eigen):
    try:
        Iterative = pd.read_json(uploaded_file_iterative, lines=True)  # Assuming the JSON is in a line-delimited format
        Eigen = pd.read_json(uploaded_file_eigen, lines=True)  # Adjust the parameters if your JSON format is different
        return Iterative, Eigen
    except Exception as e:
        st.warning(f"Error reading the files: {e}")
        return None, None


def load_data(uploaded_file_iterative, uploaded_file_eigen, sprint_bins):
    if uploaded_file_iterative is not None and uploaded_file_eigen is not None:
        # Determine the file type by extension
        if uploaded_file_iterative.name.endswith('.csv') and uploaded_file_eigen.name.endswith('.csv'):
            Iterative, Eigen = read_csv_files(uploaded_file_iterative, uploaded_file_eigen)
        elif uploaded_file_iterative.name.endswith('.json') and uploaded_file_eigen.name.endswith('.json'):
            Iterative, Eigen = read_json_files(uploaded_file_iterative, uploaded_file_eigen)
        else:
            st.error("Unsupported file format. Please upload either CSV or JSON files.")
            return None

        if Iterative is not None and Eigen is not None:
            Iterative, Eigen = preprocess___data(Iterative, Eigen)
            if Iterative is not None and Eigen is not None:
                df = merge_data(Iterative, Eigen, sprint_bins)
                
                # Save DataFrame to CSV
                csv_data = df.to_csv(index=False)
                
                # Add download button for CSV
                st.sidebar.download_button(
                    label="Download processed data table as CSV",
                    data=csv_data,
                    file_name='preprocessed_data.csv',
                    mime='text/csv'
                )
                

                return df
            else:
                return None
        else:
            return None
    else:
        st.warning("Please upload both files to proceed!")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------#

# Constants
DEFAULT_BG_COLOR = 'rgba(0, 0, 0, 0)'

def get_assignee_rates(df):
    """
    Get assignee rates from the DataFrame.
    """
    assignee_rates = {}
    # Check for assignees with missing rates or assignees not in the dataframe and set them to default
    for assignee, default_rate in DEFAULT_RATES.items():
        if assignee not in assignee_rates or assignee_rates[assignee] is None:
            assignee_rates[assignee] = default_rate

    return assignee_rates




def display_assignee_rates(assignee_rates):
    """
    Display the rates of assignees in the sidebar.
    """
    for assignee, rate in assignee_rates.items():
        st.sidebar.write(f"Rate for Assignee {assignee}: {rate}")









def update_chart_layout(chart, title, height=None, width=None):
    """
    Update chart layout and styling.
    """
    chart.update_layout(
        height=height,
        width=width,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor=DEFAULT_BG_COLOR,
        paper_bgcolor=DEFAULT_BG_COLOR,
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black', width=1),
                fillcolor=DEFAULT_BG_COLOR
            )
        ],
        title=title
    )


def create_combined_chart(df, sprint_summary, sprint_avg_ratio):
    # Create the combined chart
    combined_chart = go.Figure()
    combined_chart.add_trace(go.Bar(
        x=sprint_summary['Sprint'],
        y=sprint_summary['Story Points'],
        name='Story Points',
        text=sprint_summary['Story Points'],
        textposition='auto'
    ))
    combined_chart.add_trace(go.Bar(
        x=sprint_summary['Sprint'],
        y=sprint_summary['days'],
        name='days',
        text=sprint_summary['days'],
        textposition='auto'
    ))

    # Update the chart layout and styling
    combined_chart.update_layout(
        barmode='group',
        title='Delivered Story Points vs. Worked Man-Days by Sprint',
        xaxis=dict(title='Sprint'),
        yaxis=dict(title='Value'),  # Changed the y-axis title since it's no longer normalized
        legend=dict(title='Metrics'),
        font=dict(color='black')
    )

    # Apply styling to the bars and markers
    for trace in combined_chart.data:
        if 'marker' in trace and 'line' in trace['marker']:
            trace.marker.line.color = 'black'
        if 'textfont' in trace:
            trace.textfont.color = 'black'
            trace.textfont.size = 14

    return combined_chart


def display_tab1(df, assignee_rates):
    """
    Display visualizations and data for Tab 1.
    """
    df['Work date'] = pd.to_datetime(df['Work date'], infer_datetime_format=True)
    df['Cost'] = df['Hours'] * df['Assignee'].map(assignee_rates)

    epic_cost_data = df.groupby(['Epic', 'CoreTimeClient'])['Cost'].sum().reset_index()
    epic_sum_data = epic_cost_data.groupby('Epic')['Cost'].sum().reset_index()
    epic_hours_fig = px.bar(
        epic_cost_data,
        x='Cost',
        y='Epic',
        color='CoreTimeClient',
        orientation='h',
        labels={'Cost': '', 'Epic': 'Epic', 'CoreTimeClient': 'CoreTimeClient'},
        title='Cost by Epics'
    )
    max_cost = epic_sum_data['Cost'].max()
    for i, row in epic_sum_data.iterrows():
        x = max_cost * 1.01
        y = row['Epic']
        text = f"{row['Cost']:.2f} €"
        epic_hours_fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(color='white'),
            xshift=10,
            align='left'
        )
    epic_hours_fig.update_layout(
        height=600,
        width=1400,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='white', width=1),
                fillcolor='rgba(0, 0, 0, 0)'
            )
        ]
    )
    treemap_data = df.groupby(['Sprint', 'Assignee', 'CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'])['Cost'].sum().reset_index()
    treemap_fig = px.treemap(
        treemap_data,
        path=['Sprint', 'Assignee','CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'],
        values='Cost',
        title='Cost by CoreTimeClient, Project, Phase, and Activity'
    )
    treemap_fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black', width=1),
                fillcolor='rgba(0, 0, 0, 0)'
            )
        ]
    )
    
    line_chart_story_points = px.line(
        df.groupby(['Work date', 'CoreTimeClient'])['Story Points'].sum().reset_index(),
        x='Work date',
        y='Story Points',
        color='CoreTimeClient',
        labels={'Work date': 'Week', 'Story Points': 'Story Points', 'CoreTimeClient': 'Client'},
        title='Story Points Delivered Weekly by Client'
    )
    line_chart_cost = px.line(
        df.groupby(['Work date', 'CoreTimeClient'])['Cost'].sum().reset_index(),
        x='Work date',
        y='Cost',
        color='CoreTimeClient',
        labels={'Work date': 'Week', 'Cost': 'Cost', 'CoreTimeClient': 'Client'},
        title='Weekly Cost by Client'
    )

    # Outer container
    outer_container = st.container()
    
    with outer_container:
        container2 = st.container()
        col2, col3 = container2.columns(2)

        col2.plotly_chart(epic_hours_fig, use_container_width=True)
        col2.plotly_chart(line_chart_cost, use_container_width=True)
        
        col3.plotly_chart(treemap_fig, use_container_width=True)
        col3.plotly_chart(line_chart_story_points, use_container_width=True)

    search_value = st.text_input("Search for value in table rows:", "")

    # Filter the DataFrame based on the search input
    filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)] if search_value else df
    # Display the filtered DataFrame as a table
    filtered_df = pd.DataFrame(filtered_df)


    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)

    # Create a container for the download button
    with st.container():
        # Convert DataFrame to CSV
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv',
            key='download-csv'
        )

    # Create a container for the download button
    with st.container():
        # Convert DataFrame to Json
        json_data = to_json(filtered_df)
        st.download_button(
            label="Download data as JSON",
            data=json_data,
            file_name='processed_data.json',
            mime='application/json'
        )


#-----------------------------------------------------------------------------------------------------------------------#

def calculate_average_ratio_by_project(df):
    avg_ratio_by_project = df.groupby('CoreTimeProject')['Avg_Ratio'].mean().reset_index()
    return avg_ratio_by_project

def create_cfd_chart(avg_ratio_by_project):
    # Sort the data by CoreTimeProject to ensure it appears in the desired order on the chart
    avg_ratio_by_project = avg_ratio_by_project.sort_values(by='CoreTimeProject')

    # Create the Cumulative Flow Diagram
    cfd_chart = go.Figure()
    cfd_chart.add_trace(go.Scatter(
        x=avg_ratio_by_project['CoreTimeProject'],
        y=avg_ratio_by_project['Avg_Ratio'],
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=8, color='blue'),
        name='Average Ratio by CoreTimeProject'
    ))

    # Update the chart layout and styling
    cfd_chart.update_layout(
        title='Daily Story Points Delivered by CoreTime Project',
        xaxis=dict(title='CoreTimeProject'),
        yaxis=dict(title='Average Ratio'),
        font=dict(color='black')
    )

    # Apply styling to the markers
    cfd_chart.data[0].marker.line.color = 'black'

    return cfd_chart


def generate_word_cloud_from_file(file_path):
        # Read the text data from the file
        with open(file_path, 'r') as file:
            text_data = file.read()

        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

        # Generate the word cloud image and return it
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide axes
        plt.tight_layout()

        # Save the word cloud to a BytesIO object
        word_cloud_data = BytesIO()
        plt.savefig(word_cloud_data, format='png')
        plt.close()

        # Rewind the BytesIO object
        word_cloud_data.seek(0)

        return word_cloud_data




def display_tab2(df, assignee_rates):
    # Assume 'sprint_bins' is accessible here, mapping sprints to their numeric order
    sprint_order = {f"Sprint {80 + i}": i for i, _ in enumerate(sprint_bins)}
    
    # Apply the sprint order to the DataFrame
    df['Sprint_Order'] = df['Sprint'].map(sprint_order)
    
    # Sort DataFrame by this new order for plotting
    df = df.sort_values(by='Sprint_Order')
    
    # Filter the DataFrame to only include rows where 'Status' is 'Done'
    done_df = df[df['Status'] == 'Done']
    
    # Group by 'Sprint' and aggregate based on the 'Story Points' and 'days' columns
    sprint_summary = done_df.groupby('Sprint').agg({'Story Points': 'sum', 'days': 'sum'}).reset_index()

    # Calculate the total story points and total days for each sprint
    sprint_totals = df.groupby('Sprint').agg({'Story Points': 'sum', 'days': 'sum'}).reset_index()

    # Calculate the average ratio as story points divided by days for each sprint
    sprint_totals['Avg_Ratio'] = sprint_totals['Story Points'] / sprint_totals['days']

    # Remove whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    # Filter the DataFrame to only include rows where 'Done' is True
    done_df = df[df['Status'] == 'Done']

    # Group by 'Sprint' and aggregate based on the 'Story Points' and 'days' columns
    sprint_summary = done_df.groupby('Sprint').agg({'Story Points': 'sum', 'days': 'sum'}).reset_index()



    
    # Using the sorted 'sprint_totals' DataFrame for plotting the Team Average Ratio by Sprint
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sprint_totals['Sprint'],
        y=sprint_totals['Avg_Ratio'],
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2),
    ))
    
    # Update layout for the line chart
    fig2.update_layout(
        title='Daily Story Points delivered by Sprints',
        xaxis=dict(title='Sprint'),
        yaxis=dict(title='Average Ratio'),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
    )



    avg_ratio_assignee_phase = df.groupby(['Assignee', 'CoreTimePhase'])['Avg_Ratio'].mean().reset_index()

    fig3 = go.Figure()

    # Add traces for each Assignee
    for assignee in avg_ratio_assignee_phase['Assignee'].unique():
        assignee_data = avg_ratio_assignee_phase[avg_ratio_assignee_phase['Assignee'] == assignee]
        fig3.add_trace(go.Scatterpolar(
            r=assignee_data['Avg_Ratio'],
            theta=assignee_data['CoreTimePhase'],
            mode='lines+markers',
            name=assignee,
            hovertemplate='%{theta}: %{r:.2f}<extra></extra>',
        ))

    # Update layout for the Radar Chart
    fig3.update_layout(
        title='Performance by Phases Assignee Peak Delivery Times',  # Add title for the radar chart
        polar=dict(
            radialaxis=dict(
                title='Average Ratio',
                tickmode='linear',
                tickvals=[0, 0.5, 1],
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=avg_ratio_assignee_phase['CoreTimePhase'].unique(),
            ),
        ),
        showlegend=True,
        legend=dict(
            title='Assignee',
            traceorder='normal',
            font=dict(size=10),
        ),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    # Calculate average ratio by CoreTimeProject
    avg_ratio_by_project = calculate_average_ratio_by_project(df)

    # Create the CFD chart
    cfd_chart = create_cfd_chart(avg_ratio_by_project)



    

    # First, filter the data for the current sprint
    current_sprint_number = get_last_sprint_number(df)  # Assuming you have a function to get the current sprint number
    # Replace NaN values with an empty string before checking for containment
    df['Sprint'] = df['Sprint'].astype(str)
    df_current_sprint = df[df['Sprint'].str.contains(str(current_sprint_number))]

        # Calculate assignee capacity
    assignee_capacity = assignee_median_capacity(df_current_sprint, df_current_sprint['Avg_Ratio'])
    df_current_sprint['Assignee Capacity'] = assignee_capacity

    # Calculate workload as sum of story points for the filtered data
    workload = df_current_sprint['Story Points'].sum()

    # Calculate Assignee capacity in the current sprint after deducting the workload
    assignee_capacity_current_sprint = assignee_capacity - workload

    # Create a DataFrame to hold the values for the horizontal bar chart
    data = pd.DataFrame({
        'Assignee': df_current_sprint['Assignee'],
        'Workload': workload,
        'Capacity': assignee_capacity_current_sprint
    })

    # Normalize the capacity and workload to percentages (of 100)
    total_capacity = assignee_capacity
    data['Assignee Workload'] = (data['Workload'] / total_capacity) * 100
    data['Assignee Capacity'] = (data['Capacity'] / total_capacity) * 100

    # Create the horizontal bar chart
    assignee_capacity_fig = px.bar(
        data,
        x=['Assignee Capacity', 'Assignee Workload'],
        y='Assignee',
        orientation='h',
        barmode='relative',  # To show both workload and capacity as percentages of total capacity
        title=f'Assignee Capacity and Workload in Sprint {current_sprint_number}',
        labels={'value': 'Percentage', 'variable': 'Metric'},
        color_discrete_map={'Assignee Capacity': 'green', 'Assignee Workload': 'red'}
    )

    # Set the range of the X-axis to [0, 100]
    assignee_capacity_fig.update_xaxes(range=[0, 100])



    # Create a box plot to visualize the distribution of story points by assignee
    box_plot = px.box(df, x='Assignee', y='Story Points', title='The Most Frequent Size of Issues by Assignee')

    # Customize the box plot appearance
    box_plot.update_traces(marker=dict(size=5))
    box_plot.update_xaxes(title='Assignee')
    box_plot.update_yaxes(title='Story Points')

    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig2, use_container_width=True)
        #st.subheader("Sprint Retrospective Word Cloud")
         #word_cloud_data = generate_word_cloud_from_file("retro.txt")
         #st.image(word_cloud_data)
        st.plotly_chart(box_plot, use_container_width=True)


    with col2:
        combined_chart = create_combined_chart(df, sprint_summary, sprint_totals)
        st.plotly_chart(combined_chart, use_container_width=True)
        st.plotly_chart(cfd_chart, use_container_width=True)

    # Add a search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)

# Call the display_tab2 function with your DataFrame and assignee_rates
# display_tab2(df, assignee_rates)


#-------------------------------------------------------------------------------------------------------------------------------------#



def display_tab3(df, assignee_rates):
    
    def custom_container(content, title=None):
        """Generates an HTML representation of a container mimicking mui.Paper."""
        style = """
            border: 1px solid #e0e0e0;
            padding: 16px;
            border-radius: 4px;
            resize: both;
            overflow: auto;
            margin: 8px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        """
        header = f"<h4>{title}</h4>" if title else ""
        return f"""
            <div style="{style}">
                {header}
                {content}
            </div>
        """

    # Calculate the top assignees by ticket count
    top_assignees_table = df['Assignee'].value_counts().head(10).reset_index()
    top_assignees_table.columns = ['Assignee', 'Ticket Count']

    # Calculate the top clients by ticket count
    top_clients_table = df['CoreTimeClient'].value_counts().head(10).reset_index()
    top_clients_table.columns = ['CoreTimeClient', 'Ticket Count']

    # Create a scatter plot: Spent days vs. Delivered Story Points by Project
    project_data = df.groupby('CoreTimeProject').agg({'Hours': 'sum', 'days': 'sum', 'Story Points': 'sum'}).reset_index()
    Spent_days_Delivered_SPs = px.scatter(
        project_data,
        x='days',
        y='Story Points',
        color='CoreTimeProject',
        labels={'days': 'Spent days', 'Story Points': 'Delivered Story Points'},
        hover_data=['Hours'],
        title='Most Productive Projects - Spent days vs. Delivered Story Points'
    )

    # Create a scatter plot: Story Points by Project, Assignee, and Sprint
    # Create a scatter plot: Story Points by Project, Assignee, and Sprint
    df_grouped = df.groupby(['Project name', 'Assignee', 'Sprint'])['Story Points'].sum().reset_index()
    Projects_Assignees_Sprints = px.scatter(
        df_grouped,
        x='Project name',
        y='Sprint',
        size='Story Points',
        color='Assignee',
        labels={'Project name': 'Project Name', 'Assignee': 'Assignee', 'Sprint': 'Sprint', 'Story Points': 'Story Points'},
        size_max=60,
        title='Contribution in Projects by Sprint'
    )

        # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.table(top_clients_table) 
    st.plotly_chart(Projects_Assignees_Sprints, use_container_width=True)


        
    with col2:
        st.table(top_assignees_table)
    st.plotly_chart(Spent_days_Delivered_SPs, use_container_width=True)


       

    # Add a search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)        



#-------------------------------------------------------------------------------------------------------------------------------------#
def get_last_sprint_number(df):
    # Ensure that all sprint values are strings and then extract the numbers
    sprint_numbers = [int(re.search(r'\d+', str(sprint)).group()) for sprint in df['Sprint'] if re.search(r'\d+', str(sprint))]

    if not sprint_numbers:  # Check if the list is empty
        st.warning("No valid sprint numbers found in the dataset.")
        return None

    return max(sprint_numbers)

#-------------------------------------------------------------------------------------------------------------------------------------#

def assignee_median_capacity(df, assignee):
    # Filter the dataframe for the specific assignee
    assignee_data = df[df['Assignee'] == assignee]

    # Group by sprint and sum the story points for each sprint
    total_story_points_per_sprint = assignee_data.groupby('Sprint')['Story Points'].sum()

    # Return the median of the summed story points across sprints
    return total_story_points_per_sprint.median()



def display_tab4(df, assignee_rates):

    # Create a radar chart for average ratio by assignee and core time phase
    avg_ratio_assignee_phase = df.groupby(['Assignee', 'CoreTimePhase'])['Avg_Ratio'].mean().reset_index()
    radar_chart_assignee_phase = px.line_polar(avg_ratio_assignee_phase, r='Avg_Ratio', theta='CoreTimePhase',
                                               line_close=True,
                                               title='Performance Phases - Assignee Peak Delivery Times',
                                               labels={'Avg_Ratio': 'Average Ratio', 'CoreTimePhase': 'Core Time Phase'},
                                               color='Assignee')



    # Create a new column 'Assignee Capacity' to store the median story points per sprint for each assignee
    df['Assignee Capacity'] = df['Assignee'].apply(lambda x: assignee_median_capacity(df, x))

    assignee_capacity_fig = px.box(
        df,
        x='Assignee',
        y='Assignee Capacity',
        title='Assignee Sprint Capacity: Story Point Potential per Sprint'
    )

    # Add data labels to the box plot for individual data points only
    assignee_capacity_fig.update_traces(
        boxpoints='all',  # Display all the points
        jitter=0.3,
        pointpos=-1.8,
        hovertemplate='Capacity: %{y}<br><extra></extra>'
    )

    # Set the maximum value of the Y-axis
    assignee_capacity_fig.update_yaxes(range=[0, df['Assignee Capacity'].max() + 10])


    # Create average ratio bar chart
    avg_ratio_data = df.groupby(['Issue Type', 'Assignee'])['Avg_Ratio'].mean().reset_index()
    avg_ratio_data.loc[avg_ratio_data['Issue Type'].isin(['Task', 'Sub-task']), 'Issue Type'] = 'Task & Sub-task'
    filtered_avg_ratio_data = avg_ratio_data[avg_ratio_data['Issue Type'].isin(['Task & Sub-task', 'Bug'])]
    color_map = {'Bug': 'darkred', 'Task & Sub-task': 'blue'}
    avg_ratio_chart = px.bar(
        filtered_avg_ratio_data,
        x='Assignee',
        y='Avg_Ratio',
        color='Issue Type',
        barmode='group',
        color_discrete_map=color_map,
        title='Issue-Type Delivery Insights: Daily Story Points per Assignee'
    )

    avg_ratio_chart.update_traces(texttemplate='%{value:.2f}', textposition='inside')

    # Line chart of Average Ratio by Sprint and Assignee
    line_chart_avg_ratio = px.line(
        df.groupby(['Sprint', 'Assignee'])['Avg_Ratio'].mean().reset_index(),
        x='Sprint',
        y='Avg_Ratio',
        color='Assignee',
        title='Trend Analysis: Assignee Story Point Delivery Over Sprints'
    )

    # Outer container
    outer_container = st.container()

    # Container 2: Charts
    with outer_container:
        container2 = st.container()
        col2, col3 = container2.columns(2)

        # Column 2
        col2.plotly_chart(avg_ratio_chart, use_container_width=True)
        col2.plotly_chart(radar_chart_assignee_phase, use_container_width=True)

        # Column 3
        col3.plotly_chart(assignee_capacity_fig, use_container_width=True)
        col3.plotly_chart(line_chart_avg_ratio, use_container_width=True)

    # Search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab4")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)

#----------------------------------------------------------------------------------------#
def calculate_assignee_capacity(df, avg_ratio, sprint_duration_weeks=3, planning_days=1, working_days_per_week=5):
    # Calculate the number of development days
    dev_days = (sprint_duration_weeks * working_days_per_week) - planning_days
    
    total_story_points_delivered = df['Story Points'].sum()
    
    # Calculate average daily story point rate using only development days
    #avg_daily_story_point_rate = total_story_points_delivered / dev_days

    # Capacity is calculated using the total number of working days, including planning
    total_working_days = (sprint_duration_weeks * dev_days)
    assignee_capacity = avg_ratio * dev_days

    return assignee_capacity


def display_tab5(df, assignee_rates, sprint_bins):
    # Remove whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    # First, filter the data for the current sprint
    current_sprint_number = get_last_sprint_number(df)  # Assuming you have a function to get the current sprint number
    
    # Handle NaN values when filtering by string contains
    mask = df['Sprint'].astype(str).str.contains(str(current_sprint_number)).fillna(False)
    df_current_sprint = df[mask]

    # Assuming the calculate_assignee_capacity function is defined elsewhere
    assignee_capacity = calculate_assignee_capacity(df_current_sprint, df_current_sprint['Avg_Ratio'])
    df_current_sprint['Assignee Capacity'] = assignee_capacity

    workload = df_current_sprint['Story Points'].sum()
    assignee_capacity_current_sprint = assignee_capacity - workload

    data = pd.DataFrame({
        'Assignee': df_current_sprint['Assignee'],
        'Workload': workload,
        'Capacity': assignee_capacity_current_sprint
    })

    total_capacity = assignee_capacity
    data['Assignee Workload'] = (data['Workload'] / total_capacity) * 100
    data['Assignee Capacity'] = (data['Capacity'] / total_capacity) * 100

    assignee_capacity_fig = px.bar(
        data,
        x=['Assignee Capacity', 'Assignee Workload'],
        y='Assignee',
        orientation='h',
        barmode='relative',
        title=f'Assignee Capacity and Workload in Sprint {current_sprint_number}',
        labels={'value': 'Percentage', 'variable': 'Metric'},
        color_discrete_map={'Assignee Capacity': 'green', 'Assignee Workload': 'red'}
    )
    assignee_capacity_fig.update_xaxes(range=[0, 100])
    assignee_capacity_fig.update_traces(texttemplate='%{value:.1f}', textposition='inside')

    # Convert datetime columns
    df_current_sprint['Created'] = pd.to_datetime(df_current_sprint['Created'], format='%d/%m/%Y %H:%M')
    df_current_sprint['Resolved'] = pd.to_datetime(df_current_sprint['Resolved'], format='%d/%m/%Y %H:%M')

    # Define the start and end dates for the current sprint
    start_date = pd.to_datetime("02/Aug/23 2:03 PM", format='%d/%b/%y %I:%M %p')
    end_date = pd.to_datetime("23/Aug/23 2:03 PM", format='%d/%b/%y %I:%M %p')

    # Filter tasks that were both created and resolved within the current sprint
    df_current_sprint_filtered = df_current_sprint[
        (df_current_sprint['Created'] >= start_date) & 
        (df_current_sprint['Resolved'] <= end_date)
    ]

    # Calculate the 'Resolution Time' column for the filtered tasks
    df_current_sprint = df_current_sprint.dropna(subset=['Resolved'])

    df_current_sprint_filtered.loc[:, 'Resolution Time'] = (df_current_sprint_filtered['Resolved'] - df_current_sprint_filtered['Created']).dt.days



    # Aggregate the resolution time by assignee (use the filtered data)
    resolution_by_assignee = df_current_sprint_filtered.groupby('Assignee')['Resolution Time'].sum().reset_index()

    resolution_time_fig = px.bar(
        resolution_by_assignee,
        x='Assignee',
        y='Resolution Time',
        title=f'Under construction - Resolution Time per Assignee in Sprint {current_sprint_number}',
        labels={'Resolution Time': 'Total Resolution Time (days)'},
        orientation='v'
    )
    resolution_time_fig.update_traces(texttemplate='%{value}', textposition='outside')

        # Define the start date for the current sprint
    start_date = pd.to_datetime("02/Aug/23 2:03 PM", format='%d/%b/%y %I:%M %p')

    total_story_points_current_sprint = df_current_sprint['Story Points'].sum()

    df_after_date = df_current_sprint[df_current_sprint['Created'] > start_date]
    total_story_points_after_date = df_after_date['Story Points'].sum()

    data_added_to_sprint = pd.DataFrame({
        'Category': ['Total in Current Sprint', f'Added After {start_date.strftime("%d/%b/%Y %I:%M %p")}'],
        'Story Points': [total_story_points_current_sprint, total_story_points_after_date]
    })

    added_to_sprint_fig = px.bar(
        data_added_to_sprint,
        x='Category',
        y='Story Points',
        title=f'Story Points Added to Sprint {current_sprint_number}',
        labels={'Category': 'Category', 'Story Points': 'Story Points'},
        orientation='v'
    )
    added_to_sprint_fig.update_traces(texttemplate='%{value}', textposition='outside')



    grouped_df = df_current_sprint.groupby(['Assignee', 'Status'])['Story Points'].sum().reset_index()
    
    status_assignee_fig = px.bar(
        grouped_df,
        x='Assignee',
        y='Story Points',
        color='Status',
        title=f'Story Points by Status and Assignee in Sprint {current_sprint_number}',
        labels={'Story Points': 'Story Points'},
        orientation='v'
    )
    status_assignee_fig.update_traces(texttemplate='%{value}', textposition='outside')

    # Filter the DataFrame for the current sprint and then group by CoreTimeClient
    df_current_sprint_hours = df[df['Sprint'].str.contains(str(current_sprint_number))]
    hours_by_client = df_current_sprint_hours.groupby('CoreTimeClient')['Hours'].sum().reset_index()


    # Create the 3D pie chart
    fig = go.Figure(data=[go.Pie(
        labels=hours_by_client['CoreTimeClient'],
        values=hours_by_client['Hours'],
        hovertemplate="%{label}: %{value} hours",
        textinfo='label+percent',
        pull=[0.1] * len(hours_by_client)  # Set the distance of each sector from the center
    )])

    fig.update_layout(
        title="Worked Hours in the Current SprintS by CoreTimeClient",
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.7),
            camera_eye=dict(x=1.2, y=1.2, z=0.6),
            dragmode="turntable",
        ),
        showlegend=False  # Hide the legend for a cleaner look
    )


    
    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(assignee_capacity_fig, use_container_width=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.plotly_chart(added_to_sprint_fig, use_container_width=True)
        st.plotly_chart(status_assignee_fig, use_container_width=True)

    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)


def Similarity_Analysis(df):
    st.header("Similarity Analysis")

    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        st.session_state['data_frame'] = df  # This line is similar to the one in the "Connect to Jira" section
        st.write(df)
        st.success("Data successfully uploaded!")

        # UI elements for column selection
        text_column = st.selectbox("Select Text Column for Analysis", df.columns, key='text_column_selector')
        identifier_column = st.selectbox("Select Identifier Column", df.columns, key='identifier_column_selector')
        additional_columns = st.multiselect("Select Additional Columns to Display", df.columns, key='additional_columns_selector')
        
        # Save selected columns
        if st.button('Save Selected Columns for Analysis', key='save_selected_columns_button'):           
            st.session_state.text_column = text_column
            st.session_state.identifier_column = identifier_column
            st.session_state.additional_columns = additional_columns
        # Check if necessary columns are in session_state before calling similarity function
        
        
        if all(key in st.session_state for key in ['text_column', 'identifier_column', 'additional_columns']):
            similarity_func(df)

    # Place the information box within the sidebar using an expander
    with st.sidebar.expander("How it works", expanded=False):
        st.markdown("""
        The script is designed to process and analyze text data, identifying and quantifying similarities between different pieces of text.
    
        **Text Preprocessing**:
        - The script begins by accessing a list of common words (stopwords) like "the", "and", and "in", which are usually removed as they don’t contribute significantly to meaning.
        - Each text entry is cleaned by removing stopwords, email addresses, URLs, and HTML tags, resulting in a concise and meaningful version of the original text. This cleaned text is then stored for further analysis.
    
        **Similarity Calculation**:
        - *TF-IDF Vectorization*: The cleaned text is converted into numerical format using TF-IDF Vectorization, a technique that helps the computer understand the importance of each word relative to the dataset.
        - *Calculating & Filtering Similarities*: The script calculates the similarity between each piece of text and filters out pairs that have a similarity score above a set threshold. The result is a list of text pairs deemed sufficiently similar, accompanied by their similarity scores and relevant information from the original data.
        """, unsafe_allow_html=True)

    


#---------------------------------------------------------------------------------------------------#
def display_LLM(df=None, assignee_rates=None):
    
    # Streamlit App Layout
    st.title('Connect Data To LLM')
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = {
            'jira_df': pd.DataFrame(),
            'prompt_templates': [],
            'openai_api_key': '',
            'selected_columns': []
        }
    
    # Function to initialize default prompt templates
    # Function to initialize default prompt templates
    def initialize_default_prompt_templates():
        templates = [
            {
                "name": "Issue Tracker",
                "instructions": "As a Project Coordinator, provide an integrated report on the status, recent developments, and any blockers or dependencies related to a specific issue.",
                "example_input": "Provide a detailed report on issue KEY-123.",
                "example_output": "Issue KEY-123, titled [Summary], is currently [Status]. The latest update was on [Updated], with [Comment count] comments, and the last comment: [Latest Comment]. There are [Blocker count] blockers impeding progress, with the main dependency on [Dependency Key].",
                "query_template": "Provide a detailed report on issue {issue_key}.",
                "few_shot_count": 3
            },
    
            {
                "name": "Analyze",
                "instructions": "Analyze selected columns from Jira text data to identify patterns, similarities, and inefficiencies. Provide insights into repetitive tasks and potential areas for process optimization.",
                "example_input": "Analyze the impact and frequency of issues related to 'Network Connectivity' over the last quarter.",
                "example_output": "The analysis reveals that 'Network Connectivity' issues have increased by 25% over the last quarter, with most incidents reported during peak usage hours, suggesting a need for infrastructure scaling.",
                "query_template": "Analyze {issue_topic} issues over the last {time_period}.",
                "few_shot_count": 3
            },
            # New Template: Resource Allocation
            {
                "name": "Resource Allocation",
                "instructions": "As a Resource Manager, outline the workload distribution across the team, flagging any over-allocations.",
                "example_input": "How is the workload spread for the current sprint?",
                "example_output": "In the current sprint, [Assignee 1] is at 80% capacity, [Assignee 2] is over-allocated by 20%, and [Assignee 3] can take on more work. Adjustments are recommended.",
                "query_template": "How is the workload spread for the current sprint?",
                "few_shot_count": 3
            },
            # New Template: Risk Assessment
            {
                "name": "Risk Assessment",
                "instructions": "As a Risk Assessor, identify high-risk issues based on priority, due dates, and current status.",
                "example_input": "What are the high-risk issues for the upcoming release?",
                "example_output": "High-risk issues for the release include [Issue 1] due in [Days] days at [Priority] priority, and [Issue 2] which is [Status] and past the due date.",
                "query_template": "What are the high-risk issues for the upcoming release?",
                "few_shot_count": 3
            },
        ]
        st.session_state.data['prompt_templates'] = templates if not st.session_state.data['prompt_templates'] else st.session_state.data['prompt_templates']
    
    initialize_default_prompt_templates()
    
    
    # JIRA Utility Functions
    @st.cache(suppress_st_warning=False, allow_output_mutation=True)
    def fetch_project_names(username, api_token, jira_domain):
        base_url = f'{jira_domain}/rest/api/latest/project'  # Use jira_domain in the URL
        headers = {'Accept': 'application/json'}
        try:
            response = requests.get(base_url, auth=(username, api_token), headers=headers)
            response.raise_for_status()  # This will raise an exception for HTTP errors
            data = response.json()
            return [project['name'] for project in data]
        except requests.exceptions.RequestException as e:
            st.error(f'Failed to fetch project names: {e}')
            return []
    
    @st.cache(suppress_st_warning=False, allow_output_mutation=True)
    def fetch_jira_issues(username, api_token, project_name, jql_query, jira_domain):
        base_url = f'{jira_domain}/rest/api/latest/search'  # Use jira_domain in the URL
        params = {
            'jql': jql_query,
            'maxResults': 1000,
            'fields': '*all'
        }
        try:
            response = requests.get(base_url, auth=(username, api_token), headers={'Accept': 'application/json'}, params=params)
            response.raise_for_status()
            data = response.json()
            return pd.json_normalize(data['issues'])
        except requests.exceptions.RequestException as e:
            st.error(f'Failed to fetch issues: {e}')
            return pd.DataFrame()
    
    # Add/Edit Prompt Template Functions
    def add_template(name, instructions, example_input, example_output, query_template, few_shot_count):
        template = {
            "name": name,
            "instructions": instructions,
            "example_input": example_input,
            "example_output": example_output,
            "query_template": query_template,
            "few_shot_count": few_shot_count
        }
        st.session_state.data['prompt_templates'].append(template)
    
    def update_template(index, name, instructions, example_input, example_output, query_template, few_shot_count):
        st.session_state.data['prompt_templates'][index] = {
            "name": name,
            "instructions": instructions,
            "example_input": example_input,
            "example_output": example_output,
            "query_template": query_template,
            "few_shot_count": few_shot_count
        }
    
    # Construct Full Prompt with Dynamic Few-Shot Examples
    def construct_full_prompt(template, actual_input):
        # Incorporate few-shot examples dynamically based on the template's 'few_shot_count'
        few_shot_examples = "\n".join([f"Example Input: {template['example_input']}\nExample Output: {template['example_output']}" for _ in range(template['few_shot_count'])])
        return f"{template['instructions']}\n{few_shot_examples}\n{template['query_template']}\n\n{actual_input}"
    
    def execute_prompt(template, test_input, data, selected_columns):
        try:
            # Filter the data to include only the selected columns
            if selected_columns:
                data_df = pd.read_json(data)
                filtered_data = data_df[selected_columns].to_json()
            else:
                filtered_data = data
    
            full_prompt = f"{template}\n\n{test_input}\n\n{filtered_data}"
    
            # Split the full_prompt into segments of appropriate length
            segments = [full_prompt[i:i+4096] for i in range(0, len(full_prompt), 4096)]
    
            responses = []
    
            for segment in segments:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": segment}]
                )
                responses.append(response.choices[0].message['content'])
    
            return " ".join(responses)
        except Exception as e:
            return f"Error: {e}"
        
    
    
    
    
    # Sidebar for User Input and Mode Selection
    with st.sidebar:
        st.title("Settings")
        app_mode = st.selectbox('Choose the Mode', ['Home', 'Manage Data Sources', 'Manage Prompts'])
    
    # Home Page
    if app_mode == 'Home':
        st.markdown("""
        Streamlit Application for Data Analysis with LLM Integration
    
        This Streamlit application is designed to seamlessly integrate data from Jira, user-uploaded CSV files with the capabilities of Large Language Models (LLMs). 
        This facilitates data analysis and interaction through tailored prompts, making it easier to derive insights from various data sources.
    
        How It Works
        ------------
    
        - Data Source Integration:
            - Jira: Fetch projects and issues from your Jira account. Specify the project and customize the JQL query to retrieve relevant data.
            - Upload .csv: Easily upload your own datasets through CSV files for quick and direct analysis, enhancing flexibility in data integration.
    
        - Prompt Management:
            - Add/Edit Prompt Templates: Define templates with specific instructions, example inputs/outputs, query templates, and the number of few-shot examples. These guide the LLM in generating responses based on your data.
            - Execute Prompts: Utilize your templates to query the integrated data, receiving responses crafted by the LLM for dynamic interaction and insights.
    
    
        Tips for Effective Prompt Engineering
        -------------------------------------
    
        - Be Specific: Clearly define the role and knowledge scope you expect the LLM to assume in your prompts for more accurate and relevant responses.
    
        - Utilize Few-Shot Learning: Incorporate examples in your prompts to guide the LLM. Few-shot examples can significantly improve the model's understanding and output quality.
    
        - Dynamic Queries: Leverage the power of dynamic variables in your templates to make your prompts adaptable to different queries and data points.
    
        Explore the capabilities of connecting data to LLM for an enhanced data analysis experience.
    
        """)
    
    
    
    # Data Source Management
    if app_mode == 'Manage Data Sources':
        with st.sidebar:
            data_source = st.radio("Select Data Source", ['Upload CSV', 'Jira'])
        if data_source == 'Upload CSV':
            uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
            if uploaded_file is not None:
                try:
                    st.session_state.data['jira_df'] = pd.read_csv(uploaded_file)
                    st.success('CSV file uploaded successfully!')
                    st.write(st.session_state.data['jira_df'])
                except Exception as e:
                    st.error(f'Failed to read CSV file: {e}')
            else:
                st.info("Upload a CSV file to proceed.")
    
        if data_source == 'Jira':
            with st.sidebar:
                jira_domain = st.text_input('Jira Domain', 'https://your-jira-domain.atlassian.net')
                username = st.text_input('Jira username')
                api_token = st.text_input('Jira API token', type='password')
            if username and api_token:
                project_names = fetch_project_names(username, api_token, jira_domain)
                if project_names:
                    with st.sidebar:
                        selected_project_name = st.selectbox('Select a Jira project', project_names)
                    jql_query = st.text_area("Enter JQL Query:")
                    if st.button('Fetch Jira Issues'):
                        with st.spinner('Fetching Jira Issues...'):
                            st.session_state.data['jira_df'] = fetch_jira_issues(username, api_token, selected_project_name, jql_query, jira_domain)
                            if not st.session_state.data['jira_df'].empty:
                                st.success('Jira issues fetched successfully!')
                                # Display Raw Data (Option: As JSON)
                                st.json(st.session_state.data['jira_df'].to_json(orient="records"))
                            else:
                                st.error('No data available for the selected project.')
                else:
                    st.error("Check your credentials.")
    
            if not st.session_state.data['jira_df'].empty:
                st.write("Fetched Jira Data")
                st.dataframe(st.session_state.data['jira_df'])
    
        if not st.session_state.data['jira_df'].empty:
            st.title("Select Columns for Analysis")
            selected_columns = st.multiselect("Select columns to save for analysis:", st.session_state.data['jira_df'].columns)
            st.session_state.data['selected_columns'] = selected_columns
    
    
    
    if app_mode == 'Manage Prompts':
        with st.sidebar:
            openai_api_key = st.text_input('OpenAI API key', type='password')
            if openai_api_key:
                st.session_state.data['openai_api_key'] = openai_api_key
                openai.api_key = openai_api_key
    
        # Display only the selected columns from the saved DataFrame
        if 'selected_columns' in st.session_state.data and st.session_state.data['selected_columns']:
            st.write("Selected Columns for Analysis:")
            st.write(st.session_state.data['selected_columns'])
            
            if not st.session_state.data['jira_df'].empty or not st.session_state.data['neo4j_df'].empty:
                st.write("Saved DataFrame (Selected Columns):")
                # Determine which DataFrame to use based on what's available
                data_source_df = st.session_state.data['jira_df'] if not st.session_state.data['jira_df'].empty else st.session_state.data['neo4j_df']
                # Filter the DataFrame to only include the selected columns
                filtered_df = data_source_df[st.session_state.data['selected_columns']]
                st.dataframe(filtered_df)
            else:
                st.write("No DataFrame saved.")
        else:
            st.write("No columns selected or DataFrame saved.")
    
        # Edit existing prompt template
        existing_prompt_names = [tpl['name'] for tpl in st.session_state.data['prompt_templates']]
        selected_template_idx = st.selectbox(
            "Prompt templates:", 
            range(len(existing_prompt_names)), 
            format_func=lambda x: existing_prompt_names[x]
        )
        selected_template = st.session_state.data['prompt_templates'][selected_template_idx]
    
    
        # Execute prompt with user input
        user_input = st.text_input("Enter your query or question:")
    
        if user_input:  # Check if user has entered any input
            if st.button("Execute Prompt"):
                selected_template = st.session_state.data['prompt_templates'][selected_template_idx]  # Moved inside the button condition
                if selected_template:
                    if st.session_state.data['jira_df'].empty:
                        st.warning("No Jira data available. Please fetch Jira issues first.")
                    else:
                        with st.spinner('Executing Prompt...'):  # Move the spinner inside the button condition
                            # Construct the full prompt
                            full_prompt = construct_full_prompt(selected_template, user_input)
    
                            # Execute the prompt with the Jira data
                            response = execute_prompt(selected_template, user_input, st.session_state.data['jira_df'].to_json(), st.session_state.data['selected_columns'])
    
                            # Display the response
                            st.write("Response:")
                            st.write(response)
        else:
            st.info("Please enter your query or question before executing the prompt.")
    
    
        # Edit functionality under expander
        with st.expander("Edit Prompt Template"):
            selected_template = st.session_state.data['prompt_templates'][selected_template_idx]
            edited_name = st.text_input("Prompt Name:", value=selected_template['name'], key=f"name_{selected_template_idx}", help="Provide a concise yet descriptive name. This will help users identify the prompt's purpose at a glance.")
            edited_instructions = st.text_area("Instructions:", value=selected_template['instructions'], key=f"instructions_{selected_template_idx}", help="""Detail the intended interaction model or role the AI should assume. For example, 'You are a helpful assistant that provides concise answers.' Be specific about the tone, style, and any constraints the AI should adhere to.""")
            edited_example_input = st.text_area("Example Input:", value=selected_template['example_input'], key=f"input_{selected_template_idx}", help="Include a representative input that the prompt is expected to handle. This should illustrate the kind of questions or commands the AI will respond to.")
            edited_example_output = st.text_area("Example Output:", value=selected_template['example_output'], key=f"output_{selected_template_idx}", help="Provide an example response that aligns with the instructions and input. Ensure it demonstrates the desired output format and content.")
            edited_query_template = st.text_area("Query Template:", value=selected_template['query_template'], key=f"template_{selected_template_idx}", help="""Craft the structure of the query that will be generated. Use placeholders for dynamic parts. For instance, '{user_query}' could be replaced with actual user input during execution.""")
            edited_few_shot_count = st.slider("Number of Few-Shot Examples", min_value=1, max_value=10, value=selected_template['few_shot_count'], key=f"few_shot_{selected_template_idx}", help="Adjust the number of few-shot examples. Few-shot learning helps the model understand the task by providing examples.")
    
            if st.button("Save Changes", key=f"save_{selected_template_idx}"):
                update_template(selected_template_idx, edited_name, edited_instructions, edited_example_input, edited_example_output, edited_query_template, edited_few_shot_count)
                st.success("Prompt template updated successfully!")
    
#-----------------------------------------------------#



# Define a function for displaying Tab 6
def display_tab6():
    st.header("Tab 6: Chat with Documents")

    # User Inputs and chatbot functionality
    uploaded_files = st.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
    user_query = st.text_input("Ask me anything!")

    if uploaded_files and user_query:
        obj = CustomDataChatbot()
        qa_chain = obj.setup_qa_chain(uploaded_files)

        # Display user input
        utils.display_msg(user_query, 'user')

        # Send user query to the assistant
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display PDF upload and chat input
    with st.expander("PDF Upload"):
        st.write("Upload PDF documents here.")
        if not uploaded_files:
            st.warning("Please upload PDF documents.")
    
    with st.expander("Chat with Documents"):
        st.write("Ask questions related to the uploaded documents.")
        if not user_query:
            st.warning("Ask a question to get started.")

#---------------------------------------------------------------------------------------------------#


def display_predictions():
    st.title('Task Time Prediction Results')

    # Check if predictions are available in the session state
    if 'predictions' in st.session_state:
        predictions = st.session_state.predictions
        true_values = st.session_state.true_values  # Assuming true values are also stored in session state for comparison

        # Display predictions alongside true values
        results_df = pd.DataFrame({
            "Actual Duration": true_values,
            "Predicted Duration": predictions.flatten()  # Adjusting shape if necessary
        })

        # Display the DataFrame in Streamlit
        st.dataframe(results_df)

        # Plotting actual vs predicted values using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual Duration'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted Duration'], mode='lines+markers', name='Predicted'))
        fig.update_layout(title='Actual vs. Predicted Task Durations', xaxis_title='Task', yaxis_title='Duration (hours)', legend_title="Legend")
        st.plotly_chart(fig)

        # Optionally, provide download links for the results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download results as CSV", data=csv, file_name='task_time_predictions.csv', mime='text/csv')

        # Display any additional metrics or charts as needed
        # For example, showing a histogram of residuals (errors)
        residuals = results_df['Actual Duration'] - results_df['Predicted Duration']
        fig_residuals = px.histogram(residuals, nbins=50, title="Histogram of Prediction Errors")
        fig_residuals.update_layout(xaxis_title="Error", yaxis_title="Count")
        st.plotly_chart(fig_residuals)

    else:
        st.warning('No predictions to display. Please run the model to generate predictions.')




            
#---------------------------------------------------------------------------------------------------#
DEFAULT_RATES = {
    "Pawel G": 00.00,
    "Marcin Ko": 00.00,
    "Marcin Kl": 00.00,
    "Lukasz": 00.00,
    "Alek D": 00.00,
    "Dawid N": 00.00
}


def run_app():
    st.set_page_config(layout='wide')
    df = None  # Initialize df within run_app function for proper scope

    # Add instructions for users on how to use the application
    st.title("Welcome to the Performance Analysis Application")
    st.markdown("""
    This application helps you analyze team and individual performance, costs, productivity, and workload.
    To get started, please follow these steps:
    
    1. **Download Files**: Click the button below to download the necessary files.
    2. **Upload Files**: Use the sidebar to upload the downloaded CSV files.
    3. **Navigate**: Select the desired tab from the sidebar to view specific analysis.

    Click the button below to go to the fileshare page to download the required CSV files:
    """)
    fileshare_url = ""
    st.markdown(f'<a href="{fileshare_url}" target="_blank">Download Files</a>', unsafe_allow_html=True)

    # Define tabs dictionary
    tabs = {
        "Team Performance": display_tab2,
        "Individual Performance": display_tab4,
        "Costs": display_tab1,
        "Productivity & Workload": display_tab3,
        "LLM": display_LLM,  
        "Similarity": Similarity_Analysis,
        "Prediction": display_predictions,
    }
    # Get user selection
    selected_tab = st.sidebar.radio("Select a Tab", list(tabs.keys()))
    if 'selected_tab' not in st.session_state or st.session_state.selected_tab != selected_tab:
        st.session_state.selected_tab = selected_tab

    # Initialize variables
    df = None
    assignee_rates = None

    # Conditional loading for tabs that require CSV file uploads
    if selected_tab not in ["LLM", "Similarity", "Assistant"]:  # Add other tabs that don't require CSV uploads as necessary
        with st.sidebar.expander("Upload Files"):
            uploaded_file_iterative = st.file_uploader("Choose Iterative CSV file", type="csv")
            uploaded_file_eigen = st.file_uploader("Choose Eigen CSV file", type="csv")

        if uploaded_file_iterative and uploaded_file_eigen:
            df = load_data(uploaded_file_iterative, uploaded_file_eigen, sprint_bins)
            if df is not None:
                assignee_rates = get_assignee_rates(df)

    # Handle each tab specifically
    if selected_tab in tabs:
        if selected_tab == "LLM":
            display_LLM()  # LLM tab function called without parameters
        elif selected_tab == "Similarity":
            tabs[selected_tab](None)  # Call with None if needed, adjust as necessary
        elif selected_tab == "Assistant":
            # Assistant tab functionality here...
            pass
        elif df is not None and assignee_rates is not None:
            # For tabs requiring CSV uploads and processing
            tabs[selected_tab](df, assignee_rates)
        else:
            st.warning("Please upload files for the selected tab.")



    if df is not None:
        # Assuming df has been properly loaded and processed at this point
        # Now that df is confirmed to exist, it's safe to convert to JSON and provide download link
        json_data = to_json(df)
        st.sidebar.download_button(
            label="Download processed data as JSON",
            data=json_data,
            file_name='processed_data.json',
            mime='application/json'
        )

# Ensure the run_app function is called to execute the app
if __name__ == "__main__":
    run_app()
