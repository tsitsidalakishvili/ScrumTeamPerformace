import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import re
import datetime



def load_data(uploaded_file_iterative, uploaded_file_eigen):
    if uploaded_file_iterative is not None and uploaded_file_eigen is not None:
        try: 
            Iterative = pd.read_csv(uploaded_file_iterative, encoding='iso-8859-1')
            Eigen = pd.read_csv(uploaded_file_eigen, encoding='iso-8859-1')

            Iterative = Iterative[['Issue summary', 'Hours', 'Full name', 'Work date', 'CoreTime', 'Issue Type', 'Issue Status']]
            Eigen = Eigen[['Summary', 'Issue key', 'Status', 'Resolved', 'Epic Link Summary', 'Labels', 'Priority', 'Issue Type', 'Assignee', 'Creator', 'Sprint', 'Created', 'Resolution', 'Custom field (Story Points)',
                           'Custom field (CoreTimeActivity)', 'Custom field (CoreTimeClient)', 'Custom field (CoreTimePhase)', 'Custom field (CoreTimeProject)', 'Project name']]

            Eigen = Eigen.rename(columns={'Custom field (CoreTimeActivity)': 'CoreTimeActivity', 'Custom field (CoreTimeClient)': 'CoreTimeClient',
                                          'Custom field (CoreTimePhase)': 'CoreTimePhase', 'Custom field (CoreTimeProject)': 'CoreTimeProject', 'Epic Link Summary': 'Epic', 'Custom field (Story Points)': 'Story Points'})
    
            def extract_key_ID(df):
                df['Issue key'] = df['Issue summary'].str.extract('\[(.*?)\]', expand=False)
                df = df.drop('Issue summary', axis=1)
                return df

            Iterative = extract_key_ID(Iterative)
            Iterative = Iterative.rename(columns={'Full name': 'Assignee'})

            merged_df = pd.merge(Iterative, Eigen, on='Issue key')
            merged_df = merged_df.rename(columns={'Assignee_x': 'Assignee', 'Issue Type_x': 'Issue Type'})
            merged_df = merged_df.drop('Issue Type_y', axis=1)
            merged_df = merged_df[['Issue key', 'Work date', 'Assignee', 'Sprint', 'Status', 'Priority', 'Project name', 'Issue Type', 'Creator', 'Created', 'Resolution',
                                   'Story Points', 'Hours', 'Resolved','Epic', 'CoreTimeActivity', 'CoreTimeClient', 'CoreTimePhase', 'CoreTimeProject', 'Labels']]
            grouped_df = merged_df.groupby('Issue key').agg({'Hours': 'sum', 'Story Points': 'first', 'Assignee': 'first', 'Sprint': 'first', 'Resolution': 'first', 'Resolved': 'first', 'Created': 'first', 'Priority': 'first',
                                                             'Creator': 'first', 'CoreTimeActivity': 'first', 'CoreTimeClient': 'first', 'CoreTimePhase': 'first', 'Epic': 'first',
                                                             'CoreTimeProject': 'first', 'Project name': 'first', 'Status': 'first', 'Labels': 'first', 'Work date': 'first', 'Issue Type' : 'first'}).reset_index()
            df = grouped_df
            df['days'] = df['Hours'] / 8
            df['Avg_Ratio'] = df['Story Points'] / df['days']
            df['Work date'] = pd.to_datetime(df['Work date'])

            df.to_csv('df.csv', index=False)
            return df

        except Exception as e:
            st.warning(f"Error processing the files: {e}")
            return None

    else:
        st.warning("Please upload both CSV files to proceed!")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------#


def get_assignee_rates(df):
    assignee_rates = {}
    for assignee in df['Assignee'].unique():
        key = f'rate_{assignee}'  # Unique key based on assignee name
        rate = st.sidebar.number_input(label=f"Rate for {assignee}", value=0, step=1, key=key)
        assignee_rates[assignee] = rate
    return assignee_rates

def display_assignee_rates(assignee_rates):
    for assignee, rate in assignee_rates.items():
        st.sidebar.write(f"Rate for Assignee {assignee}: {rate}")




#--------------------------------------------------------------------------------------------------------------#


def create_combined_chart(df, sprint_summary, sprint_avg_ratio):
    # Normalize the data
    normalized_story_points = (sprint_summary['Story Points'] - sprint_summary['Story Points'].min()) / (
        sprint_summary['Story Points'].max() - sprint_summary['Story Points'].min())
    normalized_worked_days = (sprint_summary['days'] - sprint_summary['days'].min()) / (
        sprint_summary['days'].max() - sprint_summary['days'].min())
    normalized_avg_ratio = (sprint_avg_ratio['Avg_Ratio'] - sprint_avg_ratio['Avg_Ratio'].min()) / (
        sprint_avg_ratio['Avg_Ratio'].max() - sprint_avg_ratio['Avg_Ratio'].min())

    # Create the combined chart
    combined_chart = go.Figure()
    combined_chart.add_trace(go.Bar(
        x=sprint_summary['Sprint'],
        y=normalized_story_points,
        name='Story Points',
        text=sprint_summary['Story Points'],
        textposition='auto'
    ))
    combined_chart.add_trace(go.Bar(
        x=sprint_summary['Sprint'],
        y=normalized_worked_days,
        name='days',
        text=sprint_summary['days'],
        textposition='auto'
    ))
    combined_chart.add_trace(go.Scatter(
        x=sprint_avg_ratio['Sprint'],
        y=normalized_avg_ratio,
        name='Average Ratio',
        mode='lines+markers',
        line=dict(color='red'),
        marker=dict(symbol='circle', size=8, color='red')
    ))

    # Update the chart layout and styling
    combined_chart.update_layout(
        barmode='group',
        title='Team Average Ratio and Delivered Story Points vs. Worked days by Sprint',
        xaxis=dict(title='Sprint'),
        yaxis=dict(title='Normalized Value'),
        legend=dict(title='Metrics'),
        font=dict(color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Apply styling to the bars and markers
    for trace in combined_chart.data:
        if 'marker' in trace and 'line' in trace['marker']:
            trace.marker.line.color = 'black'
        if 'textfont' in trace:
            trace.textfont.color = 'black'
            trace.textfont.size = 14

    return combined_chart


#--------------------------------------------------------------------------------------------------------------#

    
def display_tab1(df, assignee_rates):
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
            font=dict(color='black'),
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
                line=dict(color='black', width=1),
                fillcolor='rgba(0, 0, 0, 0)'
            )
        ]
    )
    treemap_data = df.groupby(['CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'])['Cost'].sum().reset_index()
    treemap_fig = px.treemap(
        treemap_data,
        path=['CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'],
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
        title='Team Average Ratio by CoreTimeProject',
        xaxis=dict(title='CoreTimeProject'),
        yaxis=dict(title='Average Ratio'),
        font=dict(color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white'
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
    # Remove whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    sprint_summary = df.groupby('Sprint').agg({'Story Points': 'sum', 'days': 'sum'}).reset_index()


    #  Team Average Ratio by Sprint
    sprint_avg_ratio = df.groupby('Sprint')['Avg_Ratio'].mean().reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sprint_avg_ratio['Sprint'],
        y=sprint_avg_ratio['Avg_Ratio'],
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2),
    ))

    # Update layout for the line chart
    fig2.update_layout(
        title='Team Average Ratio by Sprint',  # Add title for the line chart
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
        title='Average Ratio by Assignee and Core Time Phase',  # Add title for the radar chart
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
    df_current_sprint = df[df['Sprint'].str.contains(str(current_sprint_number))]

        # Calculate assignee capacity
    assignee_capacity = calculate_assignee_capacity(df_current_sprint, df_current_sprint['Avg_Ratio'])
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

    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Sprint Retrospective Word Cloud")
        word_cloud_data = generate_word_cloud_from_file("retro.txt")
        st.image(word_cloud_data)


    with col2:
        combined_chart = create_combined_chart(df, sprint_summary, sprint_avg_ratio)
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
        title='Spent days vs. Delivered Story Points by Project'
    )

    # Create a scatter plot: Story Points by Project, Assignee, and Sprint
    df_grouped = df.groupby(['Project name', 'Assignee', 'Sprint'])['Story Points'].sum().reset_index()
    Projects_Assignees_Sprints = px.scatter(
        df_grouped,
        x='Project name',
        y='Assignee',
        size='Story Points',
        color='Sprint',
        labels={'Project name': 'Project Name', 'Assignee': 'Assignee', 'Sprint': 'Sprint', 'Story Points': 'Story Points'},
        size_max=60,
        title='Story Points by Project, Assignee, and Sprint'
    )

        # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.table(top_assignees_table)
        st.plotly_chart(Projects_Assignees_Sprints, use_container_width=True)


    with col2:
        st.table(top_clients_table)
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


#--------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------------------------------#
def calculate_assignee_capacity(df, avg_ratio, sprint_duration_weeks=3, working_days_per_week=5):
    total_story_points_delivered = df['Story Points'].sum()
    total_working_days = sprint_duration_weeks * working_days_per_week
    avg_daily_story_point_rate = total_story_points_delivered / total_working_days
    assignee_capacity = avg_ratio * total_working_days
    return assignee_capacity

#-------------------------------------------------------------------------------------------------------------------------------------#
def get_last_sprint_number(df):
    # Find the last sprint number using regular expression
    sprint_numbers = [int(re.search(r'\d+', sprint).group()) for sprint in df['Sprint'] if re.search(r'\d+', sprint)]
    last_sprint_number = max(sprint_numbers)
    return last_sprint_number

#-------------------------------------------------------------------------------------------------------------------------------------#


def display_tab4(df, assignee_rates):
    # Create a radar chart for average ratio by assignee and core time phase
    avg_ratio_assignee_phase = df.groupby(['Assignee', 'CoreTimePhase'])['Avg_Ratio'].mean().reset_index()
    radar_chart_assignee_phase = px.line_polar(avg_ratio_assignee_phase, r='Avg_Ratio', theta='CoreTimePhase',
                                               line_close=True,
                                               title='Average Ratio by Assignee and Core Time Phase',
                                               labels={'Avg_Ratio': 'Average Ratio', 'CoreTimePhase': 'Core Time Phase'},
                                               color='Assignee')

    hours_booked_coretimephase_assignee_sprint = df.groupby(['Sprint', 'CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'])['Hours'].sum().reset_index()
    hours_booked_coretimephase_assignee_sprint_fig = px.treemap(
        hours_booked_coretimephase_assignee_sprint,
        path=['Sprint', 'CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'],
        values='Hours',
        title='CoreTime Hours by Sprint'
    )
    hours_booked_coretimephase_assignee_sprint_fig.update_layout(
        height=600,
        width=600,
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


    
    # Calculate assignee capacity
    assignee_capacity = calculate_assignee_capacity(df, df['Avg_Ratio'])
    df['Assignee Capacity'] = assignee_capacity

    assignee_capacity_fig = px.box(
        df,
        x='Assignee',
        y='Assignee Capacity',
        title='Assignee Capacity in Sprint'
    )

    # Add data labels to the box plot
    assignee_capacity_fig.update_traces(
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        hovertemplate='Capacity: %{y}<br><extra></extra>'
    )
    # Set the maximum value of the Y-axis to 100
    assignee_capacity_fig.update_yaxes(range=[0, 100])
    
    
    # Calculate average ratio by issue type and assignee
    avg_ratio_data = df.groupby(['Issue Type', 'Assignee'])['Avg_Ratio'].mean().reset_index()
    
    # Sum 'Task' and 'Sub-task' values and update the 'Issue Type' accordingly
    avg_ratio_data.loc[avg_ratio_data['Issue Type'].isin(['Task', 'Sub-task']), 'Issue Type'] = 'Task & Sub-task'
    
    # Filter the data for only Task & Sub-task and Bug issue types
    filtered_avg_ratio_data = avg_ratio_data[avg_ratio_data['Issue Type'].isin(['Task & Sub-task', 'Bug'])]
    
    # Define custom colors for the chart
    color_map = {'Bug': 'darkred', 'Task & Sub-task': 'blue'}
    
    # Create the chart with custom colors
    avg_ratio_chart = px.bar(
        filtered_avg_ratio_data,
        x='Assignee',
        y='Avg_Ratio',
        color='Issue Type',
        barmode='group',  # Set the barmode to 'group' for side-by-side bars
        color_discrete_map=color_map,  # Apply the custom colors
        title='Average Ratio by Issue Type and Assignee',
        labels={'Issue Type': 'Issue Type', 'Avg_Ratio': 'Average Ratio', 'Assignee': 'Assignee'}
    )

    avg_ratio_chart.update_traces(texttemplate='%{value:.2f}', textposition='inside')


    # Add new chart: Line chart of Average Ratio by Sprint and Assignee
    line_chart_avg_ratio = px.line(
        df.groupby(['Sprint', 'Assignee'])['Avg_Ratio'].mean().reset_index(),
        x='Sprint',
        y='Avg_Ratio',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Avg_Ratio': 'Average Ratio', 'Assignee': 'Assignee'},
        title='Average Ratio by Sprint and Assignee'
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
    
    # Add a search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)

#----------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------#


def display_tab5(df, assignee_rates):
    # Remove whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    # First, filter the data for the current sprint
    current_sprint_number = get_last_sprint_number(df)  # Assuming you have a function to get the current sprint number
    df_current_sprint = df[df['Sprint'].str.contains(str(current_sprint_number))]

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

    # Calculate the 'Resolution Time' column
    df_current_sprint['Resolution Time'] = (df_current_sprint['Resolved'] - df_current_sprint['Created']).dt.days

    total_story_points_current_sprint = df_current_sprint['Story Points'].sum()

    after_date = datetime.datetime(2023, 8, 1, 9, 59, 0)
    df_after_date = df_current_sprint[df_current_sprint['Created'] > after_date]
    total_story_points_after_date = df_after_date['Story Points'].sum()

    data_added_to_sprint = pd.DataFrame({
        'Category': ['Total in Current Sprint', 'Added After 01/08/2023 09:59:00'],
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



    # Aggregate the resolution time by assignee
    resolution_by_assignee = df_current_sprint.groupby('Assignee')['Resolution Time'].sum().reset_index()

    resolution_time_fig = px.bar(
        resolution_by_assignee,
        x='Assignee',
        y='Resolution Time',
        title=f'Under construction - Resolution Time per Assignee in Sprint {current_sprint_number}',
        labels={'Resolution Time': 'Total Resolution Time (days)'},
        orientation='v'
    )
    resolution_time_fig.update_traces(texttemplate='%{value}', textposition='outside')


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


    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(assignee_capacity_fig, use_container_width=True)
        st.plotly_chart(resolution_time_fig, use_container_width=True)

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










def display_Ad_Hoc_Analysis(df, assignee_rates):
    df['Work date'] = pd.to_datetime(df['Work date'], infer_datetime_format=True)
    # Assignee rates are already passed as an argument, no need to call get_assignee_rates again here
    df['Cost'] = df['Hours'] * df['Assignee'].map(assignee_rates)

    treemap_data = df.groupby(['CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity', pd.Grouper(key='Work date', freq='M')])['Cost'].sum().reset_index()

    treemap_fig = px.treemap(
        treemap_data,
        path=['Work date', 'CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'],
        values='Cost',
        title='Cost by CoreTimeClient, Project, Phase, Activity, and Month'
    )

    treemap_fig.update_layout(
        height=600,
        width=1000,
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

    # Add new chart: Line chart of Delivered Story Points by Sprint and Assignee
    line_chart_delivered_sp = px.line(
        df.groupby(['Sprint', 'Assignee'])['Story Points'].sum().reset_index(),
        x='Sprint',
        y='Story Points',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Story Points': 'Delivered Story Points', 'Assignee': 'Assignee'},
        title='Delivered Story Points by Sprint and Assignee'
    )

    # Add new chart: Line chart of Average Ratio by Sprint and Assignee
    line_chart_avg_ratio = px.line(
        df.groupby(['Sprint', 'Assignee'])['Avg_Ratio'].mean().reset_index(),
        x='Sprint',
        y='Avg_Ratio',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Avg_Ratio': 'Average Ratio', 'Assignee': 'Assignee'},
        title='Average Ratio by Sprint and Assignee'
    )

    # Add new chart: Line chart of Time Booked by Sprint and Assignee
    line_chart_time_booked = px.line(
        df.groupby(['Sprint', 'Assignee'])['Hours'].sum().reset_index(),
        x='Sprint',
        y='Hours',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Hours': 'Time Booked', 'Assignee': 'Assignee'},
        title='Time Booked by Sprint and Assignee'
    )

    # Add new chart: Line chart of Delivered Story Points vs Booked Hours by Sprint for Aleksander 
    aleksander_data = df[df['Assignee'] == 'Aleksander ']
    line_chart_delivered_vs_hours = px.line(
        aleksander_data.groupby('Sprint')[['Story Points', 'Hours']].sum().reset_index(),
        x='Sprint',
        y=['Story Points', 'Hours'],
        labels={'Sprint': 'Sprint', 'value': 'Value', 'variable': 'Metric'},
        title='Delivered Story Points vs Booked Hours by Sprint (Aleksander )'
    )

    # Add new chart: Line chart of Delivered Story Points and Average Ratio by Sprint for Aleksander 
    line_chart_delivered_and_ratio = px.line(
        aleksander_data.groupby('Sprint').agg({'Story Points': 'sum', 'Avg_Ratio': 'mean'}).reset_index(),
        x='Sprint',
        y=['Story Points', 'Avg_Ratio'],
        labels={'Sprint': 'Sprint', 'value': 'Value', 'variable': 'Metric'},
        title='Delivered Story Points and Average Ratio by Sprint (Aleksander )'
    )

    #st.plotly_chart(line_chart_delivered_sp, line_chart_avg_ratio, line_chart_time_booked, line_chart_delivered_vs_hours, line_chart_delivered_and_ratio, treemap_fig)
    #st.plotly_chart(line_chart_time_booked, line_chart_delivered_vs_hours, line_chart_delivered_and_ratio, treemap_fig)



#---------------------------------------------------------------------------------------------------#
def run_app():
    st.set_page_config(layout='wide')

    # Tabs at the top of the sidebar
    tabs = {
        "Current Sprint": display_tab5,
        "Team Performance": display_tab2,
        "Individual Performance": display_tab4,
        "Costs": display_tab1,
        "Productivity & Workload": display_tab3,
        "Ad Hoc Analysis": display_Ad_Hoc_Analysis
    }
    
    selected_tab = st.sidebar.radio("Select a Tab", list(tabs.keys()))
    if 'selected_tab' not in st.session_state or st.session_state.selected_tab != selected_tab:
        st.session_state.selected_tab = selected_tab  # Store the selected tab in session state

    # Place the file uploaders at the bottom of the sidebar
    with st.sidebar.expander("Upload Files"):
        uploaded_file_iterative = st.file_uploader("Choose Iterative CSV file", type="csv")
        uploaded_file_eigen = st.file_uploader("Choose Eigen CSV file", type="csv")

    # If both files are uploaded, process them
    df = None
    if uploaded_file_iterative and uploaded_file_eigen:
        df = load_data(uploaded_file_iterative, uploaded_file_eigen)

    if df is not None:
        # Get the last sprint number
        last_sprint_number = get_last_sprint_number(df)
        st.title(f"Dev Sprint 80 - {last_sprint_number}")

        # Collapsible section for assignee rates
        with st.expander("Assignee Rates"):
            assignee_rates = get_assignee_rates(df)  # Retrieve assignee rates

        # Call the appropriate function for the selected tab
        selected_function = tabs[selected_tab]
        selected_function(df, assignee_rates)  # Pass assignee_rates as a parameter
    else:
        st.warning("Upload files.")

if __name__ == "__main__":
    run_app()
