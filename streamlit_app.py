import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide
    st.title("Scrum Team Performance")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        assignee_rates = get_assignee_rates(df)  # Function to get assignee rates from the user
        st.write(df)
        display_assignee_rates(assignee_rates)

        # Use st.sidebar for the tab layout and assignee rates
        selected_tab = st.sidebar.radio("Select Tab", ("Tab 1",))
        if selected_tab == "Tab 1":
            display_tab1(df, assignee_rates)

    st.sidebar.markdown(
        """
        ---
        Created by [Tsitsi Dalakishvili](https://www.linkedin.com/in/tsitsi-dalakishvili/).
        """
    )

    # Add custom CSS styles
    components.html(
        f'<link rel="stylesheet" type="text/css" href="https://raw.githubusercontent.com/tsitsidalakishvili/ScrumTeamPerformace/main/style.css">',
        height=0
    )

def get_assignee_rates(df):
    df['Assignee'].fillna('Unassigned', inplace=True)
    assignees = df['Assignee'].unique()
    assignee_rates = {}
    default_rate = 40.0  # Set the default rate here

    for assignee in assignees:
        if assignee != 'Unassigned':  # Skip assigning rate for 'Unassigned'
            rate = st.sidebar.number_input(f"Rate for Assignee {assignee}", min_value=0.0, step=0.01, key=assignee, value=default_rate)
            assignee_rates[assignee] = rate

    return assignee_rates

def display_assignee_rates(assignee_rates):
    st.sidebar.title("Assignee Rates")
    for assignee, rate in assignee_rates.items():
        st.sidebar.write(f"Rate for Assignee {assignee}: {rate}")

def convert_seconds_to_days(seconds):
    hours = seconds / 3600
    days = hours / 8  # Assuming 8 hours per day
    return days

def display_tab1(df, assignee_rates):
    # Example: Plotting a line chart for average ratio by assignee and sprint
    df['avg_ratio'] = df['Custom field (Story Points)'] / df['Time Spent'].apply(convert_seconds_to_days)
    avg_ratio_by_assignee_sprint = df.groupby(['Assignee', 'Sprint'])['avg_ratio'].mean().reset_index()
    fig1 = px.line(avg_ratio_by_assignee_sprint, x='Sprint', y='avg_ratio', color='Assignee')
    fig1.update_layout(title="Average Ratio by Assignee and Sprint")

    # Example: Plotting a radar chart for average ratio by assignee and project name
    avg_ratio_by_assignee_project = df.groupby(['Assignee', 'Project name'])['avg_ratio'].mean().reset_index()
    fig2 = px.line_polar(avg_ratio_by_assignee_project, r='avg_ratio', theta='Project name', color='Assignee', line_close=True)
    fig2.update_layout(title="Average Ratio by Assignee and Project Name")

    # Example: Plotting a scatter plot for delivered story points vs time spent
    df['Time Spent (Days)'] = df['Time Spent'].apply(convert_seconds_to_days)
    fig3 = px.scatter(df, x='Time Spent (Days)', y='Custom field (Story Points)', color='Project name')
    fig3.update_layout(title="Delivered Story Points vs Time Spent")

    # Example: Plotting a bubble chart for workload: delivered story points by sprint and assignee
    workload_by_sprint_assignee = df.groupby(['Sprint', 'Assignee'])['Custom field (Story Points)'].sum().reset_index()
    fig4 = px.scatter(workload_by_sprint_assignee, x='Sprint', y='Assignee', size='Custom field (Story Points)', color='Assignee')
    fig4.update_layout(title="Workload: Delivered Story Points by Sprint and Assignee")

    # Display charts in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    with col2:
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)

if __name__ == "__main__":
    main()
