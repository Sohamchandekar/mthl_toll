import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

def generate_date_wise_dict(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    # Initialize an empty dictionary to hold the data
    date_wise_dict = {}

    # Iterate through each unique date in the 'Date' column
    for date in df['Date'].unique():
        # Filter the rows for the current date
        date_df = df[df['Date'] == date]

        # Initialize a sub-dictionary for the current date
        vehicle_data = {}

        # Iterate through each row of the filtered DataFrame (for the current date)
        for index, row in date_df.iterrows():
            vehicle_class = row['Vehicle_Class']
            # Create the dictionary for the current vehicle class
            vehicle_info = {
                'fasttag': {
                    'Traffic': row['fasttag_Traffic'],
                    'Revenue': row['fasttag_Revenue']
                },
                'cash': {
                    'Traffic': row['cash_Traffic'],
                    'Revenue': row['cash_Revenue']
                },
                'upi': {
                    'Traffic': row['upi_Traffic'],
                    'Revenue': row['upi_Revenue']
                },
                'authorized': {
                    'Traffic': row['authorized_Traffic'],
                    'Revenue': row['authorized_Revenue']
                },
                'force_exempt': {
                    'Traffic': row['force_exempt_Traffic'],
                    'Revenue': row['force_exempt_Revenue']
                },
                'run_through': {
                    'Traffic': row['run_through_Traffic'],
                    'Revenue': row['run_through_Revenue']
                },
                'challanable': {
                    'Traffic': row['challanable_Traffic'],
                    'Revenue': row['challanable_Revenue']
                }
            }
            # Add the vehicle class data to the date's sub-dictionary
            vehicle_data[vehicle_class] = vehicle_info

        # Add the date and its vehicle data to the overall dictionary
        date_wise_dict[date] = vehicle_data

    return date_wise_dict


# Convert data into a DataFrame
def aggregate_traffic(data):
    total_traffic = 0
    vehicle_type_traffic = {vehicle_type: 0 for vehicle_type in
                            data[next(iter(data))]}  # Initialize dict for vehicle types

    # Sum traffic for each day
    for date, vehicles in data.items():
        for vehicle_type, methods in vehicles.items():
            vehicle_type_traffic[vehicle_type] += sum(method['Traffic'] for method in methods.values())
            total_traffic += sum(method['Traffic'] for method in methods.values())

    # Define custom colors for the pie chart
    custom_colors = px.colors.sequential.Agsunset  # Use a predefined color palette

    # Format values with commas
    formatted_values = [f"{value:,}" for value in vehicle_type_traffic.values()]
    legend_labels = [f"{category}: <b>{formatted_value}</b>" for category, formatted_value in
                     zip(vehicle_type_traffic.keys(), formatted_values)]

    # Create the pie chart using Plotly Graph Objects
    fig = go.Figure(
        data=[go.Pie(
            labels=legend_labels,  # Use the custom labels for legends
            values=list(vehicle_type_traffic.values()),
            textinfo='percent',  # Show only percentage inside slices
            marker=dict(colors=custom_colors),  # Apply custom color scheme
            insidetextorientation='auto',  # Adjust text orientation to prevent overlap
            rotation=95  # Rotate the pie chart by 45 degrees
        )]
    )

    # Update layout and styling
    fig.update_layout(
        title=dict(
            text=f"<b>Vehicle Wise Traffic Distribution</b><br>Total vehicles: {total_traffic:,}",
            x=0.5,  # Center align the title
            font=dict(size=15)  # Reduce font size
        ),
        legend=dict(
            orientation="h",  # Arrange legend horizontally
            yanchor="bottom",  # Align legend below the chart
            y=-0.35,
            x=0.55,
            xanchor="center",
            font=dict(size=10),  # Reduce font size for the legend
        ),
        height=430,  # Set height for a square aspect ratio
        width=600,  # Set width for a square aspect ratio
        paper_bgcolor='rgba(255, 255, 255, 0.0)',
        margin=dict(t=80, b=80, l=10, r=10)  # Add some padding
    )

    return fig

def process_and_plot_traffic(data):
    # Flatten the data into a DataFrame
    records = []
    for date, vehicles in data.items():
        for vehicle_type, methods in vehicles.items():
            total_traffic = sum(method['Traffic'] for method in methods.values())
            records.append({'Date': date, 'VehicleType': vehicle_type, 'Traffic': total_traffic})

    df = pd.DataFrame(records)

    # Add Week and Month columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='d')
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    # Aggregate data for each timeframe
    aggregates = {
        'Daily': df.groupby(['Date', 'VehicleType'])['Traffic'].sum().reset_index(),
        'Weekly': df.groupby(['Week', 'VehicleType'])['Traffic'].sum().reset_index(),
        'Monthly': df.groupby(['Month', 'VehicleType'])['Traffic'].sum().reset_index(),
    }

    # Create Plotly figure
    fig = go.Figure()
    vehicle_types = df['VehicleType'].unique().tolist() + ['Total']
    timelines = ['Daily', 'Weekly', 'Monthly']
    column_map = {'Daily': 'Date', 'Weekly': 'Week', 'Monthly': 'Month'}

    # Use the Agsunset color theme
    colors = px.colors.sequential.Agsunset
    color_map = {vehicle: colors[i % len(colors)] for i, vehicle in enumerate(vehicle_types)}

    trace_map = {}

    # Add traces dynamically for each vehicle type and timeline
    for timeline, agg_data in aggregates.items():
        for vehicle_type in vehicle_types:
            if vehicle_type == "Total":
                # Total traffic
                total_data = agg_data.groupby(column_map[timeline])['Traffic'].sum().reset_index()
                x_data = total_data[column_map[timeline]]
                y_data = total_data['Traffic']
            else:
                # Individual vehicle type
                filtered_data = agg_data[agg_data['VehicleType'] == vehicle_type]
                x_data = filtered_data[column_map[timeline]]
                y_data = filtered_data['Traffic']

            trace = go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',  # Show both lines and markers
                line=dict(color=color_map[vehicle_type], width=2),  # Assign color and line width
                marker=dict(size=6),  # Set marker size
                text=[f"{y:,}" for y in y_data],  # Display formatted values
                textposition="top center",
                name=f"{vehicle_type} ({timeline})",
                visible=(vehicle_type == 'Total' and timeline == 'Daily')  # Default visibility
            )
            fig.add_trace(trace)
            trace_map[(timeline, vehicle_type)] = len(fig.data) - 1  # Map trace index

    # Dropdown for Vehicle Type
    vehicle_dropdown = []
    for vehicle_type in vehicle_types:
        visible_flags = [False] * len(fig.data)
        for timeline in timelines:
            visible_flags[trace_map[(timeline, vehicle_type)]] = True if timeline == 'Daily' else False
        vehicle_dropdown.append(dict(
            label=vehicle_type,
            method="update",
            args=[{"visible": visible_flags},
                  {"title": f"<b>Traffic Trend</b> <br>of {vehicle_type}"}]
        ))

    # Dropdown for Timeframe
    timeline_dropdown = []
    for timeline in timelines:
        visible_flags = [False] * len(fig.data)
        for vehicle_type in vehicle_types:
            visible_flags[trace_map[(timeline, vehicle_type)]] = (vehicle_type == 'Total')
        timeline_dropdown.append(dict(
            label=timeline,
            method="update",
            args=[{"visible": visible_flags},
                  {"title": f"Traffic Data ({timeline})"}]
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Traffic Trend</b> <br>of {vehicle_type}",
            x=0.5,  # Center align the title
            font=dict(size=15)
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title="Traffic Count",
            showgrid=True,
            gridcolor='lightgrey',
            automargin=True
        ),
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,  # Center horizontally
            xanchor="center",
            y=-0.2,  # Below the plot
            yanchor="top"
        ),
        width=900,  # Fixed width
        height=450,  # Fixed height
        margin=dict(t=60, b=80, l=60, r=20),  # Compact margins
        paper_bgcolor='rgba(255, 255, 255, 0.0)',
        plot_bgcolor='rgba(255, 255, 255, 0.0)',  # White background
        updatemenus=[
            dict(
                buttons=vehicle_dropdown,
                direction="down",
                showactive=True,
                x=0.3,
                xanchor="right",
                y=1.2,
                yanchor="top"
            ),
            dict(
                buttons=timeline_dropdown,
                direction="down",
                showactive=True,
                x=0.10,
                xanchor="right",
                y=1.2,
                yanchor="top"
            )
        ]
    )



    # Show the figure
    return fig

def plot_daily_run_through_traffic(data):
    """
    Creates a Plotly bar chart for daily run-through traffic with continuous coloring and fixed size.

    Parameters:
        data (dict): A nested dictionary where keys are dates, vehicle types, and payment types.

    Returns:
        plotly.graph_objs._figure.Figure: An interactive Plotly figure.
    """
    # Flatten data to create a DataFrame
    rows = []
    for date, vehicle_data in data.items():
        for vehicle_type, payment_data in vehicle_data.items():
            run_through_traffic = payment_data.get('run_through', {}).get('Traffic', 0)
            rows.append({
                'Date': date,
                'Vehicle_Type': vehicle_type,
                'Run_Through_Traffic': run_through_traffic,
            })

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Aggregate run-through traffic on a daily basis
    daily_run_through = df.groupby('Date')['Run_Through_Traffic'].sum().reset_index()

    # Convert 'Date' to datetime for proper sorting
    daily_run_through['Date'] = pd.to_datetime(daily_run_through['Date'])

    # Plot using Plotly with a continuous color scale
    fig = px.bar(
        daily_run_through,
        x='Date',
        y='Run_Through_Traffic',
        title='Daily Run-Through Traffic',
        labels={'Run_Through_Traffic': 'Run-Through Traffic', 'Date': 'Date'},
        text='Run_Through_Traffic',
        color='Run_Through_Traffic',  # Set the color based on traffic values
        color_continuous_scale='sunset'  # Use a visually appealing continuous color scale
    )

    # Update traces for better text positioning
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')  # Format numbers with commas

    # Update layout for fixed figure size and clean visuals
    fig.update_layout(
        width=1200,  # Fixed width
        height=500,  # Fixed height
        paper_bgcolor='rgba(255, 255, 255, 0.0)',
        plot_bgcolor='rgba(255, 255, 255, 0.0)',  # White background
        title=dict(
            text='<b>Daily Run-Through Traffic',
            x=0.5,  # Center align
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='lightgrey',
            tickangle=45,  # Rotate x-axis labels for better readability
        ),
        yaxis=dict(
            title='Run-Through Traffic',
            showgrid=True,
            gridcolor='lightgrey'
        ),
        template='plotly_white',  # Clean theme
        margin=dict(t=60, b=80, l=50, r=50),  # Compact margins
        coloraxis_showscale=False  # Turn off the color bar
    )

    return fig


def aggregate_revenue(data):
    """
    Calculates the total revenue fetched vehicle-wise and plots a pie chart styled similarly to the traffic chart.

    Parameters:
        data (dict): A nested dictionary containing date-wise vehicle revenue details.

    Returns:
        tuple: Total revenue and a dictionary with revenue per vehicle type.
    """
    total_revenue = 0
    vehicle_type_revenue = {vehicle_type: 0 for vehicle_type in data[next(iter(data))]}  # Initialize dict for vehicle types

    # Sum revenue for each vehicle type
    for date, vehicles in data.items():
        for vehicle_type, methods in vehicles.items():
            vehicle_type_revenue[vehicle_type] += sum(method['Revenue'] for method in methods.values())
            total_revenue += sum(method['Revenue'] for method in methods.values())

    # Define custom colors for the pie chart
    custom_colors = px.colors.sequential.Agsunset  # Use a predefined color palette

    # Format values with commas
    formatted_values = [f"₹{value:,}" for value in vehicle_type_revenue.values()]
    legend_labels = [f"{category}: <b>{formatted_value}</b>" for category, formatted_value in zip(vehicle_type_revenue.keys(), formatted_values)]

    # Create the pie chart using Plotly Graph Objects
    fig = go.Figure(
        data=[go.Pie(
            labels=legend_labels,  # Use the custom labels for legends
            values=list(vehicle_type_revenue.values()),
            textinfo='percent',  # Show only percentage inside slices
            marker=dict(colors=custom_colors),  # Apply custom color scheme
            insidetextorientation='auto',  # Adjust text orientation to prevent overlap
            rotation=145  # Rotate the pie chart by 95 degrees
        )]
    )

    # Update layout and styling
    fig.update_layout(
        title=dict(
            text=f"<b>Vehicle Wise Revenue Distribution</b><br>Total Revenue: ₹ {total_revenue:,}",
            x=0.5,  # Center align the title
            font=dict(size=15)  # Reduce font size
        ),
        legend=dict(
            orientation="h",  # Arrange legend horizontally
            yanchor="bottom",  # Align legend below the chart
            y=-0.35,
            x=0.55,
            xanchor="center",
            font=dict(size=10),  # Reduce font size for the legend
        ),
        height=430,  # Set height for a square aspect ratio
        width=680,  # Set width for a square aspect ratio
        paper_bgcolor='rgba(255, 255, 255, 0.0)',
        margin=dict(t=80, b=60, l=5, r=5)  # Add some padding


    )

    return fig


def process_and_plot_revenue(data):
    # Flatten the data into a DataFrame
    records = []
    for date, vehicles in data.items():
        for vehicle_type, methods in vehicles.items():
            total_revenue = sum(method['Revenue'] for method in methods.values())
            records.append({'Date': date, 'VehicleType': vehicle_type, 'Revenue': total_revenue})

    df = pd.DataFrame(records)

    # Add Week and Month columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='d')
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    # Aggregate data for each timeframe
    aggregates = {
        'Daily': df.groupby(['Date', 'VehicleType'])['Revenue'].sum().reset_index(),
        'Weekly': df.groupby(['Week', 'VehicleType'])['Revenue'].sum().reset_index(),
        'Monthly': df.groupby(['Month', 'VehicleType'])['Revenue'].sum().reset_index(),
    }

    # Create Plotly figure
    fig = go.Figure()
    vehicle_types = df['VehicleType'].unique().tolist() + ['Total']
    timelines = ['Daily', 'Weekly', 'Monthly']
    column_map = {'Daily': 'Date', 'Weekly': 'Week', 'Monthly': 'Month'}

    # Use the Agsunset color theme
    colors = px.colors.sequential.Agsunset
    color_map = {vehicle: colors[i % len(colors)] for i, vehicle in enumerate(vehicle_types)}

    trace_map = {}

    # Add traces dynamically for each vehicle type and timeline
    for timeline, agg_data in aggregates.items():
        for vehicle_type in vehicle_types:
            if vehicle_type == "Total":
                # Total revenue
                total_data = agg_data.groupby(column_map[timeline])['Revenue'].sum().reset_index()
                x_data = total_data[column_map[timeline]]
                y_data = total_data['Revenue']
            else:
                # Individual vehicle type
                filtered_data = agg_data[agg_data['VehicleType'] == vehicle_type]
                x_data = filtered_data[column_map[timeline]]
                y_data = filtered_data['Revenue']

            trace = go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',  # Show both lines and markers
                line=dict(color=color_map[vehicle_type], width=2),  # Assign color and line width
                marker=dict(size=6),  # Set marker size
                text=[f"₹{y:,.2f}" for y in y_data],  # Display formatted values
                textposition="top center",
                name=f"{vehicle_type} ({timeline})",
                visible=(vehicle_type == 'Total' and timeline == 'Daily')  # Default visibility
            )
            fig.add_trace(trace)
            trace_map[(timeline, vehicle_type)] = len(fig.data) - 1  # Map trace index

    # Dropdown for Vehicle Type
    vehicle_dropdown = []
    for vehicle_type in vehicle_types:
        visible_flags = [False] * len(fig.data)
        for timeline in timelines:
            visible_flags[trace_map[(timeline, vehicle_type)]] = True if timeline == 'Daily' else False
        vehicle_dropdown.append(dict(
            label=vehicle_type,
            method="update",
            args=[{"visible": visible_flags},
                  {"title": f"<b>Revenue Trend</b> <br>of {vehicle_type}"}]
        ))

    # Dropdown for Timeframe
    timeline_dropdown = []
    for timeline in timelines:
        visible_flags = [False] * len(fig.data)
        for vehicle_type in vehicle_types:
            visible_flags[trace_map[(timeline, vehicle_type)]] = (vehicle_type == 'Total')
        timeline_dropdown.append(dict(
            label=timeline,
            method="update",
            args=[{"visible": visible_flags},
                  {"title": f"<b>Revenue Data</b> <br>({timeline})"}]
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Revenue Trend</b> <br>of {vehicle_types[0]}",
            x=0.5,  # Center align the title
            font=dict(size=15)
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title="Revenue (₹)",
            showgrid=True,
            gridcolor='lightgrey',
            automargin=True
        ),
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,  # Center horizontally
            xanchor="center",
            y=-0.2,  # Below the plot
            yanchor="top"
        ),
        width=900,  # Fixed width
        height=450,  # Fixed height
        margin=dict(t=60, b=80, l=60, r=20),  # Compact margins
        paper_bgcolor='rgba(255, 255, 255, 0.0)',
        plot_bgcolor='rgba(255, 255, 255, 0.0)',  # White background
        updatemenus=[
            dict(
                buttons=vehicle_dropdown,
                direction="down",
                showactive=True,
                x=0.3,
                xanchor="right",
                y=1.2,
                yanchor="top"
            ),
            dict(
                buttons=timeline_dropdown,
                direction="down",
                showactive=True,
                x=0.10,
                xanchor="right",
                y=1.2,
                yanchor="top"
            )
        ]
    )

    # Show the figure
    return fig





def revenue_distribution_donut(data):
    """
    Creates a donut chart showing the revenue distribution for fasttag, cash, and UPI.

    Parameters:
        data (dict): A nested dictionary containing traffic and revenue data.

    Returns:
        None: Displays the Plotly figure.
    """
    # Flatten the data into a DataFrame
    records = []
    for date, vehicles in data.items():
        for vehicle_type, methods in vehicles.items():
            for payment_type, details in methods.items():
                if payment_type in ['fasttag', 'cash', 'upi']:
                    records.append({
                        'Date': date,
                        'PaymentType': payment_type,
                        'Revenue': details['Revenue']
                    })

    df = pd.DataFrame(records)

    # Aggregate total revenue by selected payment types
    revenue_by_payment = df.groupby('PaymentType')['Revenue'].sum().reset_index()
    total_revenue = revenue_by_payment['Revenue'].sum()  # Calculate total revenue
    # Define custom colors for the pie chart
    custom_colors = px.colors.sequential.Agsunset  # Use a predefined color palette

    # Create the donut chart
    fig = go.Figure(go.Pie(
        labels=revenue_by_payment['PaymentType'],
        values=revenue_by_payment['Revenue'],
        hole=0.55,  # Creates the donut shape
        textinfo='value+percent',  # Show value and percentage
        hoverinfo='label+percent+value',  # Hover information
        texttemplate='<b>₹ %{value:,}</b><br> %{percent}',  # Exact percentage with two decimals
        marker=dict(colors=custom_colors),  # Apply custom color scheme
        rotation=90,  # Rotate the chart for better positioning
        insidetextorientation='horizontal',
        textposition='outside'
    ))

    # Add annotation for total revenue in the center
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>Total Revenue</b><br> ₹ {total_revenue:,.0f}",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, family="Arial", color="black"),
                xanchor="center", yanchor="middle"
            )
        ]
    )

    # Adjust layout for transparency and horizontal alignment
    fig.update_layout(
        title=dict(
            text=f"<b>Revenue Distribution</b>",
            x=0.5,
            font=dict(size=16)
        ),
        margin=dict(t=100, b=100, l=30, r=30),
        width=400,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            x=0.5,
            xanchor="center",
            font=dict(size=9)
        )
    )

    # Show the figure
    return fig

def lifetime_revenue(data):
    """
    Calculates the lifetime revenue from a nested dictionary structure
    and generates an HTML card displaying the revenue.

    Parameters:
    data (dict): Dictionary containing revenue data for multiple dates and vehicle types.

    Returns:
    str: HTML string for the card displaying the lifetime revenue.
    """
    # Initialize total revenue
    total_revenue = 0

    # Iterate through each date and vehicle type to sum up revenue
    for date, vehicles in data.items():
        for vehicle_type, payment_methods in vehicles.items():
            for method, metrics in payment_methods.items():
                total_revenue += metrics.get('Revenue', 0)

    # Format the revenue with the ₹ symbol, comma-separated, and no decimals
    formatted_revenue = f"₹ {total_revenue:,.0f}"

    # Generate the HTML card
    revenue_card_html = f"""
    <div class="card custom-card bg-info text-white mb-3">
        <div class="card-body text-center">
            <h5 class="card-title text-blue">Lifetime Revenue</h5>
            <p class="card-value-text">{formatted_revenue}</p>
        </div>
    </div>
    """

    return revenue_card_html

def lifetime_traffic(data):
    """
    Calculates the lifetime revenue from a nested dictionary structure
    and generates an HTML card displaying the revenue.

    Parameters:
    data (dict): Dictionary containing revenue data for multiple dates and vehicle types.

    Returns:
    str: HTML string for the card displaying the lifetime revenue.
    """
    # Initialize total revenue
    total_traffic = 0

    # Iterate through each date and vehicle type to sum up revenue
    for date, vehicles in data.items():
        for vehicle_type, payment_methods in vehicles.items():
            for method, metrics in payment_methods.items():
                total_traffic += metrics.get('Traffic', 0)

    # Format the revenue with the ₹ symbol, comma-separated, and no decimals
    formatted_traffic = f"{total_traffic:,.0f}"

    # Generate the HTML card
    traffic_card_html = f"""
    <div class="card custom-card bg-info text-white mb-3">
        <div class="card-body text-center">
            <h5 class="card-title text-blue">Lifetime Traffic</h5>
            <p class="card-value-text">{formatted_traffic}</p>
        </div>
    </div>
    """

    return traffic_card_html


def daily_average_revenue(data):
    """
    Calculates the daily average revenue from a nested dictionary structure
    and generates an HTML card displaying the average revenue.

    Parameters:
    data (dict): Dictionary containing revenue data for multiple dates and vehicle types.

    Returns:
    str: HTML string for the card displaying the daily average revenue.
    """
    # Initialize total revenue and count of days
    total_revenue = 0
    total_days = len(data)  # The number of dates in the dictionary

    # Sum up revenue for all dates
    for date, vehicles in data.items():
        for vehicle_type, payment_methods in vehicles.items():
            for method, metrics in payment_methods.items():
                total_revenue += metrics.get('Revenue', 0)

    # Calculate the daily average revenue
    average_revenue = total_revenue / total_days if total_days > 0 else 0

    # Format the average revenue with the ₹ symbol, comma-separated, and no decimals
    formatted_average_revenue = f"₹ {average_revenue:,.0f}"

    # Generate the HTML card
    average_revenue_card_html = f"""
    <div class="card custom-card bg-success text-white mb-3">
        <div class="card-body text-center">
            <h5 class="card-title text-blue">Daily Average Revenue</h5>
            <p class="card-value-text">{formatted_average_revenue}</p>
        </div>
    </div>
    """

    return average_revenue_card_html

def daily_average_traffic(data):
    """
    Calculates the daily average traffic from a nested dictionary structure
    and generates an HTML card displaying the average traffic.

    Parameters:
    data (dict): Dictionary containing traffic data for multiple dates and vehicle types.

    Returns:
    str: HTML string for the card displaying the daily average traffic.
    """
    # Initialize total traffic and count of days
    total_traffic = 0
    total_days = len(data)  # The number of dates in the dictionary

    # Sum up traffic for all dates
    for date, vehicles in data.items():
        for vehicle_type, payment_methods in vehicles.items():
            for method, metrics in payment_methods.items():
                total_traffic += metrics.get('Traffic', 0)

    # Calculate the daily average traffic
    average_traffic = total_traffic / total_days if total_days > 0 else 0

    # Format the average traffic with comma-separated, and no decimals
    formatted_average_traffic = f"{average_traffic:,.0f}"

    # Generate the HTML card
    average_traffic_card_html = f"""
    <div class="card custom-card bg-success text-white mb-3">
        <div class="card-body text-center">
            <h5 class="card-title text-blue">Daily Average Traffic</h5>
            <p class="card-value-text">{formatted_average_traffic}</p>
        </div>
    </div>
    """

    return average_traffic_card_html