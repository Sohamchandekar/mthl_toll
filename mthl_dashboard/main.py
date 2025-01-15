from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
from plotly.io import to_html
import os
from babel.numbers import format_currency
from flask import jsonify


from functions import (generate_date_wise_dict, aggregate_traffic,process_and_plot_traffic,
                       plot_daily_run_through_traffic, aggregate_revenue, process_and_plot_revenue,
                       revenue_distribution_donut)

from functions import lifetime_revenue,lifetime_traffic,daily_average_traffic,daily_average_revenue

from model import prepare_data_for_model, simulate_scenarios,train_robust_model,prepare_data,add_custom_features

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the processed data
traffic_dictionary = {}
prediction_base_data = pd.DataFrame()

@app.route('/')
def home():

    return render_template('home.html')


# Route for the dashboard tab
@app.route('/dashboard')
def dashboard():
    global prediction_base_data, model

    traffic_pie_chart = aggregate_traffic(traffic_dictionary)
    traffic_pie_chart_html = to_html(traffic_pie_chart, full_html= False)

    process_and_plot_traffic_chart = process_and_plot_traffic(traffic_dictionary)
    process_and_plot_traffic_html = to_html(process_and_plot_traffic_chart, full_html=False)

    plot_daily_run_through_traffic_chart = plot_daily_run_through_traffic(traffic_dictionary)
    plot_daily_run_through_traffic_html = to_html(plot_daily_run_through_traffic_chart, full_html=False)

    revenue_pie_chart = aggregate_revenue(traffic_dictionary)
    revenue_pie_chart_html = to_html(revenue_pie_chart, full_html= False)

    process_and_plot_revenue_chart = process_and_plot_revenue(traffic_dictionary)
    process_and_plot_revenue_html = to_html(process_and_plot_revenue_chart, full_html=False)

    revenue_distribution_donut_chart = revenue_distribution_donut(traffic_dictionary)
    revenue_distribution_donut_html = to_html(revenue_distribution_donut_chart, full_html= False)

    revenue_card = lifetime_revenue(traffic_dictionary)
    traffic_card = lifetime_traffic(traffic_dictionary)

    average_revenue_card = daily_average_revenue(traffic_dictionary)
    average_traffic_card = daily_average_traffic(traffic_dictionary)

    prediction_base_data = prepare_data_for_model(traffic_dictionary)
    prediction_base_data = add_custom_features(prediction_base_data)
    print(prediction_base_data.head(5))

    X, y = prepare_data(prediction_base_data)
    # Train the model
    model = train_robust_model(X, y)


    # Pass the global_data to the dashboard for rendering
    return render_template('dashboard.html',
                           data=traffic_dictionary,
                           traffic_pie_chart_html = traffic_pie_chart_html,
                           process_and_plot_traffic_html = process_and_plot_traffic_html,
                           plot_daily_run_through_traffic_html = plot_daily_run_through_traffic_html,
                           revenue_pie_chart_html = revenue_pie_chart_html,
                           process_and_plot_revenue_html = process_and_plot_revenue_html,
                           revenue_distribution_donut_html = revenue_distribution_donut_html,

                           revenue_card = revenue_card,
                           traffic_card = traffic_card,
                           average_revenue_card = average_revenue_card,
                           average_traffic_card = average_traffic_card)


# Route for the upload_data tab
@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    global traffic_dictionary  # Declare to modify the global variable
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Determine the file extension (to maintain compatibility)
            if file.filename.endswith('.csv'):
                filename = 'traffic_data.csv'
            elif file.filename.endswith(('.xls', '.xlsx')):
                filename = 'traffic_data.xlsx'
            else:
                flash('Invalid file format. Please upload a CSV or Excel file.')
                return redirect(request.url)

            # Save the file with a fixed name
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded file
            try:
                if filename.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif filename.endswith('.xlsx'):
                    data = pd.read_excel(file_path)

                # Pass the file path to the generate_date_wise_dict function
                traffic_dictionary = generate_date_wise_dict(file_path)


                flash('File uploaded and processed successfully!')
                return redirect(url_for('home'))
            except Exception as e:
                flash(f'Error processing file: {e}')
                return redirect(request.url)
    return render_template('upload_data.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model  # Ensure model is globally accessible

    if model is None:
        return jsonify({'error': 'Model is not trained yet. Please train the model on the dashboard.'}), 500

    # Extract inputs from the request
    vehicle_type = request.form.get('vehicle_type')
    official_toll = float(request.form.get('official_toll'))

    # Use simulate_scenarios function to get predictions
    predicted_traffic, predicted_revenue, recalculated_revenue = simulate_scenarios(
        model=model, vehicle_type=vehicle_type, official_toll=official_toll
    )

    # Format the revenue using the Indian numbering system with rupee symbol
    predicted_revenue = format_currency(predicted_revenue, 'INR', locale='en_IN')

    # Return predictions as JSON
    return jsonify({
        'predicted_traffic': round(predicted_traffic, 0),
        'predicted_revenue': predicted_revenue,  # Include the formatted revenue
    })


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
