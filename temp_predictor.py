import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# create a pandas dataframe from the dataset csv file
temperature_data = pd.read_csv('TemperatureDataset.csv', delimiter=',', low_memory=False) 

# Convert 'date_and_time' feature to datetime
temperature_data['date_and_time'] = pd.to_datetime(temperature_data['date_and_time'])

# Filter data to only include datapoints from the year 2024
temperature_data = temperature_data[temperature_data['date_and_time'].dt.year == 2024]

# Create 'ds' column for Prophet 
temperature_data['ds'] = temperature_data['date_and_time'].dt.date

# create a new dataframe where each row is the average temperature for a uniqie date in 2024
daily_avg_temp = temperature_data.groupby('ds')['temperature_degrees_c'].mean().reset_index()

# Rename column to 'y' as stated in Prophet documentation
daily_avg_temp.rename(columns={'temperature_degrees_c': 'y'}, inplace=True)


prof = Prophet()
prof = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prof.fit(daily_avg_temp) # train the Prophet model
future= prof.make_future_dataframe(periods=365,freq='D') # predict average temperatures for the next 365 days
forecast = prof.predict(future)

# Generate the main forecast plot
fig_forecast = plot_plotly(prof, forecast)

# Save the main forecast plot as an HTML file
fig_forecast.write_html("forecast_plot.html")
print("Forecast plot saved as 'forecast_plot.html'")

# Save the main forecast plot as a PNG image
fig_forecast.write_image("forecast_plot.png")
print("Forecast plot saved as 'forecast_plot.png'")

# Generate the components plot
fig_components = plot_components_plotly(prof, forecast)

# Save the components plot as an HTML file
fig_components.write_html("components_plot.html")
print("Components plot saved as 'components_plot.html'")

# Save the components plot as a PNG image
fig_components.write_image("components_plot.png")
print("Components plot saved as 'components_plot.png'")