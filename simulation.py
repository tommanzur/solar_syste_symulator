import math
import random
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots

base_message = {
    "time": "2024-02-19 00:11:06Z",
    "system_id": "B12388",
    "powerhub_id": "Minimalistech-simulator",
    "inverter_id": "JKQ3C31D0120020301",
    "system_version": 2,
    "vendor_id": "ELECTRIQ",
    "inverter_temperature":31.8,
    "inverter_voltage":238.5,
    "inverter_current":0.0,
    "inverter_power":0.0,
    "inverter_frequency":0.0,
    "grid_import":1851.0,
    "grid_export":0.0,
    "load_power":0.0,
    "whole_home_power":1857.0,
    "solar_power":0.0,
    "solar1_power":0.0,
    "solar1_current":0.0,
    "solar1_voltage":0.0,
    "solar1_temperature":0.0,
    "solar2_power":0.0,
    "solar2_current":0.0,
    "solar2_voltage":0.0,
    "solar2_temperature":0.0,
    "battery_charge":0.0,
    "battery_discharge":6.0,
    "battery_voltage":108.9,
    "battery_current":0.0,
    "battery_charge_state":99.0,
    "battery_temperature":26.3,
    "meter_power_l2":-965.0,
    "meter_power_l1":-886.0,
    "max_cell_volt":3.553,
    "min_cell_volt":3.381
}

# Constants for the simulation
RAND_VALUE = random.uniform(0.01, 0.1)
PEAK_SOLAR = 1750
SUNRISE_HOUR = 6
SUNSET_HOUR = 18
PEAK_HOME_CONSUMPTION = 2200  # Peak consumption
BASELINE_CONSUMPTION_KW = random.uniform(300, 500)  # Minimum baseline consumption
GRID_EXPORT = 1500
PEAK_BATTERY_CHARGE_STATE = 100
MIN_BATTERY_CHARGE_STATE = 30
PEAK_INVERTER_TEMPERATURE = 122  # Degrees Fahrenheit
MIN_INVERTER_TEMPERATURE = 68  # Degrees Fahrenheit
PEAK_INVERTER_VOLTAGE = 240  # Volts
MIN_INVERTER_VOLTAGE = 200  # Volts
MAX_BATTERY_VOLTAGE = 120  # Volts
MIN_BATTERY_VOLTAGE = 96  # Volts
PEAK_BATTERY_TEMPERATURE = 86  # Degrees Fahrenheit
MIN_BATTERY_TEMPERATURE = 68  # Degrees Fahrenheit
SPIKE_POWER = 500
SPIKE_DURATION = 5
SPIKE_INTERVAL = 30
CLOUD_COVERAGE = random.uniform(0.0, 0.8)  # Representing 50% cloud coverage, range [0, 1]
SEASONAL_FACTOR = random.uniform(0.0, 1.0)  # Winter, range [0, 1], 1 being summer
LATITUDE_FACTOR = random.uniform(0.0, 1.0)  # Example for a specific latitude, range [0, 1]
API_KEY = ""

def calculate_solar_efficiency(cloud_coverage, seasonal_factor, latitude_factor):
    """
    Calculate the solar efficiency based on cloud coverage, seasonal variation, and latitude.

    :param cloud_coverage: A float representing the percentage of cloud coverage, range [0, 1].
    :param seasonal_factor: A float representing the seasonal variation in solar irradiance, range [0, 1].
    :param latitude_factor: A float representing the impact of latitude on solar irradiance, range [0, 1].
    :return: A float representing the overall solar efficiency.
    """
    # Assuming that clouds reduce efficiency linearly
    cloud_impact = 1 - cloud_coverage

    # Overall solar efficiency is a product of all factors
    solar_efficiency = cloud_impact * seasonal_factor * latitude_factor
    return solar_efficiency

def simulate_cloud_coverage(hour, base_cloud_coverage, variability=0.2):
    """
    Simulate cloud coverage variability throughout the day.

    :param hour: Integer representing the current hour of the day.
    :param base_cloud_coverage: Base cloud coverage, range [0, 1].
    :param variability: Maximum variability in cloud coverage, range [0, 1].
    :return: Float representing the simulated cloud coverage for the hour.
    """
    # Random fluctuation to simulate cloud variability
    fluctuation = random.uniform(-variability, variability)
    cloud_coverage = base_cloud_coverage + fluctuation
    return max(min(cloud_coverage, 1), 0)  # Ensure within [0, 1]

def get_weather_data(api_key, location):
    """Fetch weather data from Visual Crossing Weather API."""
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}?key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    return None

def adjust_solar_generation(weather_data):
    """
    Adjust solar power generation based on weather conditions.
    For simplicity, we'll just use cloud cover here.
    """
    daily_data = weather_data['days'][0]
    cloud_cover_percentage = daily_data['cloudcover']
    solar_efficiency_factor = 1 - (cloud_cover_percentage / 100)
    return solar_efficiency_factor

def simulate_solar_power(hour, solar_efficiency_factor, sunrise_hour=SUNRISE_HOUR, sunset_hour=SUNSET_HOUR, peak_solar=PEAK_SOLAR):
    """Simulate solar power generation based on the hour of the day and weather conditions."""
    if sunrise_hour <= hour <= sunset_hour:
        radians = ((hour - sunrise_hour) / (sunset_hour - sunrise_hour)) * np.pi
        solar_power = peak_solar * np.sin(radians) * solar_efficiency_factor
    else:
        solar_power = 0
    return max(solar_power, 0)

def adjust_value_randomly(value, variation=RAND_VALUE):
    """Randomly adjust a given value within a specified variation range."""
    adjustment_factor = 1 + random.uniform(-variation, variation)
    return value * adjustment_factor

def simulate_home_consumption(hour, current_minute):
    if 8 <= hour <= 17:
        consumption_factor = 2 - 0.5 * math.cos(math.pi * (hour - 8) / 9)
    elif 22 <= hour < 24:
        consumption_factor = 1.5 - 0.5 * math.cos(math.pi * (hour - 22) / 2)
    else:
        consumption_factor = 1
    base_consumption = BASELINE_CONSUMPTION_KW * consumption_factor
    appliance_consumption = 0
    if 18 <= hour <= 20:
        appliance_consumption += 1500
    if 7 <= hour <= 9:
        appliance_consumption += 1000
    total_consumption = base_consumption + appliance_consumption
    if (current_minute % SPIKE_INTERVAL) < SPIKE_DURATION:
        spike_consumption = SPIKE_POWER
    else:
        spike_consumption = 0
    total_consumption += spike_consumption

    return total_consumption

def simulate_grid_interaction(solar_power, consumption, battery_soc):
    """Simulate grid import/export based on solar power and consumption."""
    battery_energy = battery_soc * 10
    surplus = (solar_power + battery_energy) - consumption
    grid_export = min(surplus, GRID_EXPORT)
    grid_import = abs(surplus) if surplus < 0 else 0
    return grid_import, grid_export

PEAK_BATTERY_CHARGE_STATE = 100
MIN_BATTERY_CHARGE_STATE = 30

def simulate_battery_charge_discharge(solar_power, home_consumption, current_time):
    """Simulate battery charging and discharging with a smooth sinusoidal pattern."""
    hour = current_time.hour + current_time.minute / 60
    battery_charge = 0
    battery_discharge = 0

    if SUNRISE_HOUR <= hour < 12:
        radians = ((hour - SUNRISE_HOUR) / (18 - SUNRISE_HOUR)) * np.pi
        battery_charge = solar_power * np.sin(radians)
    elif (SUNSET_HOUR - 4) <= hour <= 24:
        radians = ((hour - (SUNSET_HOUR)) / (24 - (SUNSET_HOUR))) * np.pi
        battery_discharge = home_consumption * np.sin(radians)

    battery_charge = max(battery_charge, 0)
    battery_discharge = max(battery_discharge, 0)

    battery_charge = min(battery_charge, solar_power)
    battery_discharge = min(battery_discharge, home_consumption)

    return battery_charge, battery_discharge

def simulate_battery_charge_state(
    battery_charge, battery_discharge, initial_soc, time_delta
):
    """
    Simulate the battery charge state with gradual charging and discharging.
    """
    charge_rate = 0.5 / 60  # SOC increase per minute while charging
    discharge_rate = 0.9 / 60  # SOC decrease per minute while discharging
    if battery_charge > 0:
        soc_change = charge_rate * battery_charge * time_delta
    elif battery_discharge > 0:
        soc_change = -discharge_rate * battery_discharge * time_delta
    else:
        soc_change = -discharge_rate * time_delta + .1
    new_soc = initial_soc + soc_change
    new_soc = max(min(new_soc, PEAK_BATTERY_CHARGE_STATE), MIN_BATTERY_CHARGE_STATE)

    return new_soc


def simulate_with_fluctuations(hour, peak_value, min_value):
    """Simulate a parameter with fluctuations based on the hour."""
    if adjust_value_randomly(SUNRISE_HOUR) <= hour <= adjust_value_randomly(SUNSET_HOUR):
        value = peak_value
    else:
        value = min_value
    return adjust_value_randomly(value)

def time_to_fraction_of_day(current_time):
    """Convert current time to a fraction of the day."""
    total_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
    return total_seconds / (24 * 3600)

def modify_message_with_simulation(message, message_id, current_time, weather_data):
    """Modify the message based on the simulation."""
    day_fraction = time_to_fraction_of_day(current_time)
    hour = day_fraction * 24
    current_minute = current_time.minute
    home_consumption = simulate_home_consumption(hour, current_minute)
    solar_efficiency_factor = adjust_solar_generation(weather_data)
    solar_power = simulate_solar_power(hour, solar_efficiency_factor)
    battery_charge, battery_discharge = simulate_battery_charge_discharge(
        solar_power, home_consumption, current_time
        )
    battery_charge_state = simulate_battery_charge_state(
        battery_charge, battery_discharge, 80, hour
        )
    grid_import, grid_export = simulate_grid_interaction(solar_power, home_consumption, battery_charge_state)
    message.update({
        'system_id': f"Simulator{message_id}",
        'powerhub_id': f"Powerhub{message_id}",
        'whole_home_power': home_consumption,
        'grid_import': grid_import,
        'grid_export': grid_export,
        'solar_power': solar_power,
        'battery_charge_state': battery_charge_state,
        'solar1_power': adjust_value_randomly(solar_power / 2),
        'solar2_power': adjust_value_randomly(solar_power / 2),
        'battery_charge': battery_charge,
        'battery_discharge': battery_discharge,
        'meter_power_l1': adjust_value_randomly(home_consumption / 2),
        'meter_power_l2': adjust_value_randomly(home_consumption / 2)
    })

def generate_datapoints(base_message, weather_data, start_date, hours=24, points_per_hour=360):
    """Generate datapoints for simulation."""
    RAND_VALUE = random.uniform(0.01, 0.1)
    datapoints = []
    for hour in range(hours):
        for point in range(points_per_hour):
            current_time = start_date + timedelta(hours=hour, minutes=point * (60 / points_per_hour))
            modified_message = base_message.copy()
            modified_message['time'] = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            modify_message_with_simulation(modified_message, 1, current_time, weather_data)
            datapoints.append(modified_message)
    return datapoints

def plot_data(df, column_names):
    fig = make_subplots(rows=len(column_names), cols=1, subplot_titles=column_names)
    for i, column in enumerate(column_names, start=1):
        fig.add_trace(
            go.Bar(x=df['time'], y=df[column], name=column),
            row=i, col=1
        )

    fig.update_layout(height=300 * len(column_names), title_text="Bar Chart Visualization")
    fig.show()

start_time = time.time()

start_date = datetime(2024, 2, 19)
weather_data = get_weather_data(API_KEY, location='California')
datapoints = generate_datapoints(base_message=base_message, start_date=start_date, weather_data=weather_data, points_per_hour=60)
df = pd.DataFrame(datapoints)
factors = {
    'grid_export': 1500 / df['grid_export'].max(),
    'grid_import': 1500 / df['grid_import'].max(),
    'whole_home_power': 2000 / df['whole_home_power'].max(),
    'battery_discharge': 1200 / df['battery_discharge'].max(),
    'battery_charge': 500 / df['battery_charge'].max(),
}
for column, factor in factors.items():
    df[column] *= factor
df['time'] = pd.to_datetime(df['time'])

columns_to_plot = [
    'grid_export',
    'grid_import',
    'whole_home_power',
    'solar_power',
    'battery_discharge',
    'battery_charge',
    'battery_charge_state'
    ]
plot_data(df, columns_to_plot)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
