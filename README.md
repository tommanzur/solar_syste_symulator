# Solar Energy System Simulator

## Overview
This repository contains the code for a solar energy system simulator designed to model the performance of a residential solar power setup. The simulator takes into account various factors such as solar efficiency, home power consumption, battery charge and discharge cycles, and grid interactions.

## Features
- Simulates solar power generation based on weather data, time of day, and seasonal factors.
- Models home power consumption with variability throughout the day.
- Calculates battery charging and discharging states with a smooth sinusoidal pattern.
- Estimates grid import and export power based on surplus or deficit of power.
- Includes a visualization component to plot the data for analysis.

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/solar-energy-simulator.git

# Navigate to the repository directory
cd solar-energy-simulator

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required dependencies
pip install -r requirements.txt
```
## Usage

To run the simulation, execute the simulation.py script:

```bash
python simulation.py
Visualization
```

The plot_data function within the simulation.py script will generate bar chart visualizations for the following parameters:

- Grid Export
- Grid Import
- Whole Home Power
- Solar Power
- Battery Discharge
- Battery Charge
- Battery Charge State

The visualizations are displayed using Plotly and can be customized as needed.