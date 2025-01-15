UK Electricity Demand Forecasting
==============================

Executive Summary
------------------
A machine learning model was developed for accurately forecasting UK electricity demand with a mean absolute error (MAE) of 908.6MW and a mean absolute percentage error (MAPE) of 3.22%. 

The model performance significantly attreibuted to the inclusion of 1 day, 1 week, and 1 year lagged demands as well as a 1 year rolling mean. The model was trained on historical data from the last 15 years with the capability of generating demand forecasts up to 3 years in the future; thereby empowering energy providers, and grid operators to optimise generation and mitigate energy shortages. <br>

Furthermore, a dashboard was created to enable stakeholders interact with historical data, generate forecasts as well as view the model performance and the features that impact its forecasting strength.

Project Overview
-------------------
This project focuses on forecasting UK electricity demand to assist energy sector sstakeholders, such as energy providers, grid operators and policymakers. These stakeholders require accurate forecasts to balance supply and demand, optimise renewable energy integration and ensure grid stability. <br>

The project utilised machine learning model to predict electricity demand up to 3 years into the future, leveraging historical data and engineered features. <br>
Key Performance Indicators include:<br>
1. **Mean Absolute Percentage Error (MAPE)**: A measure of the model forecast accuracy in percentage terms.
2. **Mean Absolute Error (MAE)**: Quantifies the average error in megawatts (MW)
3. **Coefficient of Determination (R-squared)**: Indicates how well the model explains the variability in electricity demand.  

Objectives
------------
The objectives of the project are as follows:
1. Develop an understanding of historical electricity demand to make inferences on consumption
2. Identify key factors influencing electricity demand to improve decision-making for capacity planning and resource allocation and forecast modelling.
3. Develop a robust electricity demand forecasting model with high accuracy to support stakeholders in data-driven decisions on energy generation, storage and distribution.
4. Build a user-friendly interactive dashboard for stakeholders to visualise historical demand, and generate forecasts.
5. Develop actionable recommendations to address variability in demand based on the model insights to ensure energy reliability, and optimise cost for end users and providers. 

Data Overview
-------------
The dataset is the UK electricity demand data consisting of over approximately 1.2M records spanning 2009 to 2024 with half-hourly frequency found in the path,  data/raw/energy_supply_demand_data_30min.csv <br>

The dataset includes date, period (i.e., frequency of data capture), UK Transmission System Demand (TSD), demand from england and whales, holiday, solar and wind generation as well as their respective capacities. 

Methodology
------------
The methodology adopted for this project has been simplified in the illustration below. <br>
In essence, the process begins with collecting and analysing historical demand data to develop an understanding of electricity demand and generation. 

Furthermore, feature engineering and machine learning modelling are conducted in a process involving 3 machine learning algorithms and a 5 fold time series split cross validation. 

The resultant models are assessed using Mean Absolute Error, Mean Absolute Percentage Error, Root Mean Squared Error and R2. Finally, the best model is used for forecasting demand. <br> An interactive dashboard is also developed to enable user interaction with the model for forecasting, as well as provide the user with insights on the model features and visualisation of the historical data. <br>
![Alt text](./reports/figures/project_workflow.png)

Key Insights 
-------------
**Electricity Demand Analysis**:
- **Historical Transmission System Demand (TSD)**:
    - TSD exhibited a downward trend from 2009 to 2019, indicating reduced electricity demand over time. However, this trend weakened into a steady demand in the last 4 years (2020 to 2024).<br>This is attributed to factors such as energy efficiency improvements, reduced energy-intensive industries, and social behavioural changes with increased awareness of energy cost and climate impact and post Covid-19 work pattern changes, etc.
    - The average TSD was 32.6GW with a 7.7GW variation from 2009.  
    - Hourly TSD showed the highest demands occurred from 7am to 9pm, which aligns with the period most daily activities occurred. Nevertheless, the periods outside 7am to 9pm exhibited outliers, indicating the sparse days in which there was more demand than usual.
    - Yearly and daily seasonalities were observed. The highest and lowest annual demands occured during the colder months (September to April) and the warmer months (May to August) respectively.
    - Over time, The demand wave height (from highest to lowest demand in a year) reduced, with shorter wave heights observed in the last 4 to 5 years than before, indicating a reduced annual demand over time.
- **Solar and Wind Generation Contributions**:
    - Combined solar and wind generation contributed to over 20% of TSD from 2015 onwards, indicating the beginning of considerable contribution to UK electricity demand. In 2024, combined solar and wind generation contributions peaked at 71% of TSD.
    - Wind provided significant contribution from 2009 (100%) to 2015 (58.2%) compared to solar. However wind and solar contributions were within 40% to 60% from 2015 onwards with both having the same contributing in 2024. 
- **Model Performance**:
    - The first model was trained using the temporal features, day, week, month and 1, 2 3 and 5 year demand lags leading to an MAE of ~2423.96MW, MAPE of 8.55%, and R2 79%. 
    - An improved model was found using the features 1 year rolling mean, 1 hour, 1 day, 1 week, 1 year and 2 year lags, leading to an MAE: 908.66MW, MAPE of 3.22%, and R2 96%
    - The inclusion of 30 minute demand lag resulted in further reduced errors and a R2 of 99% but at an increased training complexity and risk over-fitting.

Dashbaord
==========
The interactive dashbaord iwas built using Streamlit and can be run using the command streamlit run electricity_demand_dashboard.py /src/visualisation/electricity_demand_dashbaord.py 

Recommendation
================
Given the steady electricity demand over the last 4 years, with occassional/sparse increases in demand which introduces variability, generation and storage capacities should be adjusted dynamically with sufficient storage applications on stand by. 


