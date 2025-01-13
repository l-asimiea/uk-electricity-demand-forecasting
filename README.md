UK Electricity Demand Forecasting
==============================

Project Overview
-------------------
This project aims to forecast uk electricity demand using advanced machine learning models and provide insights to energy providers using an interactive dashboard. 
It involved leveraging historical data on electricity demand, as well as incoporating temporal, statistical and lag features in building a model suitable for making accurate forecasts. This model empowers users such as energy providers and asset managers to effectively allocate, manage and balance energy generation resources against demand.  

Objectives
------------
The objectives of the project are as follows:
1. Develop an understanding of historical electricity demand to obtain useful features for a forecast modelling.
2. Explore and evaluate machine learning models to achieve a forecasting model with high accuracy, and low error metrics.
3. Build a user-friendly interactive dashboard for technical and non-technical stakeholders to visualise historical electricity demand and forecasts and the major features contributing to the model forecasting ability.

Data Overview
-------------
- **Dataset**: UK electricity demand data at 30 minutes intervals, including solar and wind generation features from Januray 2009 to October 2024 (15 years). Link:
- **Key Features**: The features of interest include:
    - Temporal features: Features such as hour, day, week, month and year extractable from the date as well as 1 year rolling mean
    - Lagged features: Short-term shifts (1 hour, 1 day, 1 week) and long-term shifts (1 year and 2 year)
    - Target varibale: Total System Demand, named tagged as tsd 
Other variables reviewed in the dataset include, solar and wind capacities. 

Methodology
------------
The methodology adopted for this project has been simplified in the illustration below. In essence, the process begins with collecting and analysing historical demand data to develop an understanding of electricity demand and generation. After which, feature engineering and machine learning modelling are conducted in a process involving 3 machine learning algorithms and a 5 fold time series split cross validation. The resultant models are assessed using Mean Absolute Error, Mean Absolute Percentage Error, Root Mean Squared Error and R2. Finally, the best model is used for forecasting deamnd up to 3 years. An interactive dashboard is also developed to enable user interaction with the model for forecasting, as well as provide the user with insights on the model features and visualisation of the historical data. <br>
![Alt text](./reports/figures/project_workflow.png)

Key Insights 
-------------
**Electricity Demand Analysis**:
- **Historical Transmission System Demand (TSD)**:
    - TSD exhibited a downward trend indicating reduced electricity demand over time except for the last 4 years (2020 to 2024) in which it weakened into a steady demand. The average tsd was 32.6GW with a 7.7GW variation in the last 15 years. <br>This is attributed to to factors such as energy efficiency improvements, economic changes with reduced energy-intensive industries like manufacturing, social behavioural changes with increase awareness of energy cost and climate impact and changes in work pattern especially due to Covid 19, shift in energy mix with more use of natural gas over electricity for heating, and renewable energy integration with the use of home solar panels. 
    - Hourly TSD showed that the highest demands occurred from 7am to 9pm, which aligns with the period most daily activities occurred. Nevertheless, the period outside 7am to 9pm exhibited outliers, indicating the sparse days in which there was more demand than usual.
    - Yearly TSD showed that the highest demands occurred in the colder months between September and April, while the lowest demands occurred in the summer months when energy for heating was less demanded. 
    - The demand wave height (from highest to lowest demand in a year) reduced over time, with shorter wave heights observed in the last 4 to 5 years than before, indicating a reduced annual demand over time.
- **Solar and Wind Generation Contributions**
    - Combined solar and wind generation contributed to over 20% of TSD from 2015 onwards, indicating the beginning of considerable contribution to UK electricity demand. In 2024, combined solar and wind generation contributions peaked at 71% of TSD.
    - Wind pprovide significant contribution from 2009 (100%) to 2015 (58.2%) comapred to solar. However wind and solar contributions within 40% to 60% from 2015 onwards with both having the same contributing in 2024. 
- **Model Performance**:
    - Initial Model (Temporal + 1,2,3 and 5-year lags):
      - MAE: ~2423.96MW, MAPE: 8.55%, R^2: 79%
    - Improved Model (Rolling 1 year mean + 1 hour, 1 day, 1 week, 1 year and 2 year lags):
      - MAE: 908.66MW, MAPE: 3.22%, R^2: 96%
- **Insights**:
    - Including both short-term (1 hour, 1 day, 1 week) and long-term (1 year, 2 year) lagged features in addition to the 1 year rolling mean feature improved model accuracy while reducing the risk of over-fitting.
    - Using a 30 minute lag (1 datapoint) increased training complexity and posed risks of overfitting.   

