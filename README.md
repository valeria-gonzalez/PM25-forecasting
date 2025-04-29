# PM2.5 Daily Prediction in Tlaquepaque, Jalisco

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![LSTM](https://img.shields.io/badge/Model-LSTM-green.svg) ![Time Series](https://img.shields.io/badge/Forecasting-Time_Series-important.svg)

## Table of Contents
- [Project Objective](#project-objective)
- [Motivation](#motivation)
- [Dataset Description](#dataset-description)
- [Proposed solution](#proposed-solution)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Project Objective

The objective of this project is to predict daily PM2.5 pollutant concentration levels in Tlaquepaque, Jalisco using a Long Short-Term Memory (LSTM) neural network model. By leveraging historical atmospheric data, the goal is to create a reliable tool capable of forecasting air quality trends and helping to anticipate periods of poor air quality.

### What you'll find

In order to make this project a little more comprehensible, it's presented in the form of three jupyter notebooks that detail the steps taken to create this project, which are:

- **Data Preprocessing:** Cleaning and combining multiple years of atmospheric data into a final training dataset.

- **Feature Engineering:** Selecting relevant meteorological features to predict PM2.5 concentrations and interpolating missing data.

- **Reframe Dataset:** Transforming the cleaned time series data into a supervised learning format, where input sequences of past time steps are used to predict future values of PM2.5.

- **Select and Train LSTM:** Designing and training a Long Short-Term Memory (LSTM) neural network capable of learning patterns from the multivariate dataset to forecast PM2.5. Optimal hyperparameter search methods were applied to improve the model's performance

- **Forecasting and Results Analysis:** Using the trained LSTM model to forecast PM2.5 levels, analyzing the model's performance, and visualizing the predictions.

Graphs of the process and the datasets created can also be found.

## Motivation

Air pollution is a significant public health concern in the Área Metropolitana de Guadalajara (AMG), with PM10 and ozone (O3) often exceeding safe limits. Exposure to particulate matter, especially during the winter, is associated with severe health risks such as lung cancer, pneumonia, and exacerbation of asthma. Predicting pollutant levels allows for timely preventive measures to protect public health.

### Why PM2.5
While PM10 is the dominant pollutant, the historical data available for PM10 is incomplete and inconsistent. However, PM2.5 which is strongly correlated with PM10 in this area, has a more complete dataset, making it a better candidate for predictive modeling.

## Dataset Description

The data used for this project was sourced from the [Secretaría de Medio Ambiente y Desarrollo Territorial de Jalisco (SEMADET)](https://aire.jalisco.gob.mx/), which provides historical hourly measurements of atmospheric and meteorological variables across different years.

However, the datasets required heavy preprocessing due to inconsistencies, missing data, and varying formats. After extensive cleaning, interpolation, and reframing, three years of data were combined into a final training dataset suitable for time series forecasting.

### Dataset Parameters

The following parameters were present in the raw datasets:

| Parameter                      | Description                                                         | Units          |
|:--------------------------------|:---------------------------------------------------------------------|:---------------|
| Fecha                           | Date of measurement                                                  | dd/mm/yyyy     |
| Hora                            | Time of measurement                                                  | HH:mm          |
| ID del sitio                    | Site identification code                                             | Alphanumeric   |
| Ozono (O3)                      | Ground-level ozone concentration                                     | ppm            |
| Bióxido de Azufre (SO2)         | Sulfur dioxide concentration                                         | ppm            |
| Óxido de Nitrógeno (NO)         | Nitric oxide concentration                                           | ppm            |
| Bióxido de Nitrógeno (NO2)      | Nitrogen dioxide concentration                                       | ppm            |
| Óxidos de Nitrógeno (NOX)       | Total nitrogen oxides concentration                                  | ppm            |
| Monóxido de Carbono (CO)        | Carbon monoxide concentration                                        | ppm            |
| PM10                            | Particulate matter ≤10 micrometers                                   | µg/m³          |
| PM2.5                           | Particulate matter ≤2.5 micrometers                                  | µg/m³          |
| Temperatura Interna (TMPI)      | Internal temperature                                                 | °C             |
| Temperatura Externa (TMP)       | External (ambient) temperature                                       | °C             |
| Humedad Relativa (RH)           | Relative humidity                                                    | %              |
| Presión Barométrica (PBA)       | Barometric pressure                                                  | mbar           |
| Velocidad del Viento (WS)       | Wind speed                                                           | m/s            |
| Dirección del Viento (WD)       | Wind direction                                                       | degrees        |
| Precipitación (PP)              | Precipitation                                                        | mm             |
| Radiación Solar (RS)            | Solar radiation                                                      | W/m²           |
| Índice UV (UVI)                 | Ultraviolet radiation index                                          | Index          |

## Proposed solution

To forecast PM2.5 concentrations, a **multivariate LSTM model** was developed.

### Why LSTM?

The Long Short-Term Memory (LSTM) network is especially powerful for time series forecasting tasks. It can learn long-term dependencies in sequential data, allowing it to model the temporal patterns crucial for pollutant prediction. Unlike traditional feedforward networks, LSTM can retain context from previous time steps, making it ideal for air quality forecasting where atmospheric conditions evolve over time.

### Features Used for Training:

The following features were selected based on their consistency across the dataset and their ability to provide meaningful context to PM2.5 predictions. During dataset exploration, it was found that using PM2.5 as a univariate feature led to lower model performance. Including these additional meteorological variables improved predictive accuracy and model stability.

- **Exogenous Variables (Contextual Features):**
  - Temperature (TMP)
  - Relative Humidity (RH)
  - Barometric Pressure (PBA)
  - Wind Speed (WS)
  - Wind Direction (WD)

- **Target Pollutant (Feature to Predict):**
  - Fine Particulate Matter (PM2.5)

## Limitations

One of the primary limitations was the inconsistent and incomplete nature of the original data, which required substantial cleaning and interpolation. Additionally, while LSTM is effective for sequence modeling, it has limited capacity to handle certain types of noise and non-linearity that may exist in real-world air quality data. Exploring more advanced architectures could potentially yield better performance.

## Future Work

To enhance model accuracy and robustness, future iterations could explore transfer learning to adapt the model for other pollutants and ultimately predict the Air Quality Index (AQI). Testing alternative machine learning models may also improve generalization.
