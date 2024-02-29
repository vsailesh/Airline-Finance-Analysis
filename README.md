# Airline-Finance-Analysis

By examining regional and airline developments and engaging in critical discussions on significant industry factors, airline finance assesses financial patterns and the long-term prospects of the airline sector. The primary variable in our dataset is total revenue. In this study, various time series models were employed to predict future revenue. Graphs were plotted, and the performance errors of all models were compared to identify the model with the minimum error. Despite our initial hypothesis favoring ARIMA due to its reputed prediction capability in airline finance, the analysis revealed that SARIMA outperformed other models by yielding the least errors.

**A.   Engineering features for time-series models**

![image](https://github.com/vsailesh/Airline-Finance-Analysis/assets/93115567/4bab25a7-a70f-4772-9949-051d39a3d219)


The behavior of the raw data is shown in the first graph. Despite a decline after the first quarter of 2020, it has an increasing tendency overall.

### **STATIONARITY CHECK**

**ACF (Autocorrelation function):** Autocorrelation implies that the series is dependent this occurs when a time series is highly correlated with the lagged version the longer the bars in the ACF plot the more dependent the series.

![image](https://github.com/vsailesh/Airline-Finance-Analysis/assets/93115567/9c638d43-66ab-4368-864c-1f0446d0412d)

**PACF (Partial Auto correlation Function):** ACF test is best method for Moving Average. However, this ACF test does not work well with the Auto regression model for this model we use the PACF test to determine the relationship between the current value of it time series and the lagged values controlling for other correlations.

![image](https://github.com/vsailesh/Airline-Finance-Analysis/assets/93115567/7291dfca-bd07-4394-8edc-4b739877a239)

****

### **RESULTS**

                                    **Performance Measurement of Models**

| Model Name | MAE(Mean Absolute Error) | MSE (Mean Squared Error) | RMSE (Root Mean Squared Error) |
| --- | --- | --- | --- |
| MA | 2.031937e+06 | 9.532595e+12 | 3.087490e+06 |
| AR | 2.035881e+06 | 9.518188e+12 | 3.085156e+06 |
| ARMA | 1.942005e+06 | 9.325340e+12 | 3.053742e+06 |
| ARIMA | 1.948824e+06 | 9.210013e+12 | 3.034800e+06 |
| SARIMA | 1.884872e+06 | 9.152446e+12 | 3.025301e+06 |

 **Table Above :** The fundamental goal of comparing these errors is to determine which model has less errors; the fewer errors, the better the model.

![image](https://github.com/vsailesh/Airline-Finance-Analysis/assets/93115567/8d8a9353-5173-49f6-8ff5-6852db9c3c6e)
![image](https://github.com/vsailesh/Airline-Finance-Analysis/assets/93115567/abaa5fbc-7207-47fc-90f6-8da4c9d1e9d0)

### Conclusion :

We analyzed the quarterly revenue data of the airline by differencing it to eliminate seasonality and trend, creating a stationary time series resembling white noise. ACFs and PACFs of the seasonally adjusted time series were examined to determine parameter values for p, q, P, and Q. Given the nature of time-series data with a trend, we found SARIMA (Seasonal Autoregressive Integrated Moving Average) to be highly effective. The SARIMA model outperformed others in forecasting from 2016 to the present, as it adeptly handles seasonal time series data, providing superior accuracy compared to univariate models. This improved accuracy contributes to better decision-making for airline industries, enhancing revenue predictions with greater precision.
