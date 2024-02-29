import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange
from pandas import Series
from matplotlib import pyplot
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import altair as alt
from altair import Chart, X,Y, Axis, SortField, OpacityValue


def interactive_plot(df):
   y_axis_val=st.selectbox("Select Y-Axis Value", options=df.columns)
   
   plot = px.line(df, y=y_axis_val)
   plot.show()
   st.plotly_chart(plot)
# Add a title and intro text
st.title('Airline Operating Revenue Prediction & Forecast')
st.text('This is a web app to allow exploration of Airline Operating Revenue Data')
# Create file uploader object
upload_file = st.file_uploader('Upload a file containing Operating Revenue data')
# Check to see if a file has been uploaded
if upload_file is not None:
   # If it has then do the following:
   # Read the file to a dataframe using pandas
   df = pd.read_csv(upload_file)
   # Create a section for the dataframe statistics
   #st.write(df)
   #df = pd.read_csv("Operating_Rev.csv")
   df.columns = df.iloc[0]
   df = df.drop(index = 0)
   df = df.dropna()
   df = df.loc[df['Quarter'] != "TOTAL"]
   df = df.reset_index()
   df = df.drop(['index'],axis = 1)
   #df = df.drop(['index','LATIN AMERICA','ATLANTIC','PACIFIC'],axis = 1)
   df = df.replace(['1','2','3','4'],['3', '6','9','12'])
   df['Year'] = df['Year'] + '-'+ df['Quarter']
   df['DOMESTIC'] = df['DOMESTIC'].str.replace(',','')
   df['INTERNATIONAL'] = df['INTERNATIONAL'].str.replace(',','')
   df['PACIFIC'] = df['PACIFIC'].str.replace(',','')
   df['LATIN AMERICA'] = df['LATIN AMERICA'].str.replace(',','')
   df['ATLANTIC'] = df['ATLANTIC'].str.replace(',','')
   df['TOTAL'] = df['TOTAL'].str.replace(',','')
   cols = df.columns
   df[cols] = df[cols].apply(pd.to_numeric,  errors='ignore')
   df['Year'] = pd.to_datetime(df['Year'])
   #df['Year'] = pd.to_datetime(df['Year']).dt.to_period('M')
   
   
   #st.header('Statistics of Dataframe')
   #st.write(df6.describe())
   
   # Create a section for the dataframe header
   
        
   
   st.header(' Dataframe')
   st.write(df)
   fig = px.line(df, x="Year", y=['LATIN AMERICA','ATLANTIC','PACIFIC','DOMESTIC','INTERNATIONAL','TOTAL'], title="RAW DATA") 
   st.plotly_chart(fig)

   df['Year'] = pd.to_datetime(df['Year']).dt.to_period('M')
   df = df.set_index('Year').resample('M').interpolate()

   st.write('After Interpolation')
   st.write(df.drop(columns = 'Quarter').to_timestamp())
   #df = df.sort_values(by="x")
   #fig = px.line(df, x="x", y="y", title="Sorted Input") 
   #fig.show()
   



   #st.plotly_chart(fig, use_container_width=True)



   # Create a section for matplotlib figure
   option = st.selectbox(
    'Select Data : ',
    ('Select a Option','LATIN AMERICA','ATLANTIC','PACIFIC','DOMESTIC','INTERNATIONAL','TOTAL'))

   st.write('You selected:', option) 
  
   if (option!= "Select a Option"):
   
      df=df[option]  
   else:
      st.header(" Data Not Selected ")
      st.stop()
      

   df = df.to_timestamp()

   #y_axis_val=st.selectbox("Select Y-Axis Value", options=df.columns)
   st.header('Plot')

   plot = px.line(df)
   
   st.plotly_chart(plot)
   #interactive_plot(df)
#### Plotting Trend, seasonality, cyclicality, and noise graph of the the data of Total Revenue 
   st.header('Seasonality Graphs ')
   result = seasonal_decompose(df, model='additive', period=12)
   pl=result.plot()
   st.pyplot(pl)
   #st.write('- In the first graph, we see the behavior of the raw data.it has a upward trend despite a drop at the end of the first quarter of 2020. - The second plot exhibits the trend of the data. Trend shows the overall movement of a time series.- Seasonality is the third plot which shows periodical ups and downs in the data. - The last graph show the residuals.After eliminating the trend and seasonal components from the time series, this was generated. If the residual is steady in the end, Then our data has a stationary structure and is prepared to go on to the modeling phase. but my data in residuals at end is not stedy so in next step we need to do the stationarity check and differencing')

   
   
   #fig = px.line(df, y="TOTAL", title='Raw Data')



   df_diff = df.diff()
   df_diff = df_diff.dropna()

   st.header('Stationary Data ')

   fig, ax = plt.subplots(figsize = (10 , 7))
   df_diff.plot(ax = ax)
   ax.set_xlabel('Year')
   ax.set_ylabel('Frequency')

   st.pyplot(fig)


   with st.expander('See Explanation'):
         st.write(''' 
         - Inorder to make the data stationary we need to differentiate the data 
         - we can differeniate the data n times until the p value in ADF is below 0.05
                  ''')

   tab1, tab2, tab3  = st.tabs(["ADF ", "ACF ","PACF"])
   

   with tab1:
      st.header("Stationarity Check")
      st.write("### ADF (Augmented Dickey Fuller)")
      from statsmodels.tsa.stattools import adfuller
      #print('Results of Dickey Fuller Test:')
      dftest = adfuller(df, autolag='AIC')

      dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
      for key,value in dftest[4].items():
         dfoutput['Critical Value (%s)'%key] = value
         
      st.write(dfoutput)

      
      #stat_test = adfuller(df)
      #st.write('The statistic value is {} and p-value is {}'.format(stat_test[0], stat_test[1]))
      #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

      st.write("### ADF after Differencing")
      dftest = adfuller(df_diff, autolag='AIC')

      dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
      for key,value in dftest[4].items():
         dfoutput['Critical Value (%s)'%key] = value
         
      st.write(dfoutput)
      #from statsmodels.tsa.stattools import adfuller
      #stat_test = adfuller(df_diff)
      #st.write('The statistic value is {} and p-value is {}'.format(stat_test[0], stat_test[1]))
      

      

      

   with tab2:

      #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
      st.header("Stationarity Check ")
      st.write('### ACF (Auto Correlation Function)')
      import statsmodels.api as sm
      fig, ax = plt.subplots(figsize = (10 , 7))
      sm.graphics.tsa.plot_acf(df, lags=20,ax =ax)
      st.pyplot(fig)

      st.write('### ACF after Differencing')
      fig, ax = plt.subplots(figsize = (10 , 7))
      sm.graphics.tsa.plot_acf(df_diff, lags=20,ax =ax)
      st.pyplot(fig)

      

      

   with tab3:
      st.header("Stationary Check")
      st.write('### PACF Partial Auto Correlation Function')
      fig, ax = plt.subplots(figsize = (10 , 7))
      sm.graphics.tsa.plot_pacf(df, lags=10,ax =ax);
      st.pyplot(fig)

      st.write('### PACF after Differencing')
      fig, ax = plt.subplots(figsize = (10 , 7))
      sm.graphics.tsa.plot_pacf(df_diff, lags=10,ax =ax)
      st.pyplot(fig)


   
   

   tab1, tab2, tab3, tab4, tab5, tab6  = st.tabs(["Moving Average", "AutoRegressive Model","ARMA", "ARIMA","SARIMA", "Performance Measurment"])

   with tab1:
      st.header("Moving Average")
      #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
      df_diff = df.diff().dropna()
      from statsmodels.tsa.arima.model import ARIMA
      from sklearn.metrics import mean_absolute_error, mean_squared_error
      train_len = int(len(df_diff) * 0.75)
      dtrain = df_diff[:train_len]
      dtest  = df_diff[train_len:]
      start = len(dtrain)
      end = len(dtrain) + len(dtest) - 1
      ma_model = ARIMA(dtrain, order=(0, 0, 6))
      ma_result = ma_model.fit()
      ma_pred = ma_result.predict(start, end)
      ma_pred.index = dtest.index
      fig, ax = plt.subplots(figsize = (10 , 7))
      plt.plot(dtrain.index,dtrain, label = 'Train')
      plt.plot(dtest.index,dtest, label = 'Test')
      plt.plot(ma_pred.index,ma_pred, label = 'Prediction')
      plt.title("MA Predictions")
      plt.legend()
      plt.xlabel('Date')
      plt.ylabel('Frequency')
      plt.show()
      st.pyplot(fig)
      
      mae_MA = mean_absolute_error(dtest, ma_pred)
      mse_MA = mean_squared_error(dtest, ma_pred)
      rmse_MA = np.sqrt(mean_squared_error(dtest, ma_pred))
     #st.write("#### Predictive Performance of \nMAE = {}  \nMSE = {} \nRMSE ={} ".format(mae_MA ,mse_MA, rmse_MA))
      ma_perf = {'mae_MA': mae_MA,
             'mse_MA':mse_MA,
             'rmse_MA':rmse_MA}
      ma_perf = pd.DataFrame([ma_perf])
      ma_perf
      ma_model = ARIMA(df_diff, order=(0, 0, 6))
      ma_result = ma_model.fit()
      forecast_index = pd.date_range(dtest.index[-1]+ pd.DateOffset(months=1), periods=24 ,freq = 'MS')
      ma_forecast = ma_result.forecast(steps=24)
      ma_forecast.index = forecast_index
      fig, ax = plt.subplots(figsize = (10 , 7))
      plt.plot(df_diff.index, df_diff, label='Actual')
      plt.plot(ma_forecast.index, ma_forecast, label='Forecast')
      plt.legend()
      plt.title('MA Forecasts')
      plt.xlabel('Dates')
      plt.show()
      st.write("### Forecast")
      st.pyplot(fig)

      



   with tab2:
      st.header("AutoRegressive Model")
      #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
      train_lenarm = int(len(df_diff) * 0.75)
      dtrain_arm = df_diff[:train_lenarm]
      dtest_arm  = df_diff[train_lenarm:]
      start = len(dtrain_arm)
      end = len(dtrain_arm) + len(dtest_arm) - 1
      arm_model = ARIMA(dtrain_arm, order=(3, 0, 0))
      arm_result = arm_model.fit()
      arm_pred = arm_result.predict(start, end)
      arm_pred.index = dtest_arm.index
      fig, ax = plt.subplots(figsize = (10 , 7))
      plt.plot(dtrain_arm.index,dtrain_arm, label = 'Train')
      plt.plot(dtest_arm.index,dtest_arm, label = 'Test')
      plt.plot(arm_pred.index,arm_pred, label = 'Prediction')
      plt.title("ARM Predictions")
      plt.legend()
      plt.xlabel('Date')
      plt.ylabel('Frequency')
      plt.show()
      st.write('### Prediction')
      st.pyplot(fig)

      forecast_indexarm = pd.date_range(dtest_arm.index[-1], periods=24)
      arm_forecast = arm_result.forecast(steps=24)
      arm_forecast.index = forecast_indexarm
      fig, ax = plt.subplots(figsize = (10 , 7))
      plt.plot(df_diff.index, df_diff, label='Actual')
      plt.plot(arm_forecast.index, arm_forecast, label='Forecast')
      plt.legend()
      plt.title('ARM Forecasts')
      plt.xlabel('Dates')
      plt.show()
      st.write('### Forecast')
      st.pyplot(fig)

      mae_arm = mean_absolute_error(dtest_arm, arm_pred)
      mse_arm = mean_squared_error(dtest_arm, arm_pred)
      rmse_arm = np.sqrt(mean_squared_error(dtest_arm, arm_pred))
      #st.write("#### Predictive Performance of Autoregressive model is \nMAE = {}  \nMSE = {} \nRMSE ={} ".format(mae_arm ,mse_arm, rmse_arm))
      ar_perf = {'mae': mae_arm,
             'mse':mse_arm,
             'rmse':rmse_arm}
      ar_perf = pd.DataFrame([ar_perf])
      ar_perf
     






   




   
   with tab3:
      st.header("ARMA")
      #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
      sm.graphics.tsa.plot_acf(df_diff, lags=20)
      sm.graphics.tsa.plot_pacf(df_diff, lags=10);
      train_len = int(len(df_diff) * 0.75)
      dtrain = df_diff[:train_len]
      dtest  = df_diff[train_len:]
      start = len(dtrain)
      end = len(dtrain) + len(dtest) - 1

      # input values
      arma_model = ARIMA(dtrain, order=(3, 1, 6))
      arma_results = arma_model.fit()
      arma_pred = arma_results.predict(start, end)
      arma_pred.index = dtest.index
      fig=plt.figure(figsize=(20,10))
      plt.plot(dtrain, label='Train')
      plt.plot(dtest, label='Test')
      plt.plot(arma_pred, label='ARMA Predictions')
      plt.title('ARMA predictions')
      plt.xlabel('Date')
      plt.legend()
      st.write('Prediction')
      st.pyplot(fig)


      mae_arma = mean_absolute_error(dtest, arma_pred)
      mse_arma = mean_squared_error(dtest, arma_pred)
      rmse_arma = np.sqrt(mean_squared_error(dtest, arma_pred))

      arma_perf = {'mae_arma':mae_arma,
            'mse_arma':mse_arma,
            'rmse_arma':rmse_arma}
      arma_perf = pd.DataFrame([arma_perf])
      arma_perf

      arma_model = ARIMA(df_diff, order=(3, 1, 6)).fit()
      arma_forecast = arma_model.forecast(steps=24)
      arma_forecast.index = forecast_index
      fig=plt.figure(figsize=(20,10))
      plt.plot(df_diff, label='Actual')
      plt.plot(arma_forecast, label='Forecast')
      plt.title('ARMA Forecast')
      plt.xlabel('Date')
      plt.legend()
      plt.show()
      
      st.write('Forecast')
      st.pyplot(fig)





   with tab4:
      st.header("ARIMA")
      #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
      #sm.graphics.tsa.plot_acf(df_diff, lags=25)
      #sm.graphics.tsa.plot_pacf(df_diff, lags=20)
      train_len = int(len(df) * 0.75)
      train = df[:train_len]
      test  = df[train_len:]
      arima_model = ARIMA(train, order=(3,1,6))
      arima_results = arima_model.fit()
      arima_predict = arima_results.predict(start, end)
      arima_predict.index = test.index
      fig=plt.figure(figsize=(20,10))
      plt.plot(train, label='Train')
      plt.plot(test, label='Test')
      plt.plot(arima_predict, label='Predictions')
      plt.title('ARIMA Predictions')
      plt.legend()
      plt.xlabel('Year')
      plt.show()
      
      st.write('Prediction')
      st.pyplot(fig)
      arima_pred_diff = arima_predict.diff().dropna()
      mae_arima = mean_absolute_error(dtest.iloc[1:], arima_pred_diff)
      mse_arima = mean_squared_error(dtest.iloc[1:], arima_pred_diff)
      rmse_arima = np.sqrt(mean_squared_error(dtest.iloc[1:], arima_pred_diff))
      arima_perf = {'mae_arima': mae_arima,
             'mse_arima':mse_arima,
             'rmse_arima':rmse_arima}
      arima_perf = pd.DataFrame([arima_perf])
      st.write(arima_perf)
      forecast_index = pd.date_range(test.index[-1]+ pd.DateOffset(months=1), periods=24 ,freq = 'MS')
      arima_model = ARIMA(df, order=(3, 1, 6)).fit()
      arima_forecast = arima_model.forecast(steps=24)
      arima_forecast.index = forecast_index
      fig=plt.figure(figsize=(20,10))
      plt.plot(df, label='Actual')
      plt.plot(arima_forecast, label='Forecast')
      plt.title('ARIMA Forecast')
      plt.xlabel('Year')
      plt.legend()
      plt.show()
      
      st.write('Forecast')
      st.pyplot(fig)      






   with tab5:
      st.header("SARIMA")
      #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
      train_len = int(len(df) * 0.75)
      train = df[:train_len]
      test  = df[train_len:]
      from statsmodels.tsa.statespace.sarimax import SARIMAX
      sarima_model = SARIMAX(train, order=(3, 1, 6), seasonal_order=(2 , 1 , 2 , 12))
      sarima_results = sarima_model.fit(disp=0)
      sarima_pred = sarima_results.predict(start, end)
      sarima_pred.index = test.index
      fig=plt.figure(figsize=(20,10))
      plt.plot(train, label='Train')
      plt.plot(test, label='Test')
      plt.plot(sarima_pred, label='Prediction')
      plt.legend()
      plt.title('SARIMA Predictions')
      plt.xlabel('Date')
      plt.show()
      
      st.write('Prediction')
      st.pyplot(fig)

      sarima_pred_diff = sarima_pred.diff().dropna()
      mae_sarima = mean_absolute_error(dtest.iloc[1:], sarima_pred_diff)
      mse_sarima = mean_squared_error(dtest.iloc[1:], sarima_pred_diff)
      rmse_sarima = np.sqrt(mean_squared_error(dtest.iloc[1:], sarima_pred_diff))
      sarima_perf = {'mae_sarima': mae_sarima,
             'mse_sarima':mse_sarima,
             'rmse_sarima':rmse_sarima}
      sarima_perf = pd.DataFrame([sarima_perf])
      st.write(sarima_perf)
      #number = st.number_input('Insert a number')
      forecast_index = pd.date_range(test.index[-1]+ pd.DateOffset(months=1), periods=12 ,freq = 'MS')
      sarima_model = SARIMAX(df, order=(3, 1, 6), seasonal_order=(2, 1, 2 , 12))
      sarima_results = sarima_model.fit(disp=0)
      sarima_forecast = sarima_results.forecast(steps=12)
      sarima_forecast.index = forecast_index
      fig=plt.figure(figsize=(20,10))
      plt.plot(df, label='Actual')
      plt.plot(sarima_forecast, label='Forecast')
      plt.title('SARIMA Forecast')
      plt.legend()
      plt.xlabel('Date')
      plt.show()
      
      st.write('Forecast')
      st.pyplot(fig)
   with tab6:
      
      st.header('Performance Measurment')
      basic_perf = pd.concat([ma_perf.T, ar_perf.T, arma_perf.T, arima_perf.T, sarima_perf.T])
      basic_perf = basic_perf.rename(columns={0:'results'})
      col1, col2, col3 = st.columns(3)

      with col1:
         st.write('Mean Absolute Error')
         basic_mae = basic_perf[basic_perf.index.str.contains('mae')] # Moving  average error
         st.write(basic_mae.sort_values(by=['results'], ascending=True))

      with col2:
         st.write(' Mean Squared Error')
         basic_mse = basic_perf[basic_perf.index.str.contains('^mse')] # mean square error
         st.write(basic_mse.sort_values(by=['results'], ascending=True))

      with col3:
         st.write('Root Mean Squared Error')
         basic_rmse = basic_perf[basic_perf.index.str.contains('rmse')] # rooot Mean Square error
         st.write(basic_rmse.sort_values(by=['results'], ascending=True))
   
   
   



      

   
   #mul_side= st.sidebar.multiselect("Select a column ", df.columns)
   #plt.plot(df[''])

   #fig, ax = plt.subplots(1,1)
   #ax.scatter(x=df['Depth'], y=df['Magnitude'])
   #ax.set_xlabel('Depth')
   #ax.set_ylabel('Magnitude')
   #st.pyplot(fig)


   