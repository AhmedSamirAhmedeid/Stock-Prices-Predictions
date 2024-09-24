import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
from datetime import timedelta
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import ta
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import joblib

st.set_page_config(
    page_title="Stocks Prices Predictions",
    layout="wide")
# تحميل النموذج المحفوظ
model_lr = joblib.load('model_lr.pkl')

# صورة على الموقع
st.image('Stock_Yahoo_Finance.png', use_column_width=True)

# عنوان التطبيق
st.title('Stocks Prices Predictions')

st.header('About the Stock Market')
st.write('A stock market is a financial market where stocks and bonds are traded among investors. The stock market acts as an intermediary that allows companies to raise capital by issuing stocks and bonds, and for investors to buy and sell these instruments for a profit. Investing in the stock market is one of the most popular ways to grow wealth over the long term, but it also involves risks due to market volatility')

df = pd.read_csv('Sctok_Yahoo_Finance_New.csv')

# قم بإنشاء قائمة من الخيارات (التكير)
ticker_options = df['ticker'].unique().tolist()


col1, col2 = st.columns(2)

# إنشاء المخططات
with col1:
    st.subheader('Select Your Ticker')
    # قائمة منسدلة لاختيار الرمز (ticker)
    selected_ticker = st.selectbox('Select Ticker:', df['ticker'].unique(),key='seleted_ticker')
    # تصفية البيانات بناءً على الرمز المختار
    filtered_data = df[df['ticker'] == selected_ticker]

    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        subplot_titles=('Candlestick', 'SMA/EMA', 'MACD', 'RSI', 'ADX'),
        vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=filtered_data['date'],
        open=filtered_data['open'],
        high=filtered_data['high'],
        low=filtered_data['low'],
        close=filtered_data['close'],
        name=f'{selected_ticker} Candlestick'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['SMA_5'],
        mode='lines',
        name='SMA 5',
        line=dict(color='orange', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['EMA_5'],
        mode='lines',
        name='EMA 5',
        line=dict(color='yellow', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['EMA_12'],
        mode='lines',
        name='EMA 12',
        line=dict(color='purple', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['EMA_26'],
        mode='lines',
        name='EMA 26',
        line=dict(color='blue', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='cyan', width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['Signal_Line'],
        mode='lines',
        name='Signal Line',
        line=dict(color='red', width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['RSI_14'],
        mode='lines',
        name='RSI 14',
        line=dict(color='green', width=2)), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['ADX'],
        mode='lines',
        name='ADX',
        line=dict(color='lime', width=2)), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['+DI'],
        mode='lines',
        name='+DI',
        line=dict(color='orange', width=2)), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['-DI'],
        mode='lines',
        name='-DI',
        line=dict(color='red', width=2)), row=4, col=1)

    fig.update_layout(
        title=f'Indicators for {selected_ticker} Over Time',
        xaxis_title='Date',
        yaxis_title='Price',
        font=dict(size=14, color='white', family='Arial'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        width=1500,height=450)

    fig.update_yaxes(title_text='Price', row=1, col=1, color='white')
    fig.update_yaxes(title_text='MACD', row=2, col=1, color='white')
    fig.update_yaxes(title_text='RSI', row=3, col=1, color='white')
    fig.update_yaxes(title_text='ADX', row=4, col=1, color='white')

# عرض المخطط في Streamlit
    st.plotly_chart(fig)

    quarterly_data = filtered_data.groupby(['year', 'quarter']).agg({'close': 'mean','volume': 'sum','SMA_5': 'mean',
    'EMA_5': 'mean','EMA_12': 'mean','EMA_26': 'mean','RSI_14': 'mean',
    'BB_Middle': 'mean','BB_Upper': 'mean','BB_Lower': 'mean','MACD': 'mean',
    'Signal_Line': 'mean','ADX': 'mean','+DI': 'mean','-DI': 'mean'}).reset_index()


    years = quarterly_data['year'].unique()

    columns = ['close', 'SMA_5', 'EMA_5', 'EMA_12', 'EMA_26', 'RSI_14', 
       'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal_Line', 'ADX', '+DI', '-DI']

    colors = {'close': 'white','SMA_5': 'yellow','EMA_5': 'orange','EMA_12': 'purple',
      'EMA_26': 'blue','RSI_14': 'green','BB_Middle': 'red','BB_Upper': 'cyan',
      'BB_Lower': 'magenta','MACD': 'cyan','Signal_Line': 'red','ADX': 'lime',
      '+DI': 'orange','-DI': 'red'}

    st.subheader("Quarterly Data Viewer")

    selected_year = st.selectbox("Select Year:", years, key="year_select")
    selected_column = st.selectbox("Select Column:", columns, key="column_select")

    # دالة لتحديث الرسم البياني
    def update_graph(selected_year, selected_column):
        filtered_data = quarterly_data[quarterly_data['year'] == selected_year]
        
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=filtered_data['quarter'], 
            y=filtered_data[selected_column], 
            mode='lines+markers',
            name=selected_column,
            line=dict(color=colors[selected_column], width=2)))
    
        fig.update_layout(
            title=f'{selected_column} Trend for {selected_ticker} in {selected_year}',
            xaxis_title='Quarter',
            yaxis_title=selected_column,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(showgrid=False, gridcolor='gray'),
            yaxis=dict(showgrid=True, gridcolor='gray'),
            font=dict(size=14, color='white', family='Arial'))
    
        st.plotly_chart(fig)

    update_graph(selected_year, selected_column)

    weekly_data = filtered_data.groupby(['year', 'week']).agg({
        'close': 'mean', 'volume': 'sum', 'SMA_5': 'mean', 'EMA_5': 'mean',
        'EMA_12': 'mean', 'EMA_26': 'mean', 'RSI_14': 'mean', 'BB_Middle': 'mean',
        'BB_Upper': 'mean', 'BB_Lower': 'mean', 'MACD': 'mean', 'Signal_Line': 'mean',
        'ADX': 'mean', '+DI': 'mean', '-DI': 'mean'}).reset_index()

    years = weekly_data['year'].unique()
    columns = ['close', 'SMA_5', 'EMA_5', 'EMA_12', 'EMA_26', 'RSI_14', 
       'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal_Line', 'ADX', '+DI', '-DI']

    colors = {'close': 'white', 'SMA_5': 'yellow', 'EMA_5': 'orange', 'EMA_12': 'purple',
      'EMA_26': 'blue', 'RSI_14': 'green', 'BB_Middle': 'red', 'BB_Upper': 'cyan',
      'BB_Lower': 'magenta', 'MACD': 'cyan', 'Signal_Line': 'red', 'ADX': 'lime',
      '+DI': 'orange', '-DI': 'red'}

    def update_graph(selected_year, selected_column):
        filtered_data = weekly_data[weekly_data['year'] == selected_year]
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=filtered_data['week'].astype(str),
            y=filtered_data[selected_column],
            mode='lines+markers',
            name=selected_column,
            line=dict(color=colors[selected_column], width=2)))
    
        fig.update_layout(
            title=f'{selected_column} Trend for {selected_ticker} in {selected_year}',
            xaxis_title='Week',
            yaxis_title=selected_column,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(
                showgrid=False,
                gridcolor='gray',
                tickmode='linear',
                tickvals=filtered_data['week'].tolist(),
                ticktext=[str(int(week)) for week in filtered_data['week'].tolist()]),
            yaxis=dict(showgrid=True, gridcolor='gray'),
            height=550,
            font=dict(size=14, color='white', family='Arial'))
    
        fig.update_traces(text=[f'{val:.2f}' for val in filtered_data[selected_column]],
                          textposition='top right',
                          showlegend=False)
    
        st.plotly_chart(fig)

    def display_data(selected_year):
        filtered_data = weekly_data[weekly_data['year'] == selected_year]
        st.write(filtered_data)

    # Streamlit interface
    st.subheader("Weekly Data Visualization")
    
    selected_year = st.selectbox('Select Year:', years, key ="Year_select")
    selected_column = st.selectbox('Select Column:', columns, key ="Column_select")
    
    update_graph(selected_year, selected_column)
    


with col2:
    yearly_data = filtered_data.groupby('year').agg({'close': 'mean','volume': 'sum','SMA_5': 'mean',
        'EMA_5': 'mean','EMA_12': 'mean','EMA_26': 'mean','RSI_14': 'mean','BB_Middle': 'mean',
        'BB_Upper': 'mean','BB_Lower': 'mean','MACD': 'mean','Signal_Line': 'mean',
        'ADX': 'mean','+DI': 'mean','-DI': 'mean'}).reset_index()
    
    columns = ['close', 'SMA_5', 'EMA_5', 'EMA_12', 'EMA_26', 'RSI_14', 
               'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal_Line', 'ADX', '+DI', '-DI']
    
    
    colors = {'close': 'white','SMA_5': 'yellow','EMA_5': 'orange','EMA_12': 'purple',
              'EMA_26': 'blue','RSI_14': 'green','BB_Middle': 'red','BB_Upper': 'cyan',
              'BB_Lower': 'magenta','MACD': 'cyan','Signal_Line': 'red','ADX': 'lime',
              '+DI': 'orange','-DI': 'red'}
    
    st.subheader('Yearly Data Viewer')
    # واجهة المستخدم: اختيار العمود
    selected_column = st.selectbox('Select Column:', columns, index=columns.index('close'),key="column1_select")
    # رسم المخطط بناءً على العمود المختار
    def update_graph(selected_column):
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=yearly_data['year'], 
            y=yearly_data[selected_column], 
            mode='lines+markers',
            name=selected_column,
            line=dict(color=colors[selected_column], width=2)))
    
        fig.update_layout(
            title=f'{selected_column} Trend For {selected_ticker} in all Years',
            xaxis_title='Year',
            yaxis_title=selected_column,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(
                showgrid=False, 
                gridcolor='gray',
                tickmode='linear',  
                tickvals=yearly_data['year'].tolist(),  
                ticktext=[str(year) for year in yearly_data['year'].tolist()]  ),
            yaxis=dict(showgrid=True, gridcolor='gray'),
            font=dict(size=14, color='white', family='Arial'))
    
        fig.update_traces(text=[f'{val:.2f}' for val in yearly_data[selected_column]], 
                          textposition='top right', 
                          showlegend=False)
    
        st.plotly_chart(fig)

    update_graph(selected_column)

    monthly_data = filtered_data.groupby(['year', 'month']).agg({
        'close': 'mean', 'volume': 'sum', 'SMA_5': 'mean', 'EMA_5': 'mean',
        'EMA_12': 'mean', 'EMA_26': 'mean', 'RSI_14': 'mean', 'BB_Middle': 'mean',
        'BB_Upper': 'mean', 'BB_Lower': 'mean', 'MACD': 'mean', 'Signal_Line': 'mean',
        'ADX': 'mean', '+DI': 'mean', '-DI': 'mean'}).reset_index()

    years = monthly_data['year'].unique()
    columns = ['close', 'SMA_5', 'EMA_5', 'EMA_12', 'EMA_26', 'RSI_14', 'BB_Middle',
       'BB_Upper', 'BB_Lower', 'MACD', 'Signal_Line', 'ADX', '+DI', '-DI']

    colors = {'close': 'white', 'SMA_5': 'yellow', 'EMA_5': 'orange', 'EMA_12': 'purple',
      'EMA_26': 'blue', 'RSI_14': 'green', 'BB_Middle': 'red', 'BB_Upper': 'cyan',
      'BB_Lower': 'magenta', 'MACD': 'cyan', 'Signal_Line': 'red', 'ADX': 'lime',
      '+DI': 'orange', '-DI': 'red'}

    def update_graph(selected_year, selected_column):
        filtered_data = monthly_data[monthly_data['year'] == selected_year]
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=filtered_data['month'].astype(str),
            y=filtered_data[selected_column],
            mode='lines+markers',
            name=selected_column,
            line=dict(color=colors[selected_column], width=2)))
    
        fig.update_layout(
            title=f'{selected_column} Trend for {selected_ticker} in {selected_year}',
            xaxis_title='Month',
            yaxis_title=selected_column,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(
                showgrid=False,
                gridcolor='gray',
                tickmode='linear',
                tickvals=filtered_data['month'].tolist(),
                ticktext=[str(int(month)) for month in filtered_data['month'].tolist()]),
            yaxis=dict(showgrid=True, gridcolor='gray'),
            font=dict(size=14, color='white', family='Arial'))
    
        fig.update_traces(text=[f'{val:.2f}' for val in filtered_data[selected_column]],
                          textposition='top right',
                          showlegend=False)
    
        st.plotly_chart(fig)

    def display_data(selected_year):
        filtered_data = monthly_data[monthly_data['year'] == selected_year]
        st.write(filtered_data)

    # Streamlit interface
    st.subheader("Monthly Data Viewer")
    selected_year = st.selectbox('Select Year:', years, key="years_select")
    selected_column = st.selectbox('Select Column:', columns,key="columns_select")
    
    update_graph(selected_year, selected_column)

    daily_data = filtered_data.groupby(['year', 'month', 'day_name']).agg({
        'close': 'mean', 'volume': 'sum', 'SMA_5': 'mean', 'EMA_5': 'mean',
        'EMA_12': 'mean', 'EMA_26': 'mean', 'RSI_14': 'mean', 'BB_Middle': 'mean',
        'BB_Upper': 'mean', 'BB_Lower': 'mean', 'MACD': 'mean', 'Signal_Line': 'mean',
        'ADX': 'mean', '+DI': 'mean', '-DI': 'mean'}).reset_index()

    years = daily_data['year'].unique()
    months = daily_data['month'].unique()
    columns = ['close', 'SMA_5', 'EMA_5', 'EMA_12', 'EMA_26', 'RSI_14',
       'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal_Line', 'ADX', '+DI', '-DI']

    colors = {'close': 'white', 'SMA_5': 'yellow', 'EMA_5': 'orange', 'EMA_12': 'purple', 'EMA_26': 'blue',
      'RSI_14': 'green', 'BB_Middle': 'red', 'BB_Upper': 'cyan', 'BB_Lower': 'magenta',
      'MACD': 'cyan', 'Signal_Line': 'red', 'ADX': 'lime', '+DI': 'orange', '-DI': 'red'}

    def update_graph(selected_year, selected_month, selected_column):
        filtered_data = daily_data[(daily_data['year'] == selected_year) & (daily_data['month'] == selected_month)]
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=filtered_data['day_name'],
            y=filtered_data[selected_column],
            mode='lines+markers',
            name=selected_column,
            line=dict(color=colors[selected_column], width=2)))
    
        fig.update_layout(
            title=f'{selected_column} Trend for {selected_ticker} in {selected_year}-{selected_month}',
            xaxis_title='Day of the Week',
            yaxis_title=selected_column,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(
                showgrid=False,
                gridcolor='gray',
                tickmode='linear',
                tickvals=filtered_data['day_name'].tolist(),
                ticktext=filtered_data['day_name'].tolist()
            ),
            yaxis=dict(showgrid=True, gridcolor='gray'),
            font=dict(size=14, color='white', family='Arial')
        )
    
        fig.update_traces(text=[f'{val:.2f}' for val in filtered_data[selected_column]],
                          textposition='top right',
                          showlegend=False)
    
        st.plotly_chart(fig)

    def display_data(selected_year, selected_month):
        filtered_data = daily_data[(daily_data['year'] == selected_year) & (daily_data['month'] == selected_month)]
        st.write(filtered_data)

    # Streamlit interface
    st.subheader("Daily Data Visualization")
    
    selected_year = st.selectbox('Select Year:', years,key = 'Years_select')
    selected_month = st.selectbox('Select Month:', months, key = 'Months_select')
    selected_column = st.selectbox('Select Column:', columns, key = 'Columns_select')
    
    update_graph(selected_year, selected_month, selected_column)

st.subheader('Prediction Close Price ')
# إدخال البيانات من المستخدم
Open = st.number_input('Enter Open Price:')
SMA_5 = st.number_input('Enter SMA_5 Indicator:')
EMA_5 = st.number_input('Enter EMA_5 Indicator:')
EMA_12 = st.number_input('Enter EMA_12 Indicator:')
EMA_26 = st.number_input('Enter EMA_26 Indicator:')
RSI_14 = st.number_input('Enter Trade RSI_14 Indicator:')


if st.button('Prediction'):
    # التوقع باستخدام النموذج
    prediction_close = model_lr.predict([[Open, SMA_5, EMA_5, EMA_12, EMA_26, RSI_14]])  # أدخل الميزات هنا
    st.write(f'Prediction Close Price: {prediction_close[0]}')

    # تحليل الاتجاه بناءً على التوقع
    def analyze_trend(filtered_data, predicted_close):
        trend = filtered_data['analysis_trend'].iloc[-1]  # استخدام آخر قيمة في عمود analysis_trend
        current_close = filtered_data['close'].iloc[-1]
    
        if predicted_close > current_close:
            return f"Uptrend (Uptrend) - {trend}"
        elif predicted_close < current_close:
            return f"Downtrend (Downtrend) - {trend}"
        else:
            return f"Sideways (Sidetrend) - {trend}"

    trend_analysis = analyze_trend(filtered_data, prediction_close[0])

    # عرض التوقع وتحليل الاتجاه
    st.subheader('Predict the closing price for the next day')
    
    st.subheader('trend analysis')
    st.write(f"Expected trend: {trend_analysis}")


