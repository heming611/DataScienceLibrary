import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthBegin, MonthEnd, DateOffset, YearEnd, YearBegin

from datetime import date, datetime, timedelta
import ishbook, time, calendar, os, getpass, re, sys, holidays, plotly, psycopg2, sqlalchemy, logging
logging.disable(sys.maxsize)

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Javascript
from plotly.tools import FigureFactory as FF
import plotly.offline 
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True) # run at the start of every notebook to use plotly
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

pd.set_option('display.float_format', lambda x: '%.3f' % x)

#plotly theme
import plotly.io as pio
#pio.templates.default = "none"


def add_holidays(train_df, training_start_date, training_end_date, country):
    '''
    Return a holiday df with country holidays and monthends
    '''
    
    country_holiday = eval('holidays.%s()'%(country))

    time_rng = pd.date_range(training_start_date, training_end_date)
    holidays_df = pd.DataFrame(index=time_rng)
    holidays_df.reset_index(inplace=True)
    holidays_df.rename(columns={'index':'day'}, inplace=True)

    holidays_df['country_holiday'] = holidays_df.apply(lambda x: x.day in country_holiday, axis=1)

    # Prophet requires the holiday df to be in a ceratin format
    holidays_fb = pd.DataFrame({
      'holiday': 'national',
      'ds': holidays_df.loc[holidays_df.country_holiday==True, 'day'],
    })

    # we add month ends as there is revenue dip on month ends 
    month_ends = pd.Series(pd.date_range(train_df.ds.min(), train_df.ds.max(), freq='M'))
    month_ends = month_ends.apply(lambda x: str(x.date()))

    # add christmas eve
    others = pd.DataFrame()
    other_holidays = pd.Series(['-12-24','-12-25'])
    for year in range(train_df.ds.min().year, train_df.ds.max().year,1): 
        tmp = other_holidays.apply(lambda x: str(year)+x)
        others = pd.concat([others, tmp],0)

    holiday_list_new = list(pd.concat([month_ends, others]).iloc[:,0].sort_values())

    holidays_new = pd.DataFrame(holiday_list_new)
    holidays_new.columns = ['ds']
    holidays_new['holiday'] = 'national'
    
    holidays_new['ds'] = pd.to_datetime(holidays_new['ds'])

    holidays_new.loc[holidays_new.ds.isin(month_ends), 'holiday'] = 'month_end'
    
    holidays_new['upper_window'] = 0
    holidays_new['lower_window'] = -1
#     holidays_new['holidays_prior_scale'] = 10
    holidays_new.loc[(holidays_new.ds.dt.month == 12) & (holidays_new.ds.dt.day == 25), 'holiday'] = 'xmas'
    holidays_new.loc[(holidays_new.ds.dt.month == 12) & (holidays_new.ds.dt.day == 31), 'holiday'] = 'nye'
    holidays_new.loc[(holidays_new.ds.dt.month == 1) & (holidays_new.ds.dt.day == 1), 'holiday'] = 'ny'
    
         
    return pd.concat([holidays_fb, holidays_new], ignore_index=True)


def covid_drop_mark(ds):
    date = pd.to_datetime(ds)
    if date >= pd.Timestamp('2020-03-15'):
        return -1
    else:
        return 0
    
    
# def generate_model(training_data, holidays_df, periods = 100, add_regressor=True):
#     '''
#     Build the Prophet model 
#     Return the model and predicted values
#     '''
#     m = Prophet(yearly_seasonality=True, 
#                 weekly_seasonality=True, 
#                 seasonality_mode='multiplicative', 
#                 holidays = holidays_df,
#                 changepoint_range=1)
#     # build and train prophet model
#     if add_regressor:
#         training_data['covid_drop'] = training_data.ds.apply(covid_drop_mark)
#         m.add_regressor('covid_drop') 
#     m.fit(training_data)
#     # no. of days for which predictions need to be made
#     future = m.make_future_dataframe(periods = periods)
    
#     if add_regressor:
#         future['covid_drop'] = future.ds.apply(covid_drop_mark)

#     forecast = m.predict(future)
    
#     return m, forecast

def generate_model(training_data, holidays_df, periods = 100, add_regressor=True):
    '''
    Build the Prophet model 
    Return the model and predicted values
    '''
    # build and train prophet model
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', 
                holidays = holidays_df, changepoint_range=1)
    if add_regressor:
        training_data['covid_drop'] = training_data.ds.apply(covid_drop_mark)
        m.add_regressor('covid_drop')
    m.fit(training_data)
    # no. of days for which predictions need to be made
    future = m.make_future_dataframe(periods = periods)
    if add_regressor:
        future['covid_drop'] = future.ds.apply(covid_drop_mark)
    forecast = m.predict(future)
        
    return m, forecast


def cross_validate(m, initial, horizon='90 days', metric='mape', plot=True):
    '''
    Function to perform cross validation and visaulize the error over hold out days
    Returns a df with the error metric for each holdout day
    '''
    
    # The output of cross_validation is a df with the true values y and the out-of-sample forecast values yhat, 
    # at each simulated forecast date and for each cutoff date
    data_cv = cross_validation(m, initial=initial, horizon = horizon)
    rolling_window = 0.1
    
    # performance_metrics utility can be used to compute some useful statistics of the prediction performance
    df_none = performance_metrics(data_cv, metrics=[metric], rolling_window=0)
    df_h = performance_metrics(data_cv, metrics=[metric], rolling_window=rolling_window)

    tick_w = max(df_none['horizon'].astype('timedelta64[ns]')) / 10.

    dts = ['D', 'h', 'm', 's', 'ms', 'us', 'ns']
    dt_names = [
        'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
        'nanoseconds'
    ]
    dt_conversions = [
        24 * 60 * 60 * 10 ** 9,
        60 * 60 * 10 ** 9,
        60 * 10 ** 9,
        10 ** 9,
        10 ** 6,
        10 ** 3,
        1.,
    ]
    for i, dt in enumerate(dts):
        if np.timedelta64(1, dt) < np.timedelta64(tick_w, 'ns'):
            break

    # record the data to use visualize cross validation
    cv_x = (df_none['horizon'].astype('timedelta64[ns]').astype(np.int64) / float(dt_conversions[i])).tolist()
    cv_x_h = (df_h['horizon'].astype('timedelta64[ns]').astype(np.int64) / float(dt_conversions[i])).tolist()
    cv_y = (df_none[metric]).tolist()
    cv_y_h = (df_h[metric]).tolist()
    
    if plot:
        trace_cv_scatter = go.Scatter(
        x = cv_x,
        y = cv_y,
        line = dict(
            color = '#666666',
            width = 0.8),
        legendgroup = 'each CV',
        mode = 'markers', # use nodes for each cross validation
        name = 'Each validation',
        )

        trace_cv_line = go.Scatter(
            x = cv_x_h,
            y = cv_y_h,
            line = dict(
                color = '#2164F3',
                width=2),
            legendgroup = 'rolling CV',
            mode='lines',

            name='Rolling average',
        )

        data = [trace_cv_line, trace_cv_scatter]

        layout = go.Layout(
 #                   title = 'Cross Validation',
                    xaxis = dict(title = 'Horizon (Days)', rangemode="tozero"),
                    yaxis = dict(title = f'{metric}', rangemode="tozero"),

                )

        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)
        
    return df_h


def compare_pred(train_df, forecast, plot=True):
    '''
    Compare actuals vs Prophet predictions 
    Returns a df with actuals and the predictions
    '''
    compare_df = forecast.merge(train_df, how='outer', on='ds')
    
    if plot:
        max_y = compare_df.yhat_upper.max()
        n_digit = 10**(len(str(int(max_y)))-1)

        date_list = compare_df['ds'].astype('str').tolist()
        date_list_reverse = date_list[::-1]

        trace1 = go.Scatter(
            x=date_list + date_list_reverse,
            y=compare_df['yhat_upper'].tolist() + compare_df['yhat_lower'].tolist() [::-1],
            fill='tozerox',
            fillcolor='rgba(255,102,0,0.3)',
            line = dict(color = 'rgba(255,102,0,0.1)'),
            name='Prediction_Bounds',
        )

        trace2 = go.Scatter(
            x = date_list,
            y = compare_df['yhat'],
         line = dict(color = 'rgba(255,102,0,1)', dash='dashdot', width=1.5),
            mode='lines',
            name='Prediction',
        )
        trace3 = go.Scatter(
            x = date_list,
            y = compare_df['y'],
            line = dict(color = 'rgba(33,100,243,0.6)', width=1.5),
            mode='lines',
            name='Actual',
        )

        data = [trace1, trace2, trace3]

        layout = go.Layout(
#            title = 'Compare Actual with Prediction (%s)'%(country),
            xaxis = dict(title = 'Date'),
            yaxis = dict(title = '(US Dollars)'),
           annotations=[
            go.layout.Annotation(
                text="End of Training Data", 
                           x=pd.Timestamp(datetime.strftime(train_df[-1:]['ds'].tolist()[0], "%Y-%m-%d")), 
                           y=np.floor(max_y/(n_digit)) * n_digit
            )
        ]

        )
        layout.update(dict(shapes = [
            {
                'type': 'line',
                'x0': pd.Timestamp(datetime.strftime(train_df[-1:]['ds'].tolist()[0], "%Y-%m-%d")),
                'y0': 0,
                'x1': pd.Timestamp(datetime.strftime(train_df[-1:]['ds'].tolist()[0], "%Y-%m-%d")),
                'y1': np.ceil(max_y/(n_digit)) * n_digit,
                'line': {
                    'color': '#909090',
                    'width': 1,
                }
            }]
            ))

        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig,  filename='Compare Actual with Prediction')
    
    return compare_df


def plot_components(m, forecast, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0):
    '''  
    Will plot whichever are available of: trend, holidays, weekly seasonality, yearly seasonality, 
    and additive and multiplicative extra regressors. 
    Returns a df with all axes values for the corresponding plots
    Refer source code - https://github.com/facebook/prophet/blob/master/python/fbprophet/plot.py
    '''

    components = ['trend']

    if m.train_holiday_names is not None and 'holidays' in forecast:
            components.append('holidays')
        # Plot weekly seasonality, if present
    if 'weekly' in m.seasonalities and 'weekly' in forecast:
        components.append('weekly')
    # Yearly if present
    if 'yearly' in m.seasonalities and 'yearly' in forecast:
        components.append('yearly')
    # Other seasonalities
    components.extend([
        name for name in sorted(m.seasonalities)
        if name in forecast and name not in ['weekly', 'yearly']
    ])
    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in forecast:
            components.append('extra_regressors_{}'.format(mode))


    for plot_name in components:
        if plot_name == 'trend':
            pass
            plot_forecast_component(
                m=m, forecast=forecast, name='trend', uncertainty=uncertainty, plot_cap=plot_cap,
            )
        elif plot_name == 'weekly':
            plot_weekly(
                m=m, uncertainty=uncertainty, weekly_start=weekly_start,
            )
        elif plot_name == 'yearly':
            plot_yearly(
                m=m, uncertainty=uncertainty, yearly_start=yearly_start,
            )
        elif plot_name in [
            'holidays',
            'extra_regressors_additive',
            'extra_regressors_multiplicative',
        ]:
            plot_forecast_component(
                m=m, forecast=forecast, name=plot_name, uncertainty=uncertainty,plot_cap=False,
            )
        else:
            plot_seasonality(
                m=m, name=plot_name,  uncertainty=uncertainty,
            )


def plot_forecast_component(m, forecast, name, uncertainty=True, plot_cap=False):
    '''
    Get axes values to plot a particular component of the forecast
    '''
    fcst_t = forecast['ds'].dt.to_pydatetime()
    data = []

    trace1 = go.Scatter( x=fcst_t, y=forecast[name], line = dict(color = '#2164F3'), name=name,  showlegend = False,)
    data.append(trace1)

    if 'cap' in forecast and plot_cap:
        trace2 = go.Scatter( x=fcst_t, y=forecast['cap'], line = dict(dash = 'dash', color = 'rgb(0, 0, 0)'))
        data.append(trace2)
    if m.logistic_floor and 'floor' in forecast and plot_cap:
        trace3 = go.Scatter( x=fcst_t, y=forecast['floor'], line = dict(dash = 'dash', color = 'rgb(0, 0, 0)'))
        data.append(trace3)
    if uncertainty:
        date_list = forecast['ds'].dt.to_pydatetime().astype('str').tolist()
        date_list_reverse = date_list[::-1]

        trace4 = go.Scatter(
            x=date_list + date_list_reverse,
            y=forecast[name + '_upper'].tolist() + forecast[name + '_lower'].tolist() [::-1],
            fill='tozerox',
            fillcolor='#4CBBF3',
            name='u-l bound',
            showlegend = False,
            line = dict(color = '#2164F3')
        )

        data.append(trace4)

    if name in m.component_modes['multiplicative']:
        layout = go.Layout(yaxis=dict(title = name, tickformat='%'), xaxis=dict(title='ds'),width=700,height=500,)
    else:
        layout = go.Layout(xaxis=dict(title='ds'), yaxis=dict(title=name),width=700,height=500,)

    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig,  filename='try')
    

def seasonality_plot_df(m, ds):
    '''
    Prepare dataframe for plotting seasonal components
    '''
    df_dict = {'ds': ds, 'cap': 1., 'floor': 0.}
    for name in m.extra_regressors:
        df_dict[name] = 0.
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df


def plot_weekly(m, uncertainty=True, weekly_start=0):
    '''
    Get axes values to plot the weekly component of the forecast.
    '''
    uncertainty=True
    weekly_start=0

    data = []

    days = (pd.date_range(start='2017-01-01', periods=7) + pd.Timedelta(days=weekly_start))
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days = days.weekday_name

    trace1 = go.Scatter(x=list(range(len(days))), y=seas['weekly'].tolist(), line = dict(color = '#2164F3'), showlegend=False, name='weekly')
    data.append(trace1)

    if uncertainty:
        x_list = list(range(len(days)))
        x_list_reverse = x_list[::-1]

        trace2 = go.Scatter(
            x= x_list + x_list_reverse,
            y=seas['weekly_upper'].tolist() + seas['weekly_lower'].tolist() [::-1],
            fill='tozerox',
            fillcolor='#4CBBF3',
            showlegend=False,
            name='u-l bound',
            line = dict(color = '#2164F3')
        )

        data.append(trace2)



    if m.seasonalities['weekly']['mode'] == 'multiplicative':
        layout = go.Layout(width=700, height=500,
            yaxis=dict(title = 'weekly', tickformat='%'), 
            xaxis=dict(title = 'Day of week',
                                  ticktext=days, 
                                  tickvals=list(range(len(days)))))
    else:
        layout = go.Layout(xaxis=dict(title = 'Day of week',ticktext=days, tickvals=list(range(len(days)))), yaxis=dict(title='weekly'), width=700,height=500,)

    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig,  filename='try')
    
    

def plot_yearly(m, uncertainty=True, yearly_start=0):
    '''
    Get axes values to plot the yearly component of the forecast.
    '''
    uncertainty=True
    yearly_start=0

    data = []


    days = (pd.date_range(start='2017-01-01', periods=365) + pd.Timedelta(days=yearly_start))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    trace1 = go.Scatter(x=df_y['ds'].dt.to_pydatetime(), y=seas['yearly'], line = dict(color = '#2164F3'), name='yearly', showlegend=False)
    data.append(trace1)

    if uncertainty:
        date_list = df_y['ds'].dt.to_pydatetime().astype('str').tolist()
        date_list_reverse = date_list[::-1]

        trace2 = go.Scatter(
            x= date_list + date_list_reverse,
            y=seas['yearly_upper'].tolist() + seas['yearly_lower'].tolist() [::-1],
            fill='tozerox',
            fillcolor='#4CBBF3',
            showlegend=False,
            name='u-l bound',
            line = dict(color = '#2164F3')
        )

        data.append(trace2)


    if m.seasonalities['yearly']['mode'] == 'multiplicative':
        layout = go.Layout(yaxis=dict(title = 'yearly', tickformat='%'), xaxis=dict(title = 'Day of year'),width=700,
    height=500,)
    else:
        layout = go.Layout(yaxis=dict(title='yearly'),  xaxis=dict(title = 'Day of year'),width=700,
    height=500,)



    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig,  filename='try')

    
def plot_seasonality(m, name, uncertainty=True):
    '''
    Get axes values to plot a custom seasonal component 
    '''

    data = []

    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    plot_points = 200
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    trace1 = go.Scatter(x=df_y['ds'].dt.to_pydatetime(), y=seas[name], line = dict(color = '#2164F3'), showlegend=False, name=name)
    data.append(trace1)


    if uncertainty:
        date_list = df_y['ds'].dt.to_pydatetime().astype('str').tolist()
        date_list_reverse = date_list[::-1]

        trace2 = go.Scatter(
            x= date_list + date_list_reverse,
            y=seas[name + '_upper'].tolist() + seas[name + '_lower'].tolist() [::-1],
            fill='tozerox',
            fillcolor='#4CBBF3',
            showlegend=False,
            name='u-l bound',
            line = dict(color = '#2164F3')
        )

        data.append(trace2)

    if m.seasonalities[name]['mode'] == 'multiplicative':
        layout = go.Layout(yaxis=dict(title = name, tickformat='%'), xaxis=dict(title = 'ds'))
    else:
        layout = go.Layout(yaxis=dict(title=name),  xaxis=dict(title = 'ds'))

    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig,  filename='try')


