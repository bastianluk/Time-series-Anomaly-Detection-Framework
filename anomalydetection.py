import locale
import time
import datetime as d
import random
import math
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd

import matplotlib
font = {'weight' : 'bold', 'size'   : 15}
matplotlib.rc('font', **font)
from matplotlib import pyplot as plt

from scipy.stats import mode as statmode
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from timedataframe import TimeDataFrame

class AD:

    def __init__(self, timeseries, freq, season='WEEKLY', default_wvs=100000, filename='scores.csv'):

        self.freq = freq
        self.default_wvs= default_wvs
        self.season = season
        self.filename = filename
        self.timeseries = timeseries
        self.df = timeseries.to_frame().reset_index()
        self.df.columns = ['time', 'value']

        self.preprocess()
        self._exec()

    def preprocess(self):
        self.df['diff'] = self.df['value'].diff()
        self.df['wvs'] = 0
        self.df['qd'] = 0        
        self.df['q1'] = 0
        self.df['q3'] = 0
        self.df['diff_q1'] = 0
        self.df['diff_q3'] = 0
        self.df['value'].fillna(-1, inplace=True)

    def add_day_column(self, row):
        locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
        return d.datetime.strptime(row['time'], '%d.%m.%Y %H:%M').date().strftime("%A").lower()

    def _exec(self):
        self.df['day'] = self.df[['time']].apply(self.add_day_column, axis=1)
        if self.season == 'WEEKLY':
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            all_days = []
            for day in days:
                day = day.lower()
                day_df = self.df[self.df['day'] == day].reset_index()
                day_df.columns = ['oi', 'time', 'value', 'diff', 'wvs', 'qd', 'q1', 'q3', 'diff_q1', 'diff_q3', 'day']
                day_df = self._qd(day_df)
                day_df = self._diff_qd(day_df)
                all_days.append(day_df)
            
            self.new_df = pd.concat(all_days)
            self.new_df = self.new_df.sort_values('oi').set_index('oi').reset_index(drop=True)
            self._adjust_diff_qd()

        elif self.season == 'DAILY':
            # DAILY seasonality score calculation goes here
            df = self._qd(self.df)
            df = self._diff_qd(df)
            self.new_df = df
            self._adjust_diff_qd()

        elif self.season == 'HOURLY':
            # HOURLY seasonality score calculation goes here
            pass
        
        elif self.season == 'MONTHLY':
            # MONTHLY seasonality score calculation goes here
            pass

        self._wvs()
        self.write()

    def add_interval_column(self, row): 
        return str(row['time']).split(" ")[1]

    def _wvs(self):
        self.new_df['wvs'] = self.new_df.apply(self._set_wv, axis=1)
    
    def _set_wv(self, row):
        wv = 0
        if row['value'] < 0:
            wv = self.default_wvs
            
        return wv
        
    def _qd(self, day_df):
        day_df['interval'] = day_df[['time']].apply(self.add_interval_column, axis=1)
        intervals = day_df['interval'].unique().tolist()
        all_intervals = []
        for interval in intervals:
            interval_df = day_df[day_df['interval'] == interval]
            q1, q3 = np.percentile(interval_df[interval_df['value'] > -1]['value'].dropna().values, [25, 75])
            interval_df['q1'] = q1
            interval_df['q3'] = q3 
            interval_df['qd'] = interval_df.apply(self._set_qd, axis=1)
            all_intervals.append(interval_df)

        day_df = pd.concat(all_intervals)
        return day_df.sort_index()

    def _set_qd(self, row):
        qd = 0
        if row['value'] > row['q3']:
            qd = row['value'] - row['q3']
        elif row['value'] < 0:
            qd = 0
        elif row['value'] < row['q1']:
            qd = row['q1'] - row['value']
        return qd

    def _diff_qd(self, day_df):
        day_df['interval'] = day_df[['time']].apply(self.add_interval_column, axis=1)
        intervals = day_df['interval'].unique().tolist()
        all_intervals = []
        for interval in intervals:
            interval_df = day_df[day_df['interval'] == interval]
            q1, q3 = np.percentile(interval_df['diff'].dropna().values, [25, 75])
            interval_df['diff_q1'] = q1
            interval_df['diff_q3'] = q3 
            interval_df['diff_qd'] = interval_df.apply(self._set_diff_qd, axis=1)
            all_intervals.append(interval_df)

        day_df = pd.concat(all_intervals)
        return day_df.sort_index()

    def _adjust_diff_qd(self):
        first = self.new_df['diff_qd'][0]
        self.new_df['diff_qd'] = self.new_df['diff_qd'].diff()
        self.new_df['diff_qd'][0] = first

    def _set_diff_qd(self, row):
        qd = 0
        if row['diff'] > row['diff_q3']:
            qd = row['diff'] - row['diff_q3']
        elif row['diff'] < row['diff_q1']:
            qd = row['diff_q1'] - row['diff']
        return qd

    def get_updated_df(self):
        an_df = self.new_df.reset_index()
        an_df.columns = ['oi','time','value','diff','wvs','qd','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']
        return an_df

    def write(self):
        self.new_df.to_csv(self.filename)
