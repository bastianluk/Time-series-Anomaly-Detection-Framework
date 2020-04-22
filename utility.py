import locale
import time
import datetime as d


def add_day_column(row):
    locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
    return d.datetime.strptime(row['time'], '%d.%m.%Y %H:%M').date().strftime("%A").lower()

def add_interval_column(row): 
    return str(row['time']).split(" ")[1]

def add_hour_column(row): 
    return int(str(row['interval']).split(":")[0])

def add_minute_column(row): 
    return str(row['interval']).split(":")[1]

def add_date_column(row): 
    return str(row['time']).split(" ")[0]

def add_day_of_month_column(row):
    return int(str(row['date']).split(".")[0])

def add_month_index_column(row):
    return str(row['date']).split(".")[1]


