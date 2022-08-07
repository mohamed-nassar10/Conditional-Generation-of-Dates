# import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None


import datetime
# from dateutil.relativedelta import relativedelta




map_day = {'[SAT]': int(0),
           '[SUN]': int(1),
           '[MON]': int(2),
           '[TUE]': int(3),
           '[WED]': int(4),
           '[THU]': int(5),
           '[FRI]': int(6)}


map_month = {'[JAN]': int(0),
             '[FEB]': int(1),
             '[MAR]': int(2),
             '[APR]': int(3),
             '[MAY]': int(4),
             '[JUN]': int(5),
             '[JUL]': int(6),
             '[AUG]': int(7),
             '[SEP]': int(8),
             '[OCT]': int(9),
             '[NOV]': int(10),
             '[DEC]': int(11)}


map_leap_year = {'[False]': int(0), '[True]': int(1)}




def parse(path: str) -> pd.DataFrame :

    data = pd.read_csv(path, sep=' ', header=None, encoding_errors='ignore')

    data.columns = ['day', 'month', 'leap_year', 'decade', 'date']

    # data.date = pd.to_datetime(data.date, format='%d-%m-%Y', infer_datetime_format=False, errors='ignore')
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y') if type(x)==str else np.NaN)



    data.day       = data.day.map(map_day)

    data.month     = data.month.map(map_month)

    data.leap_year = data.leap_year.map(map_leap_year)

    data.decade    = data.decade.apply(lambda x: x.strip('[]')).astype('int64')



    return data


# parse('../data/data.txt')
# parse('../data/output_file.txt')
# parse('../data/output_file_smote.txt')



def create(path: str) -> pd.DataFrame :

    data = pd.read_csv(path, sep=' ', header=None, encoding_errors='ignore')

    data.columns = ['day', 'month', 'leap_year', 'decade', 'date']

    # data.date = pd.to_datetime(data.date, format='%d-%m-%Y', infer_datetime_format=False)
    # data.date = pd.to_datetime(data.date, format='%d-%m-%Y', infer_datetime_format=False, errors='ignore')
    # data.date = pd.to_datetime(data.date, format='%d-%m-%Y', infer_datetime_format=False, errors='coerce')
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y') if type(x)==str else np.NaN)


    for i in range(len(data)):

        # data.day.iloc[i]       = map_day[f'[{data.date.iloc[i].day_name()[:3].upper()}]']
        data.day.iloc[i]       = map_day[f'[{data.date.iloc[i].strftime("%a").upper()}]']

        # # data.month.iloc[i]     = map_month[f'[{data.date.iloc[i].month_name()[:3].upper()}]']
        data.month.iloc[i]     = map_month[f'[{data.date.iloc[i].strftime("%b").upper()}]']


        # # data.leap_year.iloc[i] = map_leap_year[f'[{data.date.iloc[i].is_leap_year}]']
        flag = ((data.date.iloc[i].year % 100 != 0) and (data.date.iloc[i].year % 4 == 0)) \
                 or (data.date.iloc[i].year % 400 == 0)

        data.leap_year.iloc[i] = map_leap_year[f'[{flag}]']

        # data.decade.iloc[i]    = (data.date.iloc[i].year // 10)
        data.decade.iloc[i]    = (data.date.iloc[i].year // 10)




    return data




# create('../data/output_file_smote.txt')







































