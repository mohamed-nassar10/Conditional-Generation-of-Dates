import os
import sys
sys.path.append(os.path.abspath('../.'))

from typing import Tuple, Dict

import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


from model.parser import parse, create







def eval_score(yHat: np.array, y: np.array) -> Tuple[float, float, float, float, float] :

    R2 = r2_score(y, yHat)

    MAE = mean_absolute_error(y, yHat)

    MSE = mean_squared_error(y, yHat)

    MSLE = mean_squared_log_error(y, yHat)

    RMSE = np.sqrt(mean_squared_error(y, yHat))


    return R2, MAE, MSE, MSLE, RMSE




def evaluate(path: str) -> Dict[str, int]:

    valid_count   = 0
    invalid_count = 0

    valid_day       = 0
    valid_month     = 0
    valid_leap_year = 0
    valid_decade    = 0


    data       = parse(path)     # parse the real features

    predict_df = create(path)    # get day, month, leap_year and decade features from
                                 # predicted date for evaluation metrics

    data_len = len(data)
    data_str = len(str(data_len))


    for i in range(len(data)):

        flag_ls = []
        flag_ls.append(data.day.iloc[i] != predict_df.day.iloc[i])
        flag_ls.append(data.month.iloc[i] != predict_df.month.iloc[i])
        flag_ls.append(data.leap_year.iloc[i] != predict_df.leap_year.iloc[i])
        flag_ls.append(data.decade.iloc[i] != predict_df.decade.iloc[i])


        valid_day       += (0 if flag_ls[0] else 1)
        valid_month     += (0 if flag_ls[1] else 1)
        valid_leap_year += (0 if flag_ls[2] else 1)
        valid_decade    += (0 if flag_ls[3] else 1)


        if any(flag_ls):
            invalid_count += 1
        else:
            valid_count += 1




    yHat = predict_df[['day', 'month', 'leap_year', 'decade']].values.astype(int).T
    y    = data[['day', 'month', 'leap_year', 'decade']].values.astype(int).T


    day_yHat, day_y             = yHat[0, :].reshape(1, -1), y[0, :].reshape(1, -1)
    month_yHat, month_y         = yHat[1, :].reshape(1, -1), y[1, :].reshape(1, -1)
    leap_year_yHat, leap_year_y = yHat[2, :].reshape(1, -1), y[2, :].reshape(1, -1)
    decade_yHat, decade_y       = yHat[3, :].reshape(1, -1), y[3, :].reshape(1, -1)


    score_day       = eval_score(day_yHat.flatten(), day_y.flatten())
    score_month     = eval_score(month_yHat.flatten(), month_y.flatten())
    score_leap_year = eval_score(leap_year_yHat.flatten(), leap_year_y.flatten())
    score_decade    = eval_score(decade_yHat.flatten(), decade_y.flatten())


    print()

    print(f'score_day:'.ljust(16), f'R2=({score_day[0]:0.4f}), MAE=({score_day[1]:0.4f}), '
                                   f'MSE=({score_day[2]:0.4f}), MSLE=({score_day[3]:0.4f}), '
                                   f'RMSE=({score_day[4]:0.4f})')

    print(f'score_month:'.ljust(16), f'R2=({score_month[0]:0.4f}), MAE=({score_month[1]:0.4f}),'
                                     f'MSE=({score_month[2]:0.4f}), MSLE=({score_month[3]: 0.4f}),' 
                                     f'RMSE=({score_month[4]:0.4f})')

    print(f'score_leap_year:'.ljust(16), f'R2=({score_leap_year[0]:0.4f}), MAE=({score_leap_year[1]:0.4f}), '
                                         f'MSE=({score_leap_year[2]:0.4f}), MSLE=({score_leap_year[3]:0.4f}), '
                                         f'RMSE=({score_leap_year[4]:0.4f})')

    print(f'score_decade:'.ljust(16), f'R2=({score_decade[0]:0.4f}), MAE=({score_decade[1]:0.4f}), '
                                      f'MSE=({score_decade[2]:0.4f}), MSLE=({score_decade[3]: 0.4f}), '
                                      f'RMSE=({score_decade[4]:0.4f})')
    print()


    print(f'cov_day:'.ljust(14), f'{np.cov(np.array([day_yHat, day_y]).squeeze())[0, 0]:0.4f}')
    print(f'cov_month:'.ljust(14), f'{np.cov(np.array([month_yHat, month_y]).squeeze())[0, 0]:0.4f}')
    print(f'cov_leap_year:'.ljust(14), f'{np.cov(np.array([leap_year_yHat, leap_year_y]).squeeze())[0, 0]:0.4f}')
    print(f'cov_decade:'.ljust(14), f'{np.cov(np.array([decade_yHat, decade_y]).squeeze())[0, 0]:0.4f}')
    print()


    print(f'corr_day:'.ljust(15), f'{np.corrcoef(day_yHat, day_y)[1, 0]:0.4f}')
    print(f'corr_month:'.ljust(15), f'{np.corrcoef(month_yHat, month_y)[1, 0]:0.4f}')
    print(f'corr_leap_year:'.ljust(15), f'{np.corrcoef(leap_year_yHat, leap_year_y)[1, 0]:0.4f}')
    print(f'corr_decade:'.ljust(15), f'{np.corrcoef(decade_yHat, decade_y)[1, 0]:0.4f}')
    print()


    print(f'valid_day:'.ljust(16), f'{valid_day},'.ljust(data_str+1),
          f'invalid_day:'.ljust(22), f'{data_len - valid_day}'.ljust(data_str))

    print(f'valid_month:'.ljust(16), f'{valid_month},'.ljust(data_str+1),
          f'invalid_month:'.ljust(22), f'{data_len - valid_month}'.ljust(data_str))

    print(f'valid_leap_year:'.ljust(16), f'{valid_leap_year},'.ljust(data_str+1),
          f'invalid_leap_year:'.ljust(22), f'{data_len - valid_leap_year}'.ljust(data_str))

    print(f'valid_decade:'.ljust(16), f'{valid_decade},'.ljust(data_str+1),
          f'invalid_decade:'.ljust(22), f'{data_len - valid_decade}'.ljust(data_str))

    print(f'-'.ljust(16+22+data_str+data_str+5, '-'))
    print(f'Dataset length:'.ljust(16+22+data_str+3), f'{data_len}'.ljust(data_str))

    print()



    return {'valid_counts': valid_count, 'invalid_counts': invalid_count}








if __name__ == '__main__':
    # print(evaluate('../data/data.txt'))
    print(evaluate('../data/output_file.txt'), '\n\n')
    print(evaluate('../data/output_file_smote.txt'), '\n\n')














