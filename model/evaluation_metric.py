import os
import sys
sys.path.append(os.path.abspath('../.'))

import numpy as np



from model.parser import parse, create
from model.parser import map_day, map_month, map_leap_year








def evaluate(path):

    valid_count   = 0
    invalid_count = 0


    data       = parse(path)
    predict_df = create(path)


    for i in range(len(data)):

        if data.day.iloc[i] != predict_df.day.iloc[i]:
            invalid_count += 1
            continue

        elif data.month.iloc[i] != predict_df.month.iloc[i]:
            invalid_count += 1
            continue

        elif data.leap_year.iloc[i] != predict_df.leap_year.iloc[i]:
            invalid_count += 1
            continue

        elif data.decade.iloc[i] != predict_df.decade.iloc[i]:
            invalid_count += 1
            continue

        else:
            valid_count += 1



    yHat = predict_df[['day', 'month', 'leap_year', 'decade']].values.astype(int).T
    y    = data[['day', 'month', 'leap_year', 'decade']].values.astype(int).T

    print(yHat.dtype)

    # print(yHat.shape)
    # print(yHat[0].shape)
    print(yHat[0])
    # print(yHat[0, :].shape)
    # print(yHat[0, :].reshape(1, -1).shape)


    # print(yHat[0, :].reshape(1, -1))
    # print(y[0, :].reshape(1, -1))

    # a = yHat[0, :].reshape(1, -1)
    # b = y[0, :].reshape(1, -1)
    #
    # print(np.corrcoef(a, b))


    # corr = np.corrcoef(yHat.T, y.T)

    corr_day       = np.corrcoef(yHat[0, :].reshape(1, -1), y[0, :].reshape(1, -1))[1, 0]
    corr_month     = np.corrcoef(yHat[1, :].reshape(1, -1), y[1, :].reshape(1, -1))[1, 0]
    corr_leap_year = np.corrcoef(yHat[2, :].reshape(1, -1), y[2, :].reshape(1, -1))[1, 0]
    corr_decade    = np.corrcoef(yHat[3, :].reshape(1, -1), y[3, :].reshape(1, -1))[1, 0]

    print(corr_day)
    print(corr_month)
    print(corr_leap_year)
    print(corr_decade)





    return {'valid_counts': valid_count, 'invalid_counts': invalid_count}








if __name__ == '__main__':
    # print(evaluate('../data/data.txt'))
    # print(evaluate('../data/output_file.txt'))
    print(evaluate('../data/output_file_smote.txt'))







