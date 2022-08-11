import os
import sys
sys.path.append(os.path.abspath('../.'))

import warnings

# warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn', append=True)



import argparse
import pathlib
from typing import Tuple


import numpy as np
import pandas as pd

import torch

from model.model_class import get_model


import datetime
from dateutil.relativedelta import relativedelta


from model.parser import map_day, map_month, map_leap_year





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def result_to_date(result: str) -> str :

    result = str(result)

    day   = int(result[6:]) - 1
    month = int(result[4:6]) - 1

    ss = str(f'{result[:4]}0101')

    ds = datetime.datetime.strptime(ss, '%Y%m%d')

    ds = ds + relativedelta(months=month, days=day)

    result = ds.strftime('%d-%m-%Y')


    return result




def predict(input_list: list) -> Tuple[str, str] :


    X       = np.zeros((1, 10))
    X_smote = np.zeros((1, 10))



    day       = np.array(map_day[input_list[0]])
    month     = np.array(map_month[input_list[1]])
    leap_year = map_leap_year[input_list[2]]
    decade    = np.array(input_list[3].strip('[]')).astype('int64')


    X[0, 0] = scaler_day.transform(day.reshape(-1, 1)).flatten()
    X[0, 1] = np.sin(day * (2 * np.pi / 7))
    X[0, 2] = np.cos(day * (2 * np.pi / 7))

    X[0, 3] = scaler_month.transform(month.reshape(-1, 1)).flatten()
    X[0, 4] = np.sin(month * (2 * np.pi / 12))
    X[0, 5] = np.cos(month * (2 * np.pi / 12))

    X[0, 6] = leap_year

    X[0, 7] = scaler_decade.transform(decade.reshape(-1, 1)).flatten()
    X[0, 8] = np.sin(decade * (2 * np.pi / alpha))
    X[0, 9] = np.cos(decade * (2 * np.pi / alpha))

    # ---

    X_smote[0, 0] = scaler_day_smote.transform(day.reshape(-1, 1)).flatten()
    X_smote[0, 1] = X[0, 1]
    X_smote[0, 2] = X[0, 2]

    X_smote[0, 3] = scaler_month_smote.transform(month.reshape(-1, 1)).flatten()
    X_smote[0, 4] = X[0, 4]
    X_smote[0, 5] = X[0, 5]

    X_smote[0, 6] = leap_year

    X_smote[0, 7] = scaler_decade_smote.transform(decade.reshape(-1, 1)).flatten()
    X_smote[0, 8] = np.sin(decade * (2 * np.pi / alpha_smote))
    X_smote[0, 9] = np.cos(decade * (2 * np.pi / alpha_smote))

    # ---

    with torch.no_grad():
        result       = model(torch.tensor(X).to(device).float())
        result_smote = model_smote(torch.tensor(X_smote).to(device).float())

    # ---

    result       = ''.join(result.detach().cpu().numpy().round().astype(int).astype(str).flatten())
    result_smote = ''.join(result_smote.detach().cpu().numpy().round().astype(int).astype(str).flatten())



    return result_to_date(result), result_to_date(result_smote)





# input_test1 = ['[MON]', '[DEC]', '[False]', '[196]']
# output_test1 = '3-12-1962'
#
# input_test2 = ['[THU]', '[DEC]', '[True]', '[204]']
# output_test2 = '3-12-2048'
#
# input_test3 = ['[WED]', '[JAN]', '[False]', '[181]']
# output_test3 = '10-1-1810'
#
#
# input_test4 = ['[WED]', '[JUN]', '[False]', '[209]']
#
# test_model(input_test1), test_model(input_test2), test_model(input_test3), test_model(input_test4)





parser = argparse.ArgumentParser(description='Run predict.py to predict a date')
parser.add_argument('-i', '--input-file', help='specify first name', type=str, required=False, dest='input')
parser.add_argument('-o', '--output-file', help='specify last name', type=str, required=False, dest='output')





def path_smote(path: str) -> str:

    p = pathlib.Path(path)

    return path.replace(p.stem, p.stem + '_smote')





if __name__ == '__main__':


    args = parser.parse_args()


    if args.input and args.output:
        input_path  = args.input
        output_path = args.output
        output_path_smote = path_smote(output_path)

    elif args.input:
        input_path = args.input
        output_path = '../data/output_file.txt'
        output_path_smote = path_smote(output_path)

    elif args.output:
        input_path = '../data/example_input.txt'
        output_path = args.output
        output_path_smote = path_smote(output_path)

    else:
        input_path = '../data/example_input.txt'
        output_path = '../data/output_file.txt'
        output_path_smote = path_smote(output_path)


    print(f'Here are paths:-')
    print(f'\tinput_path:'.ljust(18), f'{input_path}')
    print(f'\toutput_path:'.ljust(18), f'{output_path}')
    print(f'\toutput_path_smote:'.ljust(18), f'{output_path_smote}')





    while(True):

        ch = input(f'Do you want to proceed ? [y, n]: ')

        if ch not in ['y', 'Y', 'n', 'N']:
            continue


        elif ch in ['y', 'Y']:

            # checkpoint = torch.load('checkpoint.pth.tar')
            checkpoint = torch.load('checkpoint.pth.tar', map_location=torch.device('cpu'))

            scaler_day = checkpoint['scaler_day']
            scaler_month = checkpoint['scaler_month']
            scaler_decade = checkpoint['scaler_decade']

            scaler_day_smote = checkpoint['scaler_day_smote']
            scaler_month_smote = checkpoint['scaler_month_smote']
            scaler_decade_smote = checkpoint['scaler_decade_smote']

            alpha       = checkpoint['alpha']
            alpha_smote = checkpoint['alpha_smote']

            model = get_model()
            model_smote = get_model()

            model.eval()
            model_smote.eval()

            model.load_state_dict(checkpoint['model_state_dict'])
            model_smote.load_state_dict(checkpoint['model_smote_state_dict'])


            data = pd.read_csv(input_path, sep=' ', header=None)
            data.columns = ['day', 'month', 'leap_year', 'decade']

            R       = pd.DataFrame(np.nan, index=np.arange(len(data)), columns=['result'])
            R_smote = pd.DataFrame(np.nan, index=np.arange(len(data)), columns=['result_smote'])

            for i in range(len(data)):
                R.iloc[i], R_smote.iloc[i] = predict(data.iloc[i].tolist())

            final       = pd.concat([data, R], axis=1)
            final_smote = pd.concat([data, R_smote], axis=1)

            final.to_csv(output_path, sep=' ', header=False, index=False)
            final_smote.to_csv(output_path_smote, sep=' ', header=False, index=False)

            print(f'Done.')
            break


        else:
            print(f'Done.')
            break








































