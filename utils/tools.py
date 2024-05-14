import numpy as np
import pandas as pd

def log_string(log, string):

    log.write(string + '\n') 
    log.flush() 
        
    print(string)


def add_features (df_raw):
    
    df_raw['date'] = pd.to_datetime(df_raw.date, format='mixed', dayfirst = True)

    df_raw['month'] = df_raw.date.apply(lambda row:row.month, 1)
    df_raw['dayofmonth'] = df_raw.date.apply(lambda row:row.day, 1)
    df_raw['hour'] = df_raw.date.apply(lambda row:row.hour, 1)
    df_raw['dayofweek'] = df_raw.date.apply(lambda row:row.dayofweek, 1)
    df_raw['day_of_year'] = df_raw.date.apply(lambda row:row.timetuple().tm_yday, 1)
    df_raw['week_of_year'] = df_raw.date.apply(lambda row:row.isocalendar()[1])

    return df_raw

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
  
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape

