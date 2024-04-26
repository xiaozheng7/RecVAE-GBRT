import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import pandas as pd
import os
import matplotlib.pyplot as plt

import models.VAE as VAE
from models.RecVAE_training import train, evaluate, get_feature
from sklearn import preprocessing

from utils.tools import log_string, add_features, metric

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def data_preprocessing(args, border1s, border2s):
    
    df_raw = pd.read_csv('./data/'+ args.dataset + '.csv', usecols=['date',args.target])
    df_raw_1 = df_raw.iloc[border1s[0]:border2s[2]].copy()

    data_1 = add_features(df_raw_1)
    data_1 = data_1.drop(columns = ['date'])
    data_train = data_1.iloc[border1s[0]:border2s[0]].copy()
    data_train = data_train[[args.target]]

    scaler = preprocessing.StandardScaler()
    scaler.fit(data_train)

    data_1[args.target] = scaler.transform(data_1[[args.target]])

    return data_1

def RecVAE_part(args, whole_data, border1s, border2s, log, setting, exp_path):

    args.input_size = args.z_size * 2 + args.ori_dim

    model = VAE.VAE(args)

    if args.cuda:
        print("Model running on GPU")
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, eps = 1.e-7)
       
    type = args.target
    log_string(log,'\nKey Settings: \n'+ setting +'\n')    
    log_string(log,'\nModel Settings: \n'+ str (args)+'\ntype_'+ type+'\n')

    data_train = whole_data.iloc[border1s[0]:border2s[0]].copy()
    data_val = whole_data.iloc[border1s[1]:border2s[1]].copy()

    data_train = data_train[[type]].values
    data_val = data_val[[type]].values

    train_loss = []
    train_rec_loss = []
    train_kl_loss = []

    val_loss = []
    val_rec_loss = []
    val_kl_loss = []

    best_loss = np.inf
    e = 0
    epoch = 0

    train_times = []

    for epoch in range(1, args.epochs + 1):

        t_start = time.time()

        tr_loss, rec_loss, kl_loss, model = train(epoch, np.squeeze(data_train), model, optimizer, args, log)
        train_loss.append(tr_loss)
        train_rec_loss.append(rec_loss)
        train_kl_loss.append(kl_loss)

        train_times.append(time.time()-t_start)
        log_string(log,'One training epoch took %.2f seconds' % (time.time()-t_start))

        v_loss,rec_loss, kl_loss = evaluate(np.squeeze(data_val), model, args)
        val_loss.append(v_loss)
        val_rec_loss.append(rec_loss)
        val_kl_loss.append(kl_loss)

        # early-stopping
        model_save_path = './checkpoints/' + type + '_' + setting
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if v_loss < best_loss:
            e = 0
            best_loss = v_loss

            log_string(log,'->Model saved<-')
            torch.save(model.state_dict(),model_save_path+'/checkpoint.pth')

        elif args.early_stopping_epochs > 0:
            e += 1
            if e > args.early_stopping_epochs:
                break

        log_string(log,'--> Early stopping: {}/{} (best validation loss {:.4f})\n'.format(e, args.early_stopping_epochs, best_loss))

        if math.isnan(v_loss):
            raise ValueError('NaN encountered!')

    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    log_string(log,'Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

    final_model_to_load = torch.load(model_save_path + '/checkpoint.pth')   

    get_feature(type, np.squeeze(whole_data[type].values), final_model_to_load, model, args, exp_path)

    return

def xgboost_part(args, whole_data,exp_path, log, border1s, border2s):

    def windowing(whole_data, Number_Of_Features,):

        end = len(whole_data)
        start = 0
        next = 0
        x_batches = []
        y_batches = []  

        while start + (args.num_input + args.num_output) < end + 1: 

            next = start + args.num_input
            x_batches.append(whole_data[start:next, :])
            y_batches.append(whole_data[next:next+args.num_output, 0])
            start = start+1

        y_batches = np.asarray(y_batches)
        y_batches = y_batches.reshape(-1, args.num_output, 1) 

        x_batches = np.asarray(x_batches) 
        x_batches = x_batches.reshape(-1, args.num_input, Number_Of_Features)

        return x_batches, y_batches

    def flatten(dataset):

        Instances = []

        for i in range(0, len(dataset)):
            hold = []
            for j in range(0, len(dataset[i])):
                if j == (len(dataset[i])-1):
                    hold = np.concatenate((hold, dataset[i][j][:]), axis=None)     
                else:
                    hold = np.concatenate((hold, dataset[i][j][0]), axis=None)
                
            Instances.append(hold)

        return Instances

    def xgb_preprocessing(args, Number_Of_Features, feature_file_target, Data_df, border1s, border2s):
        
        if args.z_size == 0:
            pass
        else:
            vae_features = pd.read_csv(feature_file_target, header = 0, index_col = 0) 
            for day_count in range(1, Data_df.shape[0] // args.ori_dim, 1): 
                for col_count in range (0, args.z_size, 1):  # Another design option: for col_count in range (0, vae_features.shape[1], 1): 
                    Data_df.loc[args.ori_dim * day_count:args.ori_dim * (day_count + 1), 'vae_target' + str(col_count)] = vae_features.loc[day_count - 1, str(col_count)]

        Train = Data_df.iloc[border1s[0]:border2s[0], :]
        Train = Train.values

        Vali = Data_df.iloc[border1s[1] - args.num_input:border2s[1], :]
        Vali = Vali.values

        Test = Data_df.iloc[border1s[2] - args.num_input:border2s[2], :]
        Test = Test.values

        x_trainbatches, y_trainbatches = windowing(Train, Number_Of_Features)
        x_valibatches, y_valibatches = windowing(Vali, Number_Of_Features)
        x_testbatches, y_testbatches = windowing(Test, Number_Of_Features)

        return x_trainbatches, y_trainbatches, x_valibatches, y_valibatches, x_testbatches, y_testbatches 

    feature_file_target = exp_path + '/RecVAE_features_' + args.target + '.csv'
    Number_Of_Features = 7 + args.z_size

    X_Train_Full = []
    Y_Train_Full = []
    X_Vali_Full = []
    Y_Vali_Full = []
    X_Test_Full = []
    Y_Test_Full = []

    x_TRAIN, y_TRAIN, x_VALI, y_VALI, X_Test, Y_Test = xgb_preprocessing(args, Number_Of_Features, feature_file_target, whole_data, border1s, border2s)

    for element1 in (x_TRAIN):
        X_Train_Full.append(element1)
            
    for element2 in (y_TRAIN):
        Y_Train_Full.append(element2)

    for element3 in (x_VALI):
        X_Vali_Full.append(element3)
            
    for element4 in (y_VALI):
        Y_Vali_Full.append(element4)
                
    for element5 in (X_Test):
        X_Test_Full.append(element5)
            
    for element6 in (Y_Test):
        Y_Test_Full.append(element6)  


    combined = list(zip(X_Train_Full, Y_Train_Full))
    random.shuffle(combined)
    shuffled_batch_features, shuffled_batch_y = zip(*combined)

    All_Training_Instances = flatten(shuffled_batch_features)
    All_Vali_Instances = flatten(X_Vali_Full)
    All_Testing_Instances = flatten(X_Test_Full)

    All_Training_Instances = np.reshape(All_Training_Instances, (len(All_Training_Instances), len(All_Training_Instances[0])))
    shuffled_batch_y = np.reshape(shuffled_batch_y, (len(shuffled_batch_y), args.num_output))

    All_Vali_Instances = np.reshape(All_Vali_Instances, (len(All_Vali_Instances), len(All_Vali_Instances[0])))
    Y_Vali_Full = np.reshape(Y_Vali_Full, (len(Y_Vali_Full), args.num_output))

    All_Testing_Instances=np.reshape(All_Testing_Instances, (len(All_Testing_Instances), len(All_Testing_Instances[0])))
    Y_Test_Full=np.reshape(Y_Test_Full, (len(Y_Test_Full), args.num_output))

    if args.cuda:
        xgbmodel = xgb.XGBRegressor(learning_rate = 0.015,
        max_depth = 4,
        subsample = 0.5,
        colsample_bytree = 0.5,
        n_estimators = 500,
        tree_method = 'gpu_hist')

    else:
        xgbmodel = xgb.XGBRegressor(learning_rate = 0.015,
        max_depth = 4,
        subsample = 0.5,
        colsample_bytree = 0.5,
        n_estimators = 500)


    multioutput = MultiOutputRegressor(xgbmodel).fit(All_Training_Instances,shuffled_batch_y)  

    prediction_vali = multioutput.predict(All_Vali_Instances)
    vali_mae, vali_mse, vali_rmse, vali_mape = metric(prediction_vali, Y_Vali_Full)

    dimension = All_Testing_Instances.shape[0] // 32 * 32 # 32 being batch size of transformer baselines, here is to make the testing sets match.
    All_Testing_Instances = All_Testing_Instances[0:dimension, :]
    Y_Test_Full =  Y_Test_Full[0:dimension, :]

    prediction_test = multioutput.predict(All_Testing_Instances)
    mae,mse,rmse,mape = metric(prediction_test, Y_Test_Full)

    xgboost_path = exp_path + '/' + 'input' + str(args.num_input)
    if not os.path.exists(xgboost_path):
        os.makedirs(xgboost_path)

    np.save(xgboost_path + '/pred_test.npy', prediction_test)
    np.save(xgboost_path + '/trues_test.npy', Y_Test_Full)

    np.save(xgboost_path + '/pred_vali.npy', prediction_vali)
    np.save(xgboost_path + '/trues_vali.npy', Y_Vali_Full)           

    plt.figure(figsize=(300, 10)) 

    for j in range(len(prediction_test)):   
        c, d = [], []
        for i in range(args.num_output):
            c.append(prediction_test[j, i])
            d.append(Y_Test_Full[j, i])
        plt.plot(range(1 + j, args.num_output + 1 + j), c, c='b')
        plt.plot(range(1 + j, args.num_output + 1 + j), d, c='r')

    plt.title('Test prediction vs Target')
    plt.savefig(xgboost_path+ '/test_results.png')

    log_string(log, 'results on validation set')
    log_string(log, 'MSE\t\tMAE\t\tRMSE\t\tMAPE\t\t')
    log_string(log, '%.5f\t\t%.5f\t\t%.5f\t\t%.5f%%\t\t' %
            (vali_mse, vali_mae, vali_rmse, vali_mape*100))

    log_string(log, 'results on test set')
    log_string(log, 'MSE\t\tMAE\t\tRMSE\t\tMAPE\t\t')
    log_string(log, '%.5f\t\t%.5f\t\t%.5f\t\t%.5f%%\t\t' %
            (mse, mae, rmse, mape*100))

    return 

