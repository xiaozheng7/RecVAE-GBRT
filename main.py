import argparse
import torch
import torch.utils.data
import os

from utils.tools import log_string
from exp import data_preprocessing, RecVAE_part, xgboost_part

def main():

    parser = argparse.ArgumentParser(description='RecVAE-GBRT')

    parser.add_argument('--dataset', type=str, default='Traffic', 
                        help='dataset')

    parser.add_argument('--target', type=str, default='OT',
                        help='value to be predicted.')

    parser.add_argument('--ori_dim', type=int, default=24, 
                        help='length of input section, s')

    parser.add_argument('--z_size', type=int, default=7, 
                    help='hidden state size of RecVAE, h')
                     
    parser.add_argument('-input_size', '--input_size', type=int, default=24 + 2 * 11, 
                        help='input demension of RecVAE, s+h+h')

    parser.add_argument('--cuda', type=bool, default=True, 
                        help='use gpu')

    parser.add_argument('--gpu_num', type=int, default=0, 
                        help='choose GPU to run on.')

    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='number of epochs to train RecVAE')

    parser.add_argument('-es', '--early_stopping_epochs', type=int, default=3, 
                        help='number of early stopping epochs')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, 
                        help='learning rate')

    # XGBoost prediction settings
    parser.add_argument('--num_input', type=int, default=168, 
                        help='size of lookback window, l')

    parser.add_argument('--num_output', type=int, default=24, 
                        help='output sequence length (predict horizon), Tp')


    args = parser.parse_args()

    args.cuda = True if torch.cuda.is_available() else False

    if args.cuda:
        torch.cuda.set_device(args.gpu_num)

    data_parser = { 
        'Traffic':{'data':'traffic','target':'OT','border1s': [0, 12281, 14035],'border2s':[12281, 14035, 17544]}, 
        'ETTh2':{'data':'ETTh2','target':'OT','border1s': [0, 24*30*12, 24*30*(12+4)],'border2s':[24*30*12, 24*30*(12+4), 24*30*(12+4+4)]},
        'ECL':{'data':'ECL','target':'MT_320','border1s': [0, 24*30*15, 24*30*(15+3)],'border2s':[24*30*15, 24*30*(15+3), 24*30*(15+3+4)]}, 
        'Weather':{'data':'WTH','target':'WetBulbCelsius','border1s': [0, 24*30*28, 24*30*(28+10)],'border2s':[24*30*28, 24*30*(28+10), 24*30*(28+10+10)]}
    }


    if args.dataset in data_parser.keys():
        data_info = data_parser[args.dataset]
        args.dataset = data_info['data']
        args.target = data_info['target']
        border1s = data_info['border1s']
        border2s = data_info['border2s']

    whole_data = data_preprocessing(args,border1s,border2s)

    setting = 'data_{}/od_{}/z_size_{}'.format(args.dataset, args.ori_dim, args.z_size)

    exp_path = './results/' + setting

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log = open (exp_path + '/log.txt', 'w')
    log_string(log, exp_path)


    if args.z_size != 0:
        log_string(log, '##################### Now running RecVAE part #####################')
        RecVAE_part(args, whole_data.copy(), border1s, border2s, log, setting, exp_path)

    log_string(log, '##################### Now running XGBoost part #####################')
    xgboost_part(args, whole_data.copy(), exp_path, log, border1s, border2s)

        

if __name__ == "__main__":
    main()

