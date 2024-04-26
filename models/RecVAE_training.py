import torch
from utils.VAE_loss import calculate_loss
import pandas as pd
from utils.tools import log_string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, data_all, model, opt, args, log):

    model.train()
    ini_latent = torch.zeros((2 * args.z_size), dtype=torch.float, device=device) 

    train_loss = []
    train_rec_loss = []
    train_kl_loss = []
    no_train_sample = data_all.shape[0] - args.ori_dim 
    enc_output_save = []

    for seg_id in range (0, no_train_sample, 1):

        data_enc = torch.tensor(data_all[seg_id:(seg_id + args.ori_dim)], dtype=torch.float, device=device) 
        if args.cuda:
            data_enc = data_enc.cuda()

        opt.zero_grad()

        if seg_id == 0:
            data_enc_cat = torch.cat((data_enc, ini_latent),0)
        else:
            data_enc_cat = torch.cat((data_enc, torch.tensor(enc_output_save[-1])),0)

        x_mean, z_mu, z_var = model(data_enc_cat)

        mu_var_concat = torch.cat((z_mu[-1], z_var[-1]), 0)

        enc_output_save.append(mu_var_concat)

        loss, rec, kl = calculate_loss(x_mean, data_enc_cat, z_mu, z_var)

        loss.backward()
        train_loss.append(loss.item())

        rec = rec.item()
        kl = kl.item()

        train_rec_loss.append(rec)
        train_kl_loss.append(kl)

        opt.step()

        if seg_id % 1000 == 0:
                
            log_string(log,'Epoch: {:3d}\t Seg_id:  {:3d} \tLoss: {:11.6f}\trec: {:11.6f}\tkl: {:11.6f}'.format(
                epoch, seg_id, loss.item(), rec, kl))

    train_loss_sum = 0
    rec_loss_sum = 0
    kl_loss_sum = 0

    for i in range(0,len(train_loss)):
        train_loss_sum = train_loss_sum + train_loss[i]
        rec_loss_sum = rec_loss_sum + train_rec_loss[i]       
        kl_loss_sum = kl_loss_sum + train_kl_loss[i]

    log_string(log,'====> Epoch: {:3d} Average train loss: {:.4f}'.format(
            epoch,  train_loss_sum  / no_train_sample))

    log_string(log,'====> Epoch: {:3d} Average rec loss: {:.4f}'.format(
        epoch,  rec_loss_sum  / no_train_sample))

    log_string(log,'====> Epoch: {:3d} Average kl loss: {:.4f}'.format(
        epoch,  kl_loss_sum  / no_train_sample))

    return train_loss_sum  / no_train_sample, rec_loss_sum  / no_train_sample, kl_loss_sum  / no_train_sample, model

def get_feature(type, data_2, model_to_load, model, args, exp_path):

    model.load_state_dict(model_to_load)
    model.eval()

    ini_latent = torch.zeros((2 * args.z_size), dtype=torch.float, device=device) 
    
    with torch.no_grad():
        for seg_id in range (0, data_2.shape[0] - args.ori_dim, args.ori_dim): 

            data_enc = torch.tensor(data_2[seg_id:(seg_id + args.ori_dim)], dtype=torch.float, device=device)

            if args.cuda:
                data_enc = data_enc.cuda()

            if seg_id == 0:
                data_enc_cat = torch.cat((data_enc, ini_latent), 0)
                x_mean, z_mu, z_var = model(data_enc_cat)
                mu_var_concat = torch.cat((z_mu[-1], z_var[-1]), 0)
                feature_to_save = mu_var_concat.unsqueeze(0)

            else:
                data_enc_cat = torch.cat((data_enc, mu_var_concat), 0)
                x_mean, z_mu, z_var = model(data_enc_cat)
                mu_var_concat = torch.cat((z_mu[-1],z_var[-1]), 0)                
                enc_output_save = mu_var_concat.unsqueeze(0)
                feature_to_save = torch.cat((feature_to_save, enc_output_save), 0)
              
        pd.DataFrame(feature_to_save.cpu().numpy()).to_csv(exp_path + '/RecVAE_features_' + type + '.csv')         
        return 


def evaluate(data_all, model, args):

    model.eval()
    ini_latent = torch.zeros((2 * args.z_size),dtype=torch.float, device=device)
    loss = 0.

    val_loss = []
    val_rec_loss = []
    val_kl_loss = []
    no_val_sample = data_all.shape[0] - args.ori_dim 
    enc_output_save = []

    for seg_id in range (0, no_val_sample,1):
        
        data_enc = torch.tensor(data_all[seg_id:(seg_id + args.ori_dim)], dtype=torch.float, device=device) 
        if args.cuda:
            data_enc = data_enc.cuda()

        if seg_id == 0:
            data_enc_cat = torch.cat((data_enc, ini_latent), 0)
        else:
            data_enc_cat = torch.cat((data_enc, torch.tensor(enc_output_save[-1])), 0)         

        x_mean, z_mu, z_var = model(data_enc_cat)
        mu_var_concat = torch.cat((z_mu[-1], z_var[-1]), 0)

        enc_output_save.append(mu_var_concat)

        loss, rec, kl = calculate_loss(x_mean, data_enc_cat, z_mu, z_var)

        val_loss.append(loss.item())
        rec = rec.item()
        kl = kl.item()

        val_rec_loss.append(rec)
        val_kl_loss.append(kl)

    val_loss_sum = 0
    val_rec_loss_sum = 0
    val_kl_loss_sum = 0

    for i in range(0,len(val_loss)):
        val_loss_sum = val_loss_sum + val_loss[i]
        val_rec_loss_sum =  val_rec_loss_sum + val_rec_loss[i]       
        val_kl_loss_sum = val_kl_loss_sum + val_kl_loss[i]

    loss = val_loss_sum  / no_val_sample

    return loss, val_rec_loss_sum  / no_val_sample, val_kl_loss_sum  / no_val_sample
