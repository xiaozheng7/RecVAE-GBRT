import torch

def calculate_loss(x_mean, x, z_mu, z_var):

    rec = torch.nn.MSELoss()(x, x_mean)
    log_var = z_var.log()
    kl = -.5 * torch.sum(1. + log_var - z_mu.pow(2) - log_var.exp(), dim=1, keepdim=True)

    loss = rec + kl

    return loss, rec, kl

