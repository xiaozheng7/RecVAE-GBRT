import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):

    def __init__(self, args):
        super(VAE, self).__init__()

        self.z_size = args.z_size
        self.input_size = args.input_size

        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()

        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

    def create_encoder(self):

        q_z_nn = nn.Sequential(
            nn.Linear(self.input_size, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU()
        )

        q_z_mean = nn.Linear(48, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(48, self.z_size),
            nn.Softplus()
        )

        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self): 

        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU()
        )

        p_x_mean = nn.Sequential(
            nn.Linear(48, self.input_size)
        )

        return p_x_nn, p_x_mean

    def reparameterize(self, mu, var):
 
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu) 

        return z

    def encode(self, x):

        h = self.q_z_nn(x)
        h = h.view(1, -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)

        return mean, var

    def decode(self, z):

        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def forward(self, x):

        z_mu, z_var = self.encode(x)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var


