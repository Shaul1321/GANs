import torch

import numpy as np
from torch import nn
from torch import optim
from tqdm import trange
import dynet as dy
import network
from generate_data import get_data
import matplotlib.pyplot as plt
from utils import *

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
N = 500


class Generator(nn.Module):
    def __init__(self):
        super(LineGenerator, self).__init__()
        # Takes a uniform signal and outputs a line
        
        layers = []
        layers.append(nn.Linear(N, 1024))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(1024, 2048))
        layers.append(nn.LeakyReLU(0.2))    
        layers.append(nn.Linear(2048, 2048))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(2048, out_features = 1024))
        #layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, samples):
        """
        z1 = self.dropout(self.lin1(samples))
        a1 = self.relu1(z1)
        z2 = self.lin2(a1)
        a2 = self.relu2(z2)
        z3 = self.lin3(a2)
        return z3
        """
        
        return self.net(samples)


class Discriminator(nn.Module):
    def __init__(self):
        super(LineDiscriminator, self).__init__()
        """
        self.lin1 = nn.Linear(in_features=784, out_features=512)
        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.0)
        self.lin2 = nn.Linear(in_features=1024, out_features=512)
        self.dropout2 = nn.Dropout(0.0)
        self.relu2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(in_features=512, out_features=1)
        self.sigm = nn.Sigmoid()
        """
        
        layers = []
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(1024, 512))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(512, 256))    
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(256, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, points):
        """
        z1 = self.lin1(points)
        a1 = self.dropout1(self.relu(z1))
        z2 = self.dropout2(self.relu2(self.lin2(a1)))
        z3 = self.lin3(z2)
        a3 = self.sigm(z3)
        return a3
        """
        return self.net(points)



def get_latent_samples(batch_size: int):
    return torch.randn((batch_size, N))


def monitor(real_output,
            fake_output,
            fake_output_for_gen,
            dis_fake_loss,
            dis_real_loss,
            gen_loss):
    D_x = real_output.mean().item()
    D_G_z = fake_output.mean().item()
    D_G_z2 = fake_output_for_gen.mean().item()
    dis_loss = dis_real_loss + dis_fake_loss

    means = f"D_x={D_x:3.2f}\tD_G_z={D_G_z:3.2f}\tD_G_z2={D_G_z2:3.2f}\t"
    losses = f"D_loss={dis_loss:3.3f}\tG_loss={gen_loss}"
    return means + losses


def train(gen, dis, original_model, num_epochs, dis_updates, gen_updates):
    batch_size = 100

    loss_fn = nn.BCELoss()
    gen_optimizer = optim.Adam(gen.parameters(), lr = 0.0002)
    dis_optimizer = optim.Adam(dis.parameters(), lr = 0.0002)

    l = int(num_epochs / batch_size)
    epochs = trange(l, desc="Training...")
    for t in epochs:
    
        if t % 1000 == 0 and t > 0:
        
            generate_fake_samples(gen, original_model)
            
        for i in range(dis_updates):
            # Discriminator: x~Data, z~noise  Maximize log-likelihood:
            # maximize D | log(D(x)) + log(1 - D(G(z)))
            dis.zero_grad()
            samples = get_latent_samples(batch_size)

            fake_data = gen(samples).detach()
            real_data = torch.from_numpy(get_data(batch_size)).float().type(dtype)

            true_label = torch.ones(batch_size)
            # train discriminator on real data
            real_output = dis(real_data)
            dis_real_loss = loss_fn(real_output.view(-1), true_label)
            dis_real_loss.backward()  # back-prop

            # train discriminator on fake data
            fake_label = torch.zeros(batch_size)
            fake_output = dis(fake_data)
            dis_fake_loss = loss_fn(fake_output.view(-1), fake_label)
            dis_fake_loss.backward()

            # Update weights
            dis_optimizer.step()

            # Generator: z~noise, make discriminator accept generator output:
            # Maximize log likelihood of generated data according to discriminator
            # maximize G | log(D(G(z))

        for i in range(gen_updates):
            gen.zero_grad()
            # another forward pass on D
            samples = get_latent_samples(batch_size)
            fake_data = gen(samples)
            fake_output_for_gen = dis(fake_data)
            # we maximize log(D(G(z)) by using the true label
            true_label = torch.ones(batch_size)
            gen_loss = loss_fn(fake_output_for_gen.view(-1), true_label)
            gen_loss.backward()

            # Update weights
            gen_optimizer.step()

        if t % 10 == 0:
            monitor_string = monitor(
                real_output,
                fake_output,
                fake_output_for_gen,
                dis_fake_loss,
                dis_real_loss,
                gen_loss)
            epochs.set_postfix_str(monitor_string)
    return


def generate_fake_samples(gen, original_model):
    samples = get_latent_samples(15)
    with torch.no_grad():
        gen.eval()
        generated = gen(samples).detach().cpu().numpy()
        gen.train()
    
    for row in generated:
    
        original_model.generate_from_encoding_vector(row)
        


def main():
    train_params = {
        "num_epochs": 80000000,
        "dis_updates": 1,
        "gen_updates": 1
    }


    gen = Generator().cuda()
    dis = Discriminator().cuda()
    model = dy.Model()
    rnn = network.Network(W2I, I2W, model)
    model.populate("model.m")
    train(gen, dis, rnn, **train_params)
    #plot_from_generator(gen, dis)
    #plt.show()


if __name__ == "__main__":
    main()
