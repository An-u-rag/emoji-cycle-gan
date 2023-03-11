import torch

from tqdm import tqdm
import os
from PIL import Image
from dataloader import get_data_loader
from models import Generator, Discriminator, NoLeakDiscriminator, AdversarialLoss

from options import VanillaGANOptions


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        # config dirs
        self.expdir = f'./leaky_{self.opts.format}_vanilla_gan_{self.opts.lr}'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)

        # config data
        self.trainloader, self.testloader = get_data_loader(
            self.opts.emoji_type, self.opts.batch_size, self.opts.num_workers, self.opts.format)

        # config models
        self.G = Generator(self.opts).to(self.opts.device)
        self.D = Discriminator(self.opts).to(self.opts.device)

        # config optimizers
        self.G_optim = torch.optim.Adam(
            self.G.parameters(), lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(
            self.D.parameters(), lr=self.opts.lr, betas=(0.5, 0.999))

        # config training
        self.nepochs = self.opts.nepochs

        # Loss Function
        self.criterion = AdversarialLoss()

    # helper function for noise generation
    def generate_noise(self, batch_size):
        noise = torch.randn(batch_size, self.opts.noise_size, 1, 1)
        return noise

    def run(self):
        for epoch in range(self.nepochs):
            self.train_step(epoch)

            if epoch % self.opts.eval_freq == 0:
                self.eval_step(epoch)
            if epoch % self.opts.save_freq == 0:
                self.save_checkpoint(epoch)

    def train_step(self, epoch):
        self.G.train()
        self.D.train()

        pbar = tqdm(self.trainloader)

        for i, data in enumerate(pbar):
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            real = data.to(self.opts.device)
            noise = self.generate_noise(real.shape[0]).to(self.opts.device)
            fake = self.G(noise)

            # train discriminator
            ##### TODO: compute discriminator loss and optimize#####
            d_loss = 0.
            D_real = self.D(real)
            real_labels = torch.ones(
                (real.shape[0], 1, 1, 1)).to(self.opts.device)
            real_loss = self.criterion(D_real, real_labels)

            D_fake = self.D(fake)
            fake_labels = torch.zeros(
                (real.shape[0], 1, 1, 1)).to(self.opts.device)
            fake_loss = self.criterion(D_fake, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.D_optim.step()
            ##########################################

            # train generator
            ##### TODO: compute generator loss and optimize#####
            g_loss = 0.
            noise2 = self.generate_noise(real.shape[0]).to(self.opts.device)
            fake_images = self.G(noise2)
            D_fake_images = self.D(fake_images)
            labels = torch.ones((real.shape[0], 1, 1, 1)).to(
                self.opts.device)
            g_loss = self.criterion(D_fake_images, labels)
            g_loss.backward()
            self.G_optim.step()
            ###################################################

            pbar.set_description("Epoch: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
                epoch, g_loss.item(), d_loss.item()))

    def eval_step(self, epoch):
        self.G.eval()
        self.D.eval()
        image_channels = 3 if self.opts.format == "RGB" else 4
        # Number of eval noise
        n = self.opts.valn
        with torch.no_grad():
            ##### TODO: sample from your Generator and save results in self.plotdir#####
            for i in range(n):
                noise = self.generate_noise(1).to(self.opts.device)
                image = self.G(noise).reshape(image_channels, 32, 32)
                image = image.permute(1, 2, 0)
                image = image.clamp(-1, 1).detach().cpu().numpy()
                image = ((image + 1) * 127.5).astype('uint8')
                Image.fromarray(image).save(os.path.join(
                    self.plotdir, f'fig_e{epoch}_{i}.png'))
            #################################################################################

    def save_checkpoint(self, epoch):
        ##### TODO: save your model in self.ckptdir#####
        torch.save(self.G.state_dict(), f'{self.ckptdir}/G_{epoch}.pt')
        torch.save(self.D.state_dict(), f'{self.ckptdir}/D_{epoch}.pt')
        ###############################################


if __name__ == '__main__':
    trainer = Trainer(VanillaGANOptions())
    trainer.run()
