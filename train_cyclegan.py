import torch

from tqdm import tqdm
import os
import torchvision.transforms as T
from dataloader import get_data_loader
from models import Discriminator, NoLeakDiscriminator, CycleGenerator, AdversarialLoss, CycleLoss
from PIL import Image
from options import CycleGanOptions


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        # config dirs
        self.expdir = f'./sigmoid_leaky_{self.opts.format}_cycle_gan_{self.opts.lr}'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)

        # config data
        self.apple_trainloader, self.apple_testloader = get_data_loader(
            'Apple', self.opts.batch_size, self.opts.num_workers, self.opts.format)
        self.windows_trainloader, self.windows_testloader = get_data_loader(
            'Windows', self.opts.batch_size, self.opts.num_workers, self.opts.format)

        # config models

        # apple->windows generator
        self.G_a2w = CycleGenerator(self.opts).to(self.opts.device)
        # windows->apple generator
        self.G_w2a = CycleGenerator(self.opts).to(self.opts.device)

        generator_params = list(self.G_a2w.parameters()) + \
            list(self.G_w2a.parameters())

        # apple discriminator
        self.D_a = Discriminator(self.opts).to(self.opts.device)
        # windows discriminator
        self.D_w = Discriminator(self.opts).to(self.opts.device)

        discriminator_params = list(
            self.D_a.parameters()) + list(self.D_w.parameters())

        # config optimizers
        self.G_optim = torch.optim.Adam(
            generator_params, lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(
            discriminator_params, lr=self.opts.lr, betas=(0.5, 0.999))

        # config training
        self.niters = self.opts.niters

        # Loss Function
        self.criterion = AdversarialLoss()
        self.cycle = CycleLoss()

    def run(self):

        for i in range(self.niters):
            if i % self.opts.eval_freq == 0:
                self.eval_step(i)
            if i % self.opts.save_freq == 0:
                self.save_step(i)
            self.train_step(i)

    def train_step(self, epoch):
        self.G_w2a.train()
        self.G_a2w.train()

        self.D_a.train()
        self.D_w.train()

        apple_loader = iter(self.apple_trainloader)
        windows_loader = iter(self.windows_trainloader)

        # num_iters = min(len(self.apple_trainloader) // self.opts.batch_size,
        #                 len(self.windows_trainloader) // self.opts.batch_size)

        num_iters = min(len(self.apple_trainloader),
                        len(self.windows_trainloader))

        pbar = tqdm(range(num_iters))
        for i in pbar:
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            # load data
            apple_data = next(apple_loader).to(self.opts.device)
            windows_data = next(windows_loader).to(self.opts.device)

            ##### TODO:train discriminator on real data#####
            D_real_loss = 0.
            D_real_apple = self.D_a(apple_data)
            real_apple_labels = torch.ones(
                (apple_data.shape[0], 1, 1, 1)).to(self.opts.device)
            D_real_apple_loss = self.criterion(D_real_apple, real_apple_labels)

            D_real_windows = self.D_w(windows_data)
            real_windows_labels = torch.ones(
                (windows_data.shape[0], 1, 1, 1)).to(self.opts.device)
            D_real_windows_loss = self.criterion(
                D_real_windows, real_windows_labels)
            D_real_loss = D_real_apple_loss + D_real_windows_loss
            D_real_loss.backward()
            ###############################################

            ##### TODO:train discriminator on fake data#####
            D_fake_loss = 0.
            G_fake_windows = self.G_a2w(apple_data)
            D_fake_windows = self.D_w(G_fake_windows)
            fake_windows_labels = torch.zeros(
                (apple_data.shape[0], 1, 1, 1)).to(self.opts.device)
            D_fake_windows_loss = self.criterion(
                D_fake_windows, fake_windows_labels)

            G_fake_apple = self.G_w2a(windows_data)
            D_fake_apple = self.D_a(G_fake_apple)
            fake_apple_labels = torch.zeros(
                (windows_data.shape[0], 1, 1, 1)).to(self.opts.device)
            D_fake_apple_loss = self.criterion(D_fake_apple, fake_apple_labels)
            D_fake_loss = D_fake_apple_loss + D_fake_windows_loss
            D_fake_loss.backward()
            self.D_optim.step()
            ###############################################

            ##### TODO:train generator#####
            G_loss = 0.
            G_w2a_images = self.G_w2a(windows_data)
            D_w2a = self.D_a(G_w2a_images)
            G_w2a_loss = self.criterion(D_w2a, real_windows_labels)

            G_a2w_images = self.G_a2w(apple_data)
            D_a2w = self.D_w(G_a2w_images)
            G_a2w_loss = self.criterion(D_a2w, real_apple_labels)

            G_loss = G_w2a_loss + G_a2w_loss

            if self.opts.use_cycle_loss:
                # Forward cycle consistency : apple_data -> G_a2w(apple_data) -> G_w2a(G_a2w(apple_data)) = apple_data
                forward_cycle_loss = self.cycle(
                    self.G_w2a(self.G_a2w(apple_data)), apple_data)

                # Backward cycle consistency : windows_data -> G_w2a(windows_data) -> G_a2w(G_w2a(windows_data)) = windows_data
                backward_cycle_loss = self.cycle(self.G_a2w(
                    self.G_w2a(windows_data)), windows_data)

                G_loss += (forward_cycle_loss + backward_cycle_loss)

            G_loss.backward()
            self.G_optim.step()
            ##############################

            pbar.set_description('Epoch: {}, G_loss: {:.4f}, D_loss: {:.4f}'.format(
                epoch, G_loss.item(), D_real_loss.item() + D_fake_loss.item()))

    def eval_step(self, epoch):
        self.G_w2a.eval()
        self.G_a2w.eval()
        image_channels = 3 if self.opts.format == "RGB" else 4
        ##### TODO: generate 16 images from apple to windows and windows to apple from test data and save them in self.plotdir#####
        apple_tests = iter(self.apple_testloader)
        windows_tests = iter(self.windows_testloader)

        for i in range(16):
            tenstopil = T.ToPILImage()

            apple_image = next(apple_tests).to(self.opts.device)
            apple_image_save = tenstopil(apple_image.squeeze())
            apple_image_save.save(os.path.join(
                self.plotdir, f'apple_e{epoch}_{i}.png'))

            windows_image = next(windows_tests).to(self.opts.device)
            windows_image_save = tenstopil(windows_image.squeeze())
            windows_image_save.save(os.path.join(
                self.plotdir, f'windows_e{epoch}_{i}.png'))

            a2w_translation = self.G_a2w(apple_image).reshape(
                image_channels, 32, 32).permute(1, 2, 0).clamp(-1, 1).detach().cpu().numpy()
            a2w_translation = ((a2w_translation + 1) * 127.5).astype('uint8')
            Image.fromarray(a2w_translation).save(os.path.join(
                self.plotdir, f'windowsGen_e{epoch}_{i}.png'))

            w2a_translation = self.G_w2a(windows_image).reshape(
                image_channels, 32, 32).permute(1, 2, 0).clamp(-1, 1).detach().cpu().numpy()
            w2a_translation = ((w2a_translation + 1) * 127.5).astype('uint8')
            Image.fromarray(w2a_translation).save(os.path.join(
                self.plotdir, f'appleGen_e{epoch}_{i}.png'))

    def save_step(self, epoch):
        ##### TODO: save models in self.ckptdir#####
        torch.save(self.G_a2w.state_dict(),
                   f'{self.ckptdir}/G_a2w__{epoch}.pt')
        torch.save(self.D_a.state_dict(), f'{self.ckptdir}/D_a__{epoch}.pt')
        torch.save(self.G_w2a.state_dict(),
                   f'{self.ckptdir}/G_w2a__{epoch}.pt')
        torch.save(self.D_w.state_dict(), f'{self.ckptdir}/D_w__{epoch}.pt')


if __name__ == '__main__':
    opts = CycleGanOptions()
    trainer = Trainer(opts)
    trainer.run()
