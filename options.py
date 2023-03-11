import abc


class BaseOptions(metaclass=abc.ABCMeta):
    def __init__(self):
        self.device = 'cuda'
        self.expdir = ''
        self.debug = True


class VanillaGANOptions(BaseOptions):
    def __init__(self):
        super(VanillaGANOptions, self).__init__()

        # Dataset options
        self.data_dir = '../emojis'
        self.emoji_type = 'Apple'
        self.batch_size = 8
        self.num_workers = 0

        # Discriminator options
        self.discriminator_channels = [32, 64, 128, 1]

        # Generator options
        self.generator_channels = [128, 64, 32, 3]
        self.noise_size = 100

        # Training options
        self.nepochs = 200
        self.lr = 0.0002

        # Evaluation options
        self.valn = 1

        self.eval_freq = 1
        self.save_freq = 1

        # Image format options RGB or RGBA
        self.format = "RGBA"

        # Use sigmoid activation as last layer of discriminator
        self.d_sigmoid = True


class CycleGanOptions(BaseOptions):
    def __init__(self):
        super(CycleGanOptions, self).__init__()

        # Generator options
        self.generator_channels = [32, 64]

        # Dataset options
        self.data_dir = '../emojis'

        self.batch_size = 8
        self.num_workers = 0

        # Discriminator options
        self.discriminator_channels = [32, 64, 128, 1]

        # Training options
        self.niters = 100
        self.lr = 0.0003

        self.eval_freq = 1
        self.save_freq = 1

        self.use_cycle_loss = True

        # Image format options RGB or RGBA
        self.format = "RGBA"

        # Use sigmoid activation as last layer of discriminator
        self.d_sigmoid = True
