import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_sn=True):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_sn=use_sn, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_sn=use_sn, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], numd=1, use_sn=True):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, numd=numd, norm_layer=norm_layer, use_sn=True)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)
    
    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
                
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(prediction - 1, self.get_zero_tensor(prediction))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-prediction - 1, self.get_zero_tensor(prediction))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(prediction)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_sn=False, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        self.input = SNConv(input_nc, ngf, norm_layer, k=7, s=1, pad=3, use_sn=use_sn, use_norm=True, use_act=True)
        self.encode2 = SNConv(ngf, ngf*2, norm_layer, k=3, s=2, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        self.encode4 = SNConv(ngf*2, ngf*4, norm_layer, k=3, s=2, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        self.encode8 = SNConv(ngf*4, ngf*8, norm_layer, k=3, s=2, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        self.encode16 = SNConv(ngf*8, ngf*8, norm_layer, k=3, s=2, pad=1, use_sn=use_sn, use_norm=True, use_act=True)

        resblock = []
        for i in range(n_blocks):       # add ResNet blocks
            resblock += [ResnetBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_sn=use_sn)]
        self.resblock = nn.Sequential(*resblock)
        
        
        self.decode16 =  SNConv(ngf*8, ngf*8, norm_layer, k=3, s=1, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        self.decode8 =  SNConv(ngf*8, ngf*4, norm_layer, k=3, s=1, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        self.decode4 =  SNConv(ngf*4, ngf*2, norm_layer, k=3, s=1, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        self.decode2 =  SNConv(ngf*2, ngf*1, norm_layer, k=3, s=1, pad=1, use_sn=use_sn, use_norm=True, use_act=True)
        
            
        self.rgb = nn.Sequential(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3, padding_mode='reflect'),
                                 nn.Tanh())

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.input(x)   #H
        x = self.encode2(x) #H//2
        x = self.encode4(x) #H//4
        x = self.encode8(x) #H//8
        x = self.encode16(x)#H//16
        
        x = self.resblock(x)
        
        x = self.decode16(x)
        x = self.up(x)      #H//8
        
        x = self.decode8(x) 
        x = self.up(x)      #H//4
        
        x = self.decode4(x) 
        x = self.up(x)      #H//2
        
        x = self.decode2(x)
        x = self.up(x)      #H
        
        x = self.rgb(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, din, dout, norm_layer, use_sn):
        super(ResnetBlock, self).__init__()
        self.learned_shortcut = (din != dout)
        
        self.conv1 = nn.Conv2d(din, din, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(din, dout, kernel_size=3, padding=1, padding_mode='reflect')
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(din, dout, kernel_size=1, bias=False)
            self.norms = norm_layer(dout)
        if use_sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm1 = norm_layer(din)
        self.norm2 = norm_layer(dout)

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.actvn(self.norm1(self.conv1(x)))
        dx = self.norm2(self.conv2(dx))
        out = x_s + dx

        return self.actvn(out)
    
    def shortcut(self,x):
        if self.learned_shortcut:
            x_s = self.norms(self.conv_s(x))
        else:
            x_s = x
        return x_s
    
    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SNConv(nn.Module):
    def __init__(self, din, dout, norm_layer, k, s, pad, use_sn=False, use_norm=True, use_act=True):
        super(SNConv, self).__init__()
        self.use_norm = use_norm 
        self.use_act = use_act
        self.conv = nn.Conv2d(din, dout, kernel_size=k, padding=pad, padding_mode='reflect')
        if use_sn:
            self.conv = spectral_norm(self.conv)
        if use_norm:
            self.norm = norm_layer(dout)
        if use_act:
            self.act =  nn.LeakyReLU(0.2, True)
    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sn=True):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [SNConv(input_nc, ndf, norm_layer, k=4, s=2, pad=1, use_sn=use_sn, use_norm=False, use_act=True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [SNConv(ndf * nf_mult_prev, ndf * nf_mult, norm_layer, k=4, s=2, pad=1, use_sn=use_sn, use_norm=True, use_act=True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [SNConv(ndf * nf_mult_prev, ndf * nf_mult, norm_layer, k=4, s=2, pad=1, use_sn=use_sn, use_norm=True, use_act=True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, numd=3, norm_layer=nn.BatchNorm2d, use_sn=True):
        super().__init__()
        self.numd = numd
        self.model = nn.ModuleList()
        for i in range(numd):
            self.model.append(NLayerDiscriminator(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer, use_sn=use_sn))

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input):
        result = []
        for D in self.model:
            out = D(input)
            result.append(out)
            input = self.downsample(input)

        return result