import torch, argparse
from torch.utils.data import DataLoader

from data.dataset import my_dataset
from models.cycleGAN_new import CycleGANModel
from utils.utils import get_obj_from_str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='models.cycleGAN_new.CycleGANModel', type=str, help='model class')

    args = parser.parse_args()
    return args


args = get_args()
model_obj = args.m
args.isTrain = True
args.lambda_identity = 1e-3

args.input_nc = 3
args.output_nc = 3
args.ngf = 64
# args.netG = 'unet_256'
args.netG = 'resnet_6blocks'

args.norm = 'instance'
args.no_dropout = True
args.init_type = 'normal'
args.init_gain = 0.02
args.ndf = 64
args.netD = 'basic'
args.n_layers_D = 3
args.lambda_identity = 0.5
args.pool_size = 50
args.gan_mode = 'lsgan'
args.lr = 0.0002
args.beta1 = 0.5
args.lr_policy = 'linear'
args.lr_decay_iters = 50
args.epoch_count = 1
args.n_epochs = 100
args.n_epochs_decay = 100
args.verbose = True
args.direction = 'AtoB'
args.use_dropout = True

args.gpu_ids = ['cpu']

# from cycleGAN_utils.networks import ResnetGenerator, get_norm_layer
# from torchsummary import summary
#
# norm_layer = get_norm_layer(norm_type=args.norm)
# model = ResnetGenerator(args.input_nc, args.output_nc, args.ngf, norm_layer=norm_layer, use_dropout=args.use_dropout, n_blocks=6)
# summary(model, (3, 224, 224))


model = get_obj_from_str(model_obj)(opt=args)
model.setup(args)

test_dataset = my_dataset(ds_name_list=['D2'], path_key='org_dataset', txt_name='test.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

for idx, data_dict in enumerate(test_loader):
    image = data_dict['image']
    ds_label = data_dict['ds_label']
    ped_label = data_dict['ped_label']

    model.set_input(image)
    model.optimize_parameters()
    break

# from cycleGAN_utils import networks
# from torchsummary import summary
#
# opt = args
# netG = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
#
# summary(netG, (3, 128, 128))
#


































