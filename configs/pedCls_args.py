# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py

import argparse


class BaseArgs():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--ped_model_obj', type=str, default='models.EfficientNet.efficientNetB0')
        parser.add_argument('--ds_name_list', nargs='+', default=['D1'], help='dataset list')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--data_key', type=str, default='tiny_dataset')
        parser.add_argument('--isTrain', action='store_true')
        parser.add_argument('--rand_seed', type=int, default=3)

        self.initialized = True
        return parser


    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get basic args
        opt, _ = parser.parse_known_args()
        # self.parser = parser

        return opt


class TrainArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = BaseArgs.initialize(self, parser)

        parser.add_argument('--ds_model_obj', type=str, default=None)
        parser.add_argument('--ds_weights_path', type=str, default=None)
        parser.add_argument('--epochs', type=int, default=150)
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--warmup_epochs', type=int, default=3)

        # image operator type, blur / fade
        parser.add_argument('--operator', type=str, default='fade', help='types of operator that handle the image')
        parser.add_argument('--beta', type=float, default=0.0)

        # model
        parser.add_argument('--init_method', type=str, default='orthogonal', help='the way to initialize model weights, e.g., kaiming, orthogonal')
        parser.add_argument('--base_lr', type=float, default=0.01)

        # callbacks
        parser.add_argument('--top_k', type=int, default=2)
        parser.add_argument('--patience', type=int, default=15)

        return parser

class TestArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = BaseArgs.initialize(self, parser)

        parser.add_argument('--ped_weights_path', type=str, default=None)

        return parser



# if __name__ == '__main__':
#     print('test')
#     opts = TrainArgs().parse()
#     print(opts)

























