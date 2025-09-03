import comet_ml
import argparse
import yaml
import torch.multiprocessing as mp
from accelerate.logging import get_logger
from models import create_model

from utils.misc import DictAsMember
# export QT_QPA_PLATFORM=offscreen
logger = get_logger(__name__)

def main(opt):
    # mp.set_start_method('spawn')
    # ============================================ 1. Initialize
    model = create_model(opt, logger)

    # ============================================ 2. Load dataset
    # create train and validation dataloaders
    model.setup_dataloaders()

    # psf_data, metadata = PSFSimulator.load_psfs(opt.datasets.train.name, 'PSFs_with_basis.h5')
    # basis_psfs = psf_data['basis_psfs'][:]  # [:] is used to load the whole array to memory
    # basis_coef = psf_data['basis_coef'][:]

    # ============================================== 3. setup for training
    model.prepare()

    # ============================================== 4. training
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('-test', action='store_true', help='Save in test.')
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        data = yaml.safe_load(f)
        if isinstance(data, dict):
            opt = DictAsMember(**data)
        else:
            opt = data
        opt.pretty_print()
        if args.test:
            opt['name'] = 'test'
            opt['tracker_project_name'] = 'test'
        opt.test = args.test
    print("Is test", opt.test)
    main(opt)