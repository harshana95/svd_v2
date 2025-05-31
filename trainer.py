import comet_ml
import argparse
import yaml

from accelerate.logging import get_logger
from dataset import create_dataset
from models import create_model
from psf.svpsf import PSFSimulator
from utils import ordered_yaml
from torch.utils.data import DataLoader

logger = get_logger(__name__)

def main(opt):
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
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
        if args.test:
            opt['name'] = 'test'
            opt['tracker_project_name'] = 'test'
        opt.test = args.test
    print("Is test", opt.test)
    main(opt)