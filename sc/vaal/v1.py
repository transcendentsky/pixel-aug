import torch
from torchvision import datasets, transforms
# import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from datasets.ceph_vaal import Ceph, ceph_transformer
# import model
from models import vgg_vaal as vgg
from models.vae import VAE, Discriminator
from .vaal_solver import Solver
from utils import *
from tutils import CSVLogger
# from models.network import UNet_Pretrained
from sc.select.test_specific_ids import test_specific_ids
from tutils import trans_args, trans_init, save_script, tfilename, tdir
from datetime import datetime




def main(args, logger, config):
    if args.dataset == "ceph":
        test_dataloader = data.DataLoader(
            datasets.ImageFolder(args.data_path, transform=ceph_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = Ceph(args.data_path)

        # args.num_val = 10
        args.num_images = 150
        args.budget = 3
        args.initial_budget = 5
        args.num_classes = 1000
    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))

    if config['custom'].get('init_ids', None) == None:
        initial_indices = random.sample(list(all_indices), args.initial_budget)
    else:
        initial_indices = config['custom']['init_ids']

    sampler = data.sampler.SubsetRandomSampler(initial_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                        batch_size=args.batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                     batch_size=args.batch_size, drop_last=True)

    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []
    csvlogger = CSVLogger(logdir=config['base']['runs_dir'], name="best_record.csv", mode="a+")

    for split in splits:
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        task_model = vgg.vgg16_bn(num_classes=args.num_classes)

        # task_model = UNet_Pretrained()

        vae = VAE(args.latent_dim)
        discriminator = Discriminator(args.latent_dim)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                                               sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=True)

        # train the models on the current data
        acc, vae, discriminator = solver.train(querry_dataloader,
                                               val_dataloader,
                                               task_model,
                                               vae,
                                               discriminator,
                                               unlabeled_dataloader)


        logger.info('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        accuracies.append(acc)

        _d = test_specific_ids(current_indices)
        _d['split'] = split
        _d['record_time'] = datetime.time()
        csvlogger.record(_d)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        logger.info("# ------------------------------------------------ #")
        logger.info(f"Current_indices: {current_indices}")
        logger.info(f"Sampled_indices for labeling: {sampled_indices}")
        logger.info("# ------------------------------------------------ #")
        csvlogger.record({'current_indices':current_indices, "labeling_indices":sampled_indices})

        indices = list(current_indices) + list(sampled_indices)

        indices_str_list = [str(i) for i in indices]
        indices_str = ",".join(indices_str_list)

        # subprocess.call(f"CUDA_VISIBLE_DEVICES=0 python /home1/quanquan/code/landmark/code/cas-qs/test_by_multi.py --indices {indices_str} --tag spe_test"
        #                 f" --config /home1/quanquan/code/landmark/code/cas-qs/configs/config.yaml", shell=True)
        # break

        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                            batch_size=args.batch_size, drop_last=True)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='ceph', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size used for training and testing')
    parser.add_argument('--train_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default='/home1/quanquan/datasets/Cephalometric/TmpImage/', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    parser.add_argument('--config', default='configs/vaal/vaal.yaml')
    args = trans_args(parser)
    logger, config = trans_init(args)
    tdir(args.out_path)

    main(args, logger, config)

