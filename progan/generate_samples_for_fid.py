""" Script for generating samples for fid calculation"""

import torch as th
import numpy as np
import data_processing.DataLoader as dl
import argparse
import yaml
import os
import pickle
import matplotlib.pyplot as plt
import timeit
from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# set manual seed for the training
th.manual_seed(3)

# turn on fast training of the network
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", action="store", type=int, default=6,
                        help="depth at which samples are to be generated")

    parser.add_argument("--latent_size", action="store", type=int, default=256,
                        help="depth at which samples are to be generated")

    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")

    parser.add_argument("--num_samples", action="store", type=int, default=5000,
                        help="number of samples to be generated")

    parser.add_argument("--batch_size", action="store", type=int, default=16,
                        help="batch size for the process of sample generation")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="../5K_samples_directory",
                        help="path where the generated samples are to be saved")

    args = parser.parse_args()

    return args


def get_config(conf_file):
    """
    parse and load the provided configuration
    :param conf_file: configuration file
    :return: conf => parsed configuration
    """
    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from pro_gan_pytorch.PRO_GAN import Generator

    print("Creating network architecture")
    generator = Generator(depth=args.depth, latent_size=args.latent_size).to(device)
    print("Loading generator from:", args.generator_file)
    generator.load_state_dict(th.load(args.generator_file))

    # generate the required images:
    current_gen = 0
    while current_gen < args.num_samples:
        gen_batch_size = min(args.num_samples - current_gen, args.batch_size)

        # generate current batch of images:
        gen_input_noise = th.randn(gen_batch_size, generator.latent_size).to(device)
        gen_samples = generator(gen_input_noise, depth=4, alpha=1).detach()

        # now save the images in the given directory:
        os.makedirs(args.images_dir, exist_ok=True)
        for (i, gen_sample) in enumerate(gen_samples, 1):
            plt.imsave(os.path.join(args.images_dir, str(current_gen + i) + ".png"),
                       (gen_sample.permute(1, 2, 0) / 2) + 0.5)

        # increment the current_gen pointer:
        current_gen += gen_batch_size

        print("generated %d samples ..." % current_gen)


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
