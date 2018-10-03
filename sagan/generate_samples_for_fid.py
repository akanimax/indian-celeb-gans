""" script for generating samples from a trained model """

import torch as th
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

th.manual_seed(3)  # set manual seed = 3

# enable fast training
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_config", action="store", type=str,
                        default="configs/1/gen.conf",
                        help="default configuration for generator network")

    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    parser.add_argument("--images_dir", action="store", type=str,
                        default="../5K_sample_directory/",
                        help="path for saving the generated images")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=128,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=64,
                        help="batch_size for training")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=5000,
                        help="Number of samples to be generated by the script")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function of the script
    :param args: parsed commandline arguments
    :return: None
    """
    from attn_gan_pytorch.Networks import Generator
    from attn_gan_pytorch.ConfigManagement import get_config
    from attn_gan_pytorch.Utils import get_layer

    # create generator object:
    print("Creating a generator object ...")
    gen_conf = get_config(args.generator_config)
    gen_conf = list(map(get_layer, gen_conf.architecture))
    generator = Generator(gen_conf, args.latent_size).to(device)

    # load the trained generator weights
    print("loading the trained generator weights ...")
    generator.load_state_dict(th.load(args.generator_file))

    # generate the required images:
    current_gen = 0
    while current_gen < args.num_samples:
        gen_batch_size = min(args.num_samples - current_gen, args.batch_size)

        # generate current batch of images:
        gen_input_noise = th.randn(gen_batch_size, generator.latent_size,
                                   1, 1).to(device)
        gen_samples = generator(gen_input_noise).detach()

        # now save the images in the given directory:
        os.makedirs(args.images_dir, exist_ok=True)
        for (i, gen_sample) in enumerate(gen_samples, 1):
            plt.imsave(os.path.join(args.images_dir, str(current_gen + i) + ".png"),
                       (gen_sample.permute(1, 2, 0) / 2) + 0.5)

        # increment the current_gen pointer:
        current_gen += gen_batch_size

        print("generated %d samples ..." % current_gen)


if __name__ == "__main__":
    main(parse_arguments())
