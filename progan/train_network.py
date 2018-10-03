""" Script for training the ProGAN model """

import torch as th
import numpy as np
import data_processing.DataLoader as dl
import argparse
import yaml
import os
import pickle
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

    parser.add_argument("--config", action="store", type=str, default="configs/1.conf",
                        help="default configuration for the Network")
    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")
    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="generator optimizer file")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="discriminator optimizer file")

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


def create_grid(samples, scale_factor, img_file, width=2, real_imgs=False):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param width: width for the grid
    :param real_imgs: turn off the scaling of images
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import upsample

    # upsample the image
    if scale_factor > 1 and not real_imgs:
        samples = upsample(samples, scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=width, normalize=True, range=(-1, 1))


def train_networks(pro_gan, dataset, start_epochs, epochs,
                   fade_in_percentage, batch_sizes,
                   start_depth, num_workers, feedback_factor, num_samples,
                   log_dir, sample_dir, checkpoint_factor,
                   save_dir):

    assert pro_gan.depth == len(batch_sizes), "batch_sizes not compatible with depth"

    # generate a fixed input noise for sample_generation:
    fixed_input = th.randn(num_samples, pro_gan.gen.latent_size).to(pro_gan.device)

    print("Starting the training process ... ")
    for current_depth in range(start_depth, pro_gan.depth):

        print("\n\nCurrently working on Depth: ", current_depth)
        current_res = np.power(2, current_depth + 2)
        print("Current resolution: %d x %d" % (current_res, current_res))

        data = dl.get_data_loader(dataset, batch_sizes[current_depth], num_workers)
        total_batches = len(iter(data))

        # initialize the ticker
        ticker = start_epochs[current_depth] * total_batches

        for epoch in range(start_epochs[current_depth], epochs[current_depth] + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)

            fader_point = int((fade_in_percentage[current_depth] / 100)
                              * epochs[current_depth] * total_batches)

            for (i, batch) in enumerate(data, 1):
                # calculate the alpha for fading in the layers
                alpha = ticker / fader_point if ticker <= fader_point else 1

                # extract current batch of data for training
                images = batch.to(device)

                gan_input = th.randn(images.shape[0],
                                     pro_gan.gen.latent_size).to(pro_gan.device)

                # optimize the discriminator:
                dis_loss = pro_gan.optimize_discriminator(gan_input, images,
                                                          current_depth, alpha)

                # optimize the generator:
                gan_input = th.randn(images.shape[0],
                                     pro_gan.gen.latent_size).to(pro_gan.device)
                gen_loss = pro_gan.optimize_generator(gan_input, current_depth, alpha)

                # provide a loss feedback
                if i % int(total_batches / feedback_factor) == 0 or i == 1:
                    print("batch: %d  d_loss: %f  g_loss: %f" % (i, dis_loss, gen_loss))

                    # create the log_dir if it doesn't exist:
                    os.makedirs(log_dir, exist_ok=True)

                    # also write the losses to the log file:
                    log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                    with open(log_file, "a") as log:
                        log.write(str(dis_loss) + "\t" + str(gen_loss) + "\n")

                    # create the sample_dir if it doesn't exist:
                    os.makedirs(sample_dir, exist_ok=True)

                    # create a grid of samples and save it
                    gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +
                                                "_" + str(epoch) + "_" +
                                                str(i) + ".png")
                    create_grid(
                        samples=pro_gan.gen(
                            fixed_input,
                            current_depth,
                            alpha
                        ),
                        scale_factor=int(np.power(2, pro_gan.depth - current_depth - 1)),
                        img_file=gen_img_file,
                        width=int(np.sqrt(fixed_input.shape[0])),
                    )

                # increment the alpha ticker
                ticker += 1

            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                # create the save_directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)

                # save the backups
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + ".pth")
                gen_optim_save_file = os.path.join(save_dir, "GAN_GEN_OPTIM_" +
                                                   str(current_depth) + ".pth")
                dis_optim_save_file = os.path.join(save_dir, "GAN_DIS_OPTIM_" +
                                                   str(current_depth) + ".pth")

                th.save(pro_gan.gen.state_dict(), gen_save_file, pickle)
                th.save(pro_gan.dis.state_dict(), dis_save_file, pickle)
                th.save(pro_gan.gen_optim.state_dict(), gen_optim_save_file, pickle)
                th.save(pro_gan.dis_optim.state_dict(), dis_optim_save_file, pickle)

    print("Training completed ...")


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from pro_gan_pytorch.PRO_GAN import ProGAN

    print(args.config)
    config = get_config(args.config)
    print("Current Configuration:", config)

    # create the dataset for training
    dataset = dl.FoldersDistributedDataset(
        data_dir=config.images_dir,
        transform=dl.get_transform(config.img_dims)
    )

    print("total examples in training: ", len(dataset))

    pro_gan = ProGAN(
        depth=config.depth,
        latent_size=config.latent_size,
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        eps=config.eps,
        drift=config.drift,
        n_critic=config.n_critic,
        use_eql=config.use_eql,
        loss=config.loss_function,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
        device=device
    )

    if args.generator_file is not None:
        print("Loading generator from:", args.generator_file)
        pro_gan.gen.load_state_dict(th.load(args.generator_file))

    if args.discriminator_file is not None:
        print("Loading discriminator from:", args.discriminator_file)
        pro_gan.dis.load_state_dict(th.load(args.discriminator_file))

    if args.gen_optim_file is not None:
        print("Loading generator optimizer from:", args.gen_optim_file)
        pro_gan.gen_optim.load_state_dict(th.load(args.gen_optim_file))

    if args.dis_optim_file is not None:
        print("Loading discriminator optimizer from:", args.dis_optim_file)
        pro_gan.dis_optim.load_state_dict(th.load(args.dis_optim_file))

    # train all the networks
    train_networks(
        pro_gan=pro_gan,
        dataset=dataset,
        start_epochs=config.start_epochs,
        epochs=config.epochs,
        fade_in_percentage=config.fade_in_percentage,
        start_depth=args.start_depth,
        batch_sizes=config.batch_sizes,
        num_workers=config.num_workers,
        feedback_factor=config.feedback_factor,
        num_samples=config.num_samples,
        log_dir=config.log_dir,
        sample_dir=config.sample_dir,
        checkpoint_factor=config.checkpoint_factor,
        save_dir=config.save_dir,
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
