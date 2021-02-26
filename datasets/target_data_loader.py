"""
TargetDataLoader Data loader
"""
import imageio
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import PIL
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class TargetDataLoader():
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.domain_name = config.target_domain
        if config.data_mode == "imgs":
            img_root_folder = config.datasets_root_dir
            generative_fsl_train_transforms = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(config.ch_mean, config.ch_std_dev)
            ])
            generative_fsl_test_transforms = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(config.ch_mean, config.ch_std_dev)
            ])
            # generative_fsl_transforms.append(ReshapeTransform((-1,)))
            target_dataset_train = datasets.ImageFolder(root=os.path.join(img_root_folder, self.domain_name, "train"),
                                                        transform=generative_fsl_train_transforms)
            target_dataset_test = datasets.ImageFolder(root=os.path.join(img_root_folder, self.domain_name, "test"),
                                                       transform=generative_fsl_test_transforms)
            self.train_loader = torch.utils.data.DataLoader(
                target_dataset_train, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
            self.test_loader = torch.utils.data.DataLoader(
                target_dataset_test, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
            print("length of  train loader", len(self.train_loader))
            print("length of  test loader", len(self.test_loader))
        elif config.data_mode == "download":
            raise NotImplementedError("This mode is not implemented YET")
        else:
            raise Exception(
                "Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, batch_idx, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}_{}_{}.jpg'.format(
            self.config.out_dir, epoch, self.domain_name, batch_idx)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=8,
                           padding=2,
                           normalize=True)
        print("############", img_epoch)
        return imageio.imread(img_epoch)

    def plot_samples_per_epoch_with_labels(self, batch, batch_idx, epoch, labels):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}_{}_{}.jpg'.format(
            self.config.out_dir, epoch, self.domain_name, batch_idx)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=8,
                           padding=2,
                           normalize=True)
        print("############", img_epoch)
        print("############", labels)
        return imageio.imread(img_epoch)
    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            for batch_idx in range(self.config.batch_size):
                img_epoch = '{}samples_epoch_{:d}_{}_{}.jpg'.format(
                    self.config.out_dir, epoch, self.domain_name, batch_idx)
                try:
                    gen_image_plots.append(imageio.imread(img_epoch))
                except OSError as e:
                    pass

        imageio.mimsave(self.config.out_dir +
                        'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass
