"""
Mnist Data loader, as given in Mnist tutorial
"""
import imageio
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import PIL
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler, BatchSampler, SequentialSampler, WeightedRandomSampler
import os


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class FinetuneDataLoader():
    def __init__(self, config, domain_name):
        """
        :param config:
        """
        self.domain_name = domain_name
        self.config = config
        if config.data_mode == "imgs":
            img_root_folder = config.generator_datasets_root_dir
            dafsl_train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(config.ch_mean, config.ch_std_dev)

            ])
            dafsl_test_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(config.ch_mean, config.ch_std_dev)
            ])            
            # dafsl_transforms.append(ReshapeTransform((-1,)))
            print(os.path.join(img_root_folder, self.domain_name, "test"))
            self.src_dataset_train = datasets.ImageFolder(root=os.path.join(img_root_folder, self.domain_name, "finetune"),
                                                     transform=dafsl_train_transforms)
            self.src_dataset_test = datasets.ImageFolder(root=os.path.join(img_root_folder, self.domain_name, "test"),
                                                     transform=dafsl_test_transforms)
            print("size of dataset", len(self.src_dataset_test))
            finetune_train_sampler = RandomSampler(self.src_dataset_train, replacement=False)
            finetune_test_sampler = RandomSampler(self.src_dataset_test, replacement=False)
            self.train_loader = torch.utils.data.DataLoader(
                self.src_dataset_train, batch_size=self.config.batch_size, shuffle=False, sampler=finetune_train_sampler, num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
            self.test_loader = torch.utils.data.DataLoader(
                self.src_dataset_test , batch_size=self.config.batch_size, shuffle=False, sampler=finetune_test_sampler, num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
            print("length of  train loader", len(self.train_loader))
            print("length of  test loader", len(self.test_loader))

        elif config.data_mode == "download":
            raise NotImplementedError("This mode is not implemented YET")
        else:
            raise Exception(
                "Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.jpg'.format(
            self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=8,
                           padding=2,
                           normalize=True)
        print("############", img_epoch)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(
                self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir +
                        'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass
