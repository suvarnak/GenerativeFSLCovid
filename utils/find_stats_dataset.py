from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse


from utils import config 


def compute_dataset_stats(config, domain_name):
		img_root_folder = config.generator_datasets_root_dir
		img_root_folder = os.path.join(img_root_folder, domain_name, "train")
		print("Image root folder used for calculating mean std", img_root_folder)
		dataset = datasets.ImageFolder(root=img_root_folder, transform=transforms.Compose([transforms.Resize(256),
																transforms.CenterCrop(224),
																transforms.ToTensor()]))
		loader = DataLoader(
				dataset,
				batch_size=64,
				num_workers=1,
				shuffle=False
		)

		mean = 0.
		std = 0.
		nb_samples = 0.
		for data in loader :
				data = data[0]
				batch_samples = data.size(0)
				data = data.view(batch_samples, data.size(1), -1)
				mean += data.mean(2).sum(0)
				std += data.std(2).sum(0)
				nb_samples += batch_samples

		mean /= nb_samples
		std /= nb_samples
		return mean,std


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    parsed_config = config.process_config(args.config)
    for domain_name in parsed_config.data_domains.split(','):
        print("Processing domain {} .............".format(domain_name))
        print(compute_dataset_stats(parsed_config,domain_name))
    print("done")
if __name__ == "__main__":
		main()