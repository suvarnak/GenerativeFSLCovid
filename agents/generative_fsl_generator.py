"""
Generative FSL  Main agent for generating the background knowledge
"""
import numpy as np
import os

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from agents.base import BaseAgent

from graphs.models.generative_fsl_cae_model import GenerativeFSL_CAEModel 
from datasets.generative_fsl_data_loader import GenerativeFSLDataLoader

from torchviz import make_dot
from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
from torchsummary import summary

cudnn.benchmark = True


class GenerativeFSLGeneratorAgent(BaseAgent):
    domain_name = 'cat'
    def __init__(self, config):
        super().__init__(config)
        print(torch.__version__)
        # define models
        self.model = GenerativeFSL_CAEModel()
        self.domain_name = "cat"
        # define loss
        self.loss = nn.MSELoss() #nn.NLLLoss()

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
        summary(self.model, input_size=(3, self.config.image_size,self.config.image_size))


        # define optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), alpha=0.99, lr=self.config.learning_rate, eps=1e-08,weight_decay=0, momentum=self.config.momentum)
				#optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.best_valid_loss = 0
        self.fixed_noise = Variable(torch.randn(self.config.batch_size, 3, self.config.image_size, self.config.image_size))
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='GenerativeFSL')

    def save_checkpoint(self, filename='gen_checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        encoder_state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.logger.info("Checkpoint saving  from '{}' at (epoch {}) at (iteration {})\n".format(self.config.checkpoint_dir, state['epoch'], state['iteration']))
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        shutil.copyfile(self.config.checkpoint_dir + filename, self.config.checkpoint_dir + str(state['epoch'])+ filename)
        # # If it is the best copy it to another file 'model_best.pth.tar'
        # if is_best:
        #     shutil.copyfile(self.config.checkpoint_dir + filename,
        #                     self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch'] + 1
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate_acrossdomains()
            else:
                self.train_alldomains()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
    

    def train_alldomains(self):
        """
        This function will the operator
        :return:
        """
        for domain_name in self.config.source_domains.split(','):
            self.domain_name = domain_name
            self.train(domain_name)
        

    def train_generativemodel(self, domain_name):
        self.logger.info("Training the generative models for {} domain".format(domain_name))
				# Model Loading from the latest checkpoint if not found start from scratch.
        domain_checkpoint_file = domain_name + self.config.checkpoint_file
        self.logger.info("LOADING gen model {}....................".format(domain_checkpoint_file))

        self.load_checkpoint(domain_checkpoint_file )
        self.data_loader = GenerativeFSLDataLoader(config=self.config, domain_name=domain_name)
        try: 
            if self.config.mode == 'test':
                self.validate(domain_name)
            else:
                self.train(domain_name)
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self,domain_name):
        """
        Main training function, with per-epoch model saving
        """
        self.logger.info("Training the generative models for {} domain".format(domain_name))
				# Model Loading from the latest checkpoint if not found start from scratch.
        domain_checkpoint_file = domain_name + self.config.checkpoint_file
        self.logger.info("LOADING {}....................".format(domain_checkpoint_file))

        self.load_checkpoint(domain_checkpoint_file )
        self.logger.info("$$$"+domain_name)
        self.data_loader = GenerativeFSLDataLoader(config=self.config, domain_name=domain_name)
        self.criterion = MSELoss()#BCE_KLDLoss(self.model)
        for epoch in range(self.current_epoch, self.current_epoch+self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch(domain_name)
            valid_loss = self.validate(domain_name)
            is_best = valid_loss > self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_loss
            filename= domain_name + 'genfsl_checkpoint.pth.tar'
            self.save_checkpoint(is_best=is_best,filename=filename)

    def train_one_epoch(self,domain_name):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        epoch_lossG = AverageMeter()
        for batch_idx, data in enumerate(self.data_loader.train_loader):
						# credit assignment
            self.optimizer.zero_grad()    # clear the gardients
            imgs, _ = data
            imgs = imgs.to(self.device)
            generated_imgs = self.model(imgs)
            #generated_imgs = generated_imgs[0]  
						#make_dot(generated_img[0])
            #make_dot(generated_imgs.mean(),params=dict(list(self.model.named_parameters()))).render("sourcemodel", format="png")
            #self.logger.info("Batch index"+ str(batch_idx))
            #self.logger.info("generated images " + list(generated_imgs.size()))
            #self.logger.info("input images " + list(imgs.size()))
            #print("..........................")
						# calculate loss
            loss = self.criterion(generated_imgs, imgs)
            loss.backward()
						# update model weights
            self.optimizer.step()
            epoch_lossG.update(loss.item())
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1
            self.summary_writer.add_scalar("epoch/Generator_loss_"+domain_name, epoch_lossG.val, self.current_iteration)
        
        #gen_out = self.model(self.fixed_noise)
        #out_img = self.data_loader.plot_samples_per_epoch(gen_out.data, self.current_iteration)
        #self.summary_writer.add_image('train/generated_image', out_img, self.current_iteration)
        self.visualize_one_epoch()
        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + " - Generator Loss-: " + str(epoch_lossG.val))

        self.logger.info('\nTrain set: Average loss: {:.4f}\n'.format(epoch_lossG.val))


    def visualize_one_epoch(self):
        """
        One epoch of visualizing
        :return:
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, data  in enumerate(self.data_loader.test_loader):
                testimgs, _ = data  #data.to(self.device
                testimgs = testimgs.to(self.device)                
                generated_testimgs = self.model(testimgs)
                #generated_testimgs = generated_testimgs[0]  				
								#make_dot(generated_img[0])
                print(list(generated_testimgs.size()))
                #print(list(testimgs.size()))
                #plt.figure()
                #img = testimgs[batch_idx]
                img = generated_testimgs #.reshape((generated_testimgs.size()[0], 3,224,224))
                #print(list(img.size()))
                #img  = img.permute(0,3,1,2)
                #print(list(img.size()))
                self.data_loader.plot_samples_per_epoch(img,self.current_epoch)
								#plt.imshow(img.numpy())
	      
    def validate(self,domain_name):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, data  in enumerate(self.data_loader.test_loader):
                testimgs, _ = data  #data.to(self.device)
                testimgs = testimgs.to(self.device)                
                generated_testimgs = self.model(testimgs)
								# calculate loss
                test_loss += self.criterion(generated_testimgs, testimgs)
                #test_loss += F.mse_loss(output[batch_idx], data[batch_idx], size_average=False).item()  # sum up batch loss
                #pred = target #output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                #correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        self.summary_writer.add_scalar("epoch/Generator_testloss_"+domain_name, test_loss, self.current_iteration)
        self.logger.info('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()