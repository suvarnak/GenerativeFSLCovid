"""
Generative FSL  Main agent for generating the background knowledge
"""
import csv
import os
import sys
import random
import shutil

import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets.balanced_target_data_loader import BalancedTargetDataLoader
from datasets.target_data_loader import TargetDataLoader
from graphs.models.concept_discriminator import EncoderModel
from graphs.models.generative_fsl_cae_model import GenerativeFSL_CAEModel
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MSELoss
from torch.optim import Adam
from torchsummary import summary
from torchviz import make_dot
from tqdm import tqdm
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

from agents.base import BaseAgent

cudnn.benchmark = True


class RandomFSLDiscriminatorAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        print(torch.__version__)
        # define models
        self.gen_model = GenerativeFSL_CAEModel()
        self.model = EncoderModel(self.config)
        # define loss
        self.loss = nn.MSELoss()  # nn.NLLLoss()

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            torch.cuda.manual_seed_all(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            random.seed(self.manual_seed)
            torch.cuda.manual_seed_all(self.manual_seed)
            np.random.seed(self.manual_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.logger.info("Program will run on *****CPU*****\n")
        summary(self.model, input_size=(
            3, self.config.image_size, self.config.image_size))

        # define optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(
        ), alpha=0.99, lr=self.config.learning_rate, eps=1e-08, weight_decay=0, momentum=self.config.momentum)
        # optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.best_valid_loss = sys.maxsize
        self.fixed_noise = Variable(torch.randn(
            self.config.batch_size, 3, self.config.image_size, self.config.image_size))
        # Summary Writer
        self.summary_writer = SummaryWriter(
            log_dir=self.config.summary_dir, comment='FinetuneFSL')


    def save_checkpoint(self, filename='finetuned_checkpoint.pth.tar', is_best=False):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        domain_checkpoint_file = filename
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

        self.logger.info("Checkpoint saving  from '{}' at (epoch {}) at (iteration {})\n".format(
            self.config.checkpoint_dir, state['epoch'], state['iteration']))
        # Save the state
        torch.save(state, self.config.checkpoint_dir + domain_checkpoint_file)
        shutil.copyfile(self.config.checkpoint_dir + domain_checkpoint_file,
                        self.config.checkpoint_dir + str(state['epoch']) + domain_checkpoint_file)

        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + domain_checkpoint_file,
                            self.config.checkpoint_dir + 'Best_'+ domain_checkpoint_file )

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info(
                "******Loading checkpoint '{}' from dir {}".format(filename, self.config.checkpoint_dir))
            checkpoint = torch.load(filename)
            self.logger.info("********Loaded checkpoint '{}'".format(filename))
            self.current_epoch = checkpoint['epoch'] + 1
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def set_parameter_requires_grad(self, model, feature_extract):
        print("############Setting grad")
        if feature_extract:
            lt = 2
            cntr = 0
            for child in model.children():
                print("child", child)
                cntr += 1
                if cntr < lt:
                    for param in child.parameters():
                        param.requires_grad = False

    def load_source_model(self):
        try:
            domain_name = self.config.source_domain
            self.logger.info(
                "******Loading source model for domain '{}'".format(domain_name))
            filename = os.path.join(
                "model_repo", domain_name+"genfsl_checkpoint.pth.tar")
            checkpoint = torch.load(filename)
            self.logger.info("********Loaded checkpoint '{}'".format(filename))
            self.current_epoch = 0
            self.current_iteration = 0
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
            self.set_parameter_requires_grad(self.model, feature_extract=True)
            return True
        except OSError as e:
            self.logger.info(
                "No model checkpoint {} exists for source domain {}. Skipping...".format(filename,domain_name))
            return False

    def load_model(self, domain_name):
        try:
            self.logger.info(
                "*Loading  finetuned model  for domain '{}' for testing only".format(domain_name))
            filename = os.path.join(
                "tuned_model_repo", self.config.tuned_model_name)
            checkpoint = torch.load(filename)
            self.logger.info("********Loaded checkpoint '{}'".format(filename))
            self.current_epoch = 0
            self.current_iteration = 0
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
            return True
        except OSError as e:
            self.logger.info(
                "No model checkpoint exists for target domain {}. Skipping...".format(domain_name))
            return False

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate_target_domain()
            else:
                self.finetune_target_domain()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def finetune_target_domain(self):
        """
        This function will the operator
        :return:
        """
        domain_name = self.config.target_domain
        for i in range(0,10):
            domain_name = self.config.target_domain + "_" + str(i)
            print(domain_name)
            self.finetune_model(domain_name)

    def finetune_model(self, domain_name):
        self.logger.info(
            "Fine-tuning.....Target {}, Source {} ".format(domain_name, self.config.source_domain))
        try:
            if self.load_source_model():
                self.data_loader = TargetDataLoader(config=self.config, domain_name=domain_name)
                self.train(domain_name)
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def validate_target_domain(self):
        """
        This function will the operator
        :return:
        """
        domain_name = self.config.source_domain
        self.test_model(domain_name)

    def test_model(self, domain_name):
        self.logger.info(
            "Testing.....Source {}, Target {} ".format(domain_name, self.config.target_domain))
        try:
            if self.load_model(self.config.target_domain):
                self.data_loader = TargetDataLoader(config=self.config)
                with open(self.config.results_file_name, mode='a+') as csv_file:
                    fieldnames = ['Threshold','Confusion_Matrix', 'Sensitivity', 'Specificity', 'F1', 'Accuracy']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    if self.config.thresholding == False :
                        row,_ = self.validate(0.5)
                        writer.writerow(row)
                    else:
                        for i in np.linspace(0,1,11):
                            row,_ = self.validate(threshold=i)
                            writer.writerow(row)
                    csv_file.close()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def save_validate(self, domain_name):
        self.logger.info("RANDOM validation step  for {} domain".format(domain_name))
        with open(domain_name+".csv", mode='a+') as csv_file:
            fieldnames = ['Threshold','Confusion_Matrix', 'Sensitivity', 'Specificity', 'F1', 'Accuracy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                row,valid_loss = self.validate(threshold=threshold)
                writer.writerow(row)
            csv_file.close()
            return row,valid_loss

    def train(self, domain_name):
        """
        Main training function, with per-epoch model saving
        """
        summary(self.model, input_size=(
            3, self.config.image_size, self.config.image_size))
        # weight=class_weights)  # MSELoss()#BCE_KLDLoss(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.logger.info("Finetuning the generative models for {} domain".format(domain_name))
				# Model Loading from the latest checkpoint if not found start from scratch.
        domain_checkpoint_file = domain_name + self.config.checkpoint_file
        self.logger.info("LOADING {}....................".format(domain_checkpoint_file))
        #self.best_valid_loss = sys.maxsize
        self.load_checkpoint(domain_checkpoint_file )
        for epoch in range(self.current_epoch, self.current_epoch+self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch(domain_name)
            # _,valid_loss = self.validate(0.5)
            # is_best = valid_loss < self.best_valid_loss
            # self.logger.info("Best validation loss {} for epoch {} ".format(self.best_valid_loss,epoch))
            # if is_best:
            #     self.best_valid_loss = valid_loss
            #     self.save_checkpoint(filename=domain_checkpoint_file,is_best=is_best)
        self.save_checkpoint(filename=domain_checkpoint_file,is_best=False)
        #_,valid_loss = self.save_validate(domain_name)

    def train_one_epoch(self, domain_name):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        epoch_lossD = AverageMeter()
        for batch_idx, data in enumerate(self.data_loader.train_loader):
            # credit assignment
            self.optimizer.zero_grad()    # clear the gardients
            imgs, labels = data
            imgs = imgs.to(self.device)
            predicted_labels = self.model(imgs)
            loss = self.criterion(predicted_labels, labels)
            loss.backward()
            # update model weights
            self.optimizer.step()
            epoch_lossD.update(loss.item())
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Finetune Epoch: {} [{}/{} ({:.0f}%)] Loss: {:6f}'.format(
                    self.current_epoch, batch_idx *
                    len(data), len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader.dataset), loss.item()))
            self.current_iteration += 1
            self.summary_writer.add_scalar(
                "epoch/Finetune_Training_Loss_"+domain_name, epoch_lossD.val, self.current_iteration)

        # self.visualize_one_epoch()

        self.logger.info("Finetuning at epoch-" + str(self.current_epoch) +
                         " | " + " - Finetuning Loss-: " + str(epoch_lossD.val))

    def visualize_one_epoch(self):
        """
        One epoch of visualizing
        :return:
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader.test_loader):
                testimgs, predicted_labels = data  # data.to(self.device
                testimgs = testimgs.to(self.device)
                predicted_labels = self.model(testimgs)
                #generated_testimgs = generated_testimgs[0]
                # make_dot(generated_img[0])
                print(list(predicted_labels.size()))
                # print(list(testimgs.size()))
                # plt.figure()
                #img = testimgs[batch_idx]
                # img = generated_testimgs #.reshape((generated_testimgs.size()[0], 3,224,224))
                # print(list(img.size()))
                #img  = img.permute(0,3,1,2)
                # print(list(img.size()))
                # self.data_loader.plot_samples_per_epoch_with_labels(img,self.current_epoch,labels=predicted_labels)
                # plt.imshow(img.numpy())

    def add_pr_curve_tensorboard(self, class_index, test_probs, test_preds, global_step=0):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]
        self.summary_writer.add_pr_curve('PR for Covid prediction',
                                         tensorboard_preds,
                                         tensorboard_probs,
                                         global_step=global_step)

    def validate(self,threshold=0.5):
        """
        One cycle of model validation
        :return:
        """
        self.criterion = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss = 0
        correct = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader.test_loader):
                images, labels = data  # .to(self.device)
                labels_list = [element.item() for element in labels.flatten()]
                y_true_batch = labels_list
                output = self.model(images)  # [B,2]
                #print("Batch idx{} and size{}".format(batch_idx,len(labels_list)))
                #print(output)
                # converting the output layer values into labels 0 or one based on threshold
                sm = torch.nn.Softmax(1) # constrained probabilitites
                output = sm(output)
                #print("after softmax",output)
                #thresh = torch.nn.Threshold(threshold,0,False)
                thresholded_output =  output > threshold #thresh(output)
                y_pred_batch=[]
                #print("after thresholding",thresholded_output)
                output_max_value = torch.max(thresholded_output, 1)
                #print("gadbad",output_max_value)
                y_pred_batch =  output_max_value[1]
                #print("matching in batch", len([y_pred_batch == y_true]))
                #print("Sample pred", y_pred_batch[0], len(y_pred_batch))
                y_true.extend(y_true_batch)
                y_pred.extend(y_pred_batch)
                #if batch_idx == 0 :
                #    break
        #print(len(y_true),"%%%%",len(y_pred))
        #print("Threshold",threshold)
        tn, fp, fn, tp  = sklearn.metrics.confusion_matrix(y_true,y_pred).ravel()
        cf = sklearn.metrics.confusion_matrix(y_true, y_pred)
        #print("CF", cf)        
        #print("confusion matrix ",tn,fp,fn,tp)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        p = sklearn.metrics.precision_score(y_true,y_pred)    
        #print("PRECISION",p)
        #print("computed PRECISION",tp/(tp+fp))        
        r = sklearn.metrics.recall_score(y_true,y_pred)    
        #print("recall", r)
        #print("computed recall",tp/(tp+fn))                
        f1 = sklearn.metrics.f1_score(y_true,y_pred,average="binary")    
        #print("F1",f1)
        acc = sklearn.metrics.accuracy_score(y_true,y_pred)
        #print("acc", acc)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=2)
        #print("fpr {},tpr {}, thresholds {}".format(fpr,tpr,thresholds))
        print("sensitivity {},specificity {}".format(sensitivity,specificity))
        #print("auc for covid class ", sklearn.metrics.auc(fpr, tpr))
        #test_loss /= len(self.data_loader.test_loader.dataset)
        #self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    test_loss, correct, len(self.data_loader.test_loader.dataset),
        #    100. * acc))
        #fieldnames = ['threhhold', 'Sensitivity', 'Specificity', 'F1', 'Accuracy']
        results = {'Threshold':threshold, 'Confusion_Matrix':cf,'Sensitivity':sensitivity, 'Specificity':specificity, 'F1':f1, 'Accuracy':acc} 
        print("Results {} for domain".format(results))
        #"" + str(threshold) +"," + str(r) +"," + str(p) +"," + str(f1) +"," + str(acc)
        return results,test_loss
        




    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info(
            "Please wait while finalizing the operation.. Thank you")
        #self.save_checkpoint()
        self.summary_writer.export_scalars_to_json(
            "{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        # self.data_loader.finalize()
