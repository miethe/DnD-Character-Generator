import logging

import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config import *
from data import DnDCharacterNameDataset, Vocabulary, OneHot, Genders, Races, ToTensor
from model import RNNCellModel, RNNLayerModel
from logger import Logger
from utils import save_model


class Trainer:
    """
    Base Trainer class that describes basic set attributes and methods needed to run model training.

    Every subclass needs to implement following methods:
        - init_dataset
        - init_loader
        - init_model
        - init_criterion
        - init_optimizer
    """
    def __init__(self, args, kwargs, root_dir, hidden_size, lr, epochs, batch_size, device, logfile, verbose=1):
        self.root_dir = kwargs['root_dir']
        self.device = kwargs['device']
        self.verbose = kwargs['verbose']
        self.logfile = kwargs['logfile']

        # Training params
        self.lr = kwargs['lr']
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']

        # Model params
        self.hidden_size = kwargs['hidden_size']

        # Data params
        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()

        # Initialization
        self.dataset = self.init_dataset()
        self.train_loder = self.init_loader()
        self.model = self.init_model()
        self.criterion = self.init_criterion()
        self.optimizer = self.init_optimizer()

        # Initialize logging
        self.logger = Logger(os.path.join(PROJECT_ROOT, self.logfile))

    def init_dataset(self):
        raise NotImplementedError

    def init_loader(self):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def init_criterion(self):
        raise NotImplementedError

    def init_optimizer(self):
        raise NotImplementedError

    def run_train_loop(self):
        raise NotImplementedError


class RNNCellTrainer(Trainer):
    """
    Trainer class for training the LSTMCell model (RNNCellModel). Defines methods for:
        - Initializing dataset
        - Initializing data loader
        - Initializing model
        - Initializing criterion
        - Initializing optimizer
    """
    def __init__(self, args, kwargs, root_dir='./data',
                 hidden_size=128,
                 lr=0.0005,
                 epochs=100,
                 batch_size=512,
                 device='gpu',
                 logfile='train_loss.log',
                 verbose=1):
        super().__init__(args, kwargs, root_dir, hidden_size, lr, epochs, batch_size, device, logfile, verbose)

    def init_dataset(self):
        return DnDCharacterNameDataset(root_dir=self.root_dir,
                                       name_transform=Compose([self.vocab, OneHot(self.vocab.size), ToTensor()]),
                                       race_transform=Compose([self.races, OneHot(self.races.size), ToTensor()]),
                                       gender_transform=Compose([self.genders, OneHot(self.genders.size), ToTensor()]),
                                       target_transform=Compose([self.vocab, ToTensor()]))

    def init_loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: self.dataset.collate_fn(batch))

    def init_model(self):
        model = RNNCellModel(input_size=self.vocab.size + self.races.size + self.genders.size,
                             hidden_size=self.hidden_size,
                             output_size=self.vocab.size)
        model.to(self.device)
        return model

    def init_criterion(self):
        """
        Due to variable sequence variable length, output tensor will contain -1 value. Time-steps that contain -1
        value as target (y) will not be included in loss function
        """
        return nn.CrossEntropyLoss(ignore_index=-1)

    def init_optimizer(self):
        return RMSprop(self.model.parameters())

    def run_train_loop(self):
        print("Started training!")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            running_loss = 0
            for inputs, targets, _ in self.train_loder:
                batch_size = inputs.shape[1]

                inputs = inputs.to(self.device)  # shape: [T, B, *]
                targets = targets.to(self.device)  # shape: [T, B]

                loss = 0
                hx, cx = self.model.init_states(batch_size=batch_size, device=self.device)

                # Iterate over time-steps and add loss
                for input, target in zip(inputs, targets):
                    output, hx, cx = self.model(input, hx, cx)
                    loss += self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Logging and printing
            epoch_loss = running_loss / len(self.train_loder)
            if self.logfile:
                self.logger.log("Epoch: {}, Loss: {}".format(epoch+1, epoch_loss))

            if self.verbose == 1:
                print("Epoch: {}, Loss {:.4f}".format(epoch+1, epoch_loss))

            # Save model on specific epochs
            if epoch + 1 in (1, 5, 10, 15) or epoch % 25 == 0:
                save_model(self.model, "rnn_cell_epoch_{}.pt".format(epoch+1))

        print("Finished training!")


class RNNLayerTrainer(Trainer):
    """
    Trainer class for training the LSTMLayer model (RNNLayerModel). Defines methods for:
        - Initializing dataset
        - Initializing data loader
        - Initializing model
        - Initializing criterion
        - Initializing optimizer
    """
    def __init__(self, args, kwargs, root_dir='./data',
                 hidden_size=128,
                 lr=0.0005,
                 epochs=100,
                 batch_size=512,
                 device='cpu',
                 logfile='train_loss.log',
                 verbose=1):
        super().__init__(args, kwargs, root_dir, hidden_size, lr, epochs, batch_size, device, logfile, verbose)

    def init_dataset(self):
        return DnDCharacterNameDataset(root_dir=self.root_dir,
                                       name_transform=Compose([self.vocab, OneHot(self.vocab.size), ToTensor()]),
                                       race_transform=Compose([self.races, OneHot(self.races.size), ToTensor()]),
                                       gender_transform=Compose([self.genders, OneHot(self.genders.size), ToTensor()]),
                                       target_transform=Compose([self.vocab, ToTensor()]))

    def init_loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: self.dataset.collate_fn(batch))

    def init_model(self):
        model = RNNLayerModel(input_size=self.vocab.size + self.races.size + self.genders.size,
                              hidden_size=self.hidden_size,
                              output_size=self.vocab.size)
        model.to(self.device)
        return model

    def init_criterion(self):
        """
        Due to variable sequence variable length, output tensor will contain -1 value. Time-steps that contain -1
        value as target (y) will not be included in loss function
        """
        return nn.CrossEntropyLoss(ignore_index=-1)

    def init_optimizer(self):
        return RMSprop(self.model.parameters())

    def run_train_loop(self):
        print("Started training with %s epochs!" % self.epochs)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            running_loss = 0
            for inputs, targets, lengths in self.train_loder:
                batch_size = inputs.shape[1]

                inputs = inputs.to(self.device)  # shape: [T, B, *]
                targets = targets.to(self.device)  # shape: [T, B]

                h0, c0 = self.model.init_states(batch_size=batch_size, device=self.device)
                output, hx, cx = self.model(inputs, h0, c0, lengths)

                loss = self.criterion(output.view(-1, output.shape[-1]), targets.view(-1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Logging and printing
            epoch_loss = running_loss / len(self.train_loder)
            if self.logfile:
                self.logger.log("Epoch: {}, Loss: {}".format(epoch+1, epoch_loss))

            if self.verbose == 1:
                print("Epoch: {}, Loss {:.4f}".format(epoch+1, epoch_loss))

            # Save model on specific epochs
            if epoch+1 in (1, 5, 10, 15) or (epoch+1) % 25 == 0:
                save_model(self.model, "rnn_layer_epoch_{}.pt".format(epoch+1))

        print("Finished training!")


class TrainerFactory:
    factory = {
        "cell": RNNCellTrainer,
        "layer": RNNLayerTrainer
    }

    @classmethod
    def get_trainer(cls, trainer_type, *args, **kwargs):
        return cls.factory[trainer_type](args, kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=300)
    parser.add_argument("-bs", "--batch_size", default=128)
    parser.add_argument("-hs", "--hidden_size", default=128)
    parser.add_argument("-lr", "--learning_rate", default=0.0001)
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-t", "--type", default="layer", choices=["cell", "layer"])
    parser.add_argument("-l", "--logfile", default="train_loss.log")
    parser.add_argument("-v", "--verbose", default=1)
    args = parser.parse_args()

    trainer = TrainerFactory.get_trainer(trainer_type=args.type,
                                         root_dir='./data',
                                         epochs=int(args.epochs),
                                         batch_size=args.batch_size,
                                         hidden_size=args.hidden_size,
                                         lr=args.learning_rate,
                                         device=args.device,
                                         logfile=args.logfile,
                                         verbose=args.verbose)
    trainer.run_train_loop()
