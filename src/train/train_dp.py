"""Attack Scenarios definitions"""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
#from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

from src.datasets.dataset_utils import DatasetSplit
from src.train.dp_utils import gaussian_noise
from src.utils import model_replacement, attack_test_visual_pattern


class LocalUpdateDifferentialPrivacy(object):
    def __init__(self, args, dataset, idxs, q, logger, test_dataset, clip, T, eps, delta):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.clip = clip
        self.q = q
        self.T = T
        self.clip = clip
        self.eps = eps
        self.delta = delta
        self.test_dataset = test_dataset
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.sigma = compute_noise(1, self.q, self.eps, self.T, self.delta, 1e-5)
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        train_loader = DataLoader(DatasetSplit(dataset, idxs_train),
                                  batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, idxs_val),
                                  batch_size=int(len(idxs_val) / 10), shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=int(len(idxs_test) / 10), shuffle=False)
        # print('batch_size', int(len(idxs_test) / 10))
        return train_loader, valid_loader, test_loader

    def add_visual_pattern(self, input):
        pattern = ((1, 3), (1, 5), (3, 1), (5, 1), (5, 3), (3, 5), (5, 5), (1, 1), (3, 3), (5, 5))
        for x, y in pattern:
            input[0][x][y] = 255
        return input

    def alter_data_set(self, images, targets):
        print('hello rania')
        return
        """
            for idx, image in enumerate(images):
            images[idx] = self.add_visual_pattern(image)
            targets[idx] = 5
        return images, targets
        """


    def update_weights(self, model, global_round, modelReplacement=False, attack=None):
        # Set mode to train model
        model.train()
        epoch_loss = []
        eps = self.args.eps
        x = copy.deepcopy(model.state_dict())
        idx = np.where(np.random.rand(len(self.dataset[:][0])) < self.q)[0]
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                if attack:
                    images, labels = self.alter_data_set(images, labels)

                model.zero_grad()
                # print(images.shape)
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad / len(idx)
                    self.model.zero_grad()
                for name, param in self.model.named_parameters():
                    param.grad = clipped_grads[name]
                optimizer.step()

                if attack and self.args.pgd:
                    x_adv = copy.deepcopy(model.state_dict())
                    for key in x_adv.keys():
                        x_adv[key] = torch.max(torch.min(x_adv[key], x[key] + eps), x[key] - eps)
                    model.load_state_dict(x_adv)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        sensitivity = 2.0 * self.lr * self.clip / len(idx) + (self.E - 1) * 2 * self.lr * self.clip
        new_param = copy.deepcopy(self.model.state_dict())
        for name in new_param:
            new_param[name] = torch.zeros(new_param[name].shape).to(self.device)
            new_param[name] += 1.0 * self.model.state_dict()[name]
            new_param[name] += gaussian_noise(self.model.state_dict()[name].shape, sensitivity,
                                              self.sigma, device=self.device)
        self.model.load_state_dict(copy.deepcopy(new_param))
        if modelReplacement:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_replacement(self, model, global_round, attack=False):
        # Set mode to train model
        model.train()
        epoch_loss = []
        eps = 0.1
        x = copy.deepcopy(model.state_dict())
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if attack:
                    images, labels = self.alter_data_set(images, labels)

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                # labels = labels.type(torch.cuda.FloatTensor)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                if attack:
                    x_adv = copy.deepcopy(model.state_dict())
                    for key in x_adv.keys():
                        x_adv[key] = torch.max(torch.min(x_adv[key], x[key] + eps), x[key] - eps)
                    model.load_state_dict(x_adv)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if attack:
            print('test attack after attacker', attack_test_visual_pattern(self.test_dataset, model))
        if attack:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        # return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss
