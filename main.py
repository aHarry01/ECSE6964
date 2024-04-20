import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt

from utils import closest_subset_target_sum

# GLOBAL PARAMETERS
# Hyperparameters
num_clients = 30
num_rounds = 3
epochs = 5
batch_size = 10

optimal_data_per_client = [1000 for i in range(num_clients)] # optimal data for each client (mi*) if it were running only locally
marginal_cost = 0.00013 # global marginal cost of more data beyond optimal (in terms of amount of data)
epsilon = 0.000
k_global = 2
datasize_per_client = [x for x in range(900,1800,30)] #[900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800] # how much training data each client uses, note that for MNIST traindata.data.shape[0] = 60000

torch.manual_seed(104513) # remove randomness for reproducibility

# Flags to determine which graphs to plot
VERBOSE = True
PLOT_ACCURACY = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def client_update(client_model, optimizer, train_loader, epoch=5):
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cpu(), target.cpu()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def update_client_models(global_model, client_models, client_contribution_sizes, optimal_data_per_client):
    global_dict = global_model.state_dict()
    total_available_data = sum(client_contribution_sizes)
    if VERBOSE:
        print(f"Actual available data {total_available_data}")
    client_ideal_shaped_datasize = [None] * len(client_models)
    client_shaped_datasize = [None] * len(client_models)
    client_ideal_returned_accuracy = [None] * len(client_models)
    for (i,model) in enumerate(client_models):
        # accuracy shaping
        if (client_contribution_sizes[i] <= optimal_data_per_client[i]):
            # return the model trained on the clients own data - i.e. make no changes
            if VERBOSE:
                print(f"Client {i}: contributed datasize {client_contribution_sizes[i]}, accuracy shaped training datasize (ideal) {client_contribution_sizes[i]}")
            client_shaped_datasize[i] = client_contribution_sizes[i]
            client_ideal_shaped_datasize[i] = client_contribution_sizes[i]
            client_ideal_returned_accuracy[i]= 0
        else:
            # accuracy shaping - compute ideal size of the dataset for this client's contribution
            ideal_returned_accuracy = 1 - 2*sqrt(k_global/optimal_data_per_client[i]) + (marginal_cost+epsilon)*(client_contribution_sizes[i]-optimal_data_per_client[i]) #*(1-2*sqrt(k_global/(client_contribution_sizes[i]-optimal_data_per_client[i])))
            ideal_returned_accuracy = min(0.999, ideal_returned_accuracy)
            ideal_training_datasize = max(client_contribution_sizes[i], 1/((((1-ideal_returned_accuracy)/2)**2)/k_global))
            client_ideal_returned_accuracy[i] = ideal_returned_accuracy
            if VERBOSE:
                print(f"Client {i}: contributed datasize {client_contribution_sizes[i]}, accuracy shaped training datasize (ideal) {ideal_training_datasize}")
            client_ideal_shaped_datasize[i] = ideal_training_datasize

            # can't train on more data than all the clients gave us
            if (ideal_training_datasize >= total_available_data):
                model.load_state_dict(global_dict)
                client_shaped_datasize[i] = total_available_data
            else:
                # choose model parameter updates from each client to get a model trained on ideal_training_datasize amount of data
                client_indexes, client_datasizes = closest_subset_target_sum(client_contribution_sizes, ideal_training_datasize)
                client_shaped_datasize[i] = sum(client_datasizes)
                param_dict = global_model.state_dict()
                for k in param_dict.keys():
                    param_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in client_indexes], 0).mean(0)
                model.load_state_dict(param_dict)
    return client_shaped_datasize, client_ideal_returned_accuracy
            

def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)


def test(global_model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cpu(), target.cpu()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

if __name__ == "__main__":
    # Creating decentralized datasets
    traindata = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    traindata_split = torch.utils.data.random_split(traindata, datasize_per_client + [(traindata.data.shape[0] - sum(datasize_per_client))] )
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            ), batch_size=batch_size, shuffle=True)

    # Instantiate models and optimizers
    global_model = Net().cpu()
    client_models = [Net().cpu() for _ in range(num_clients)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

    # Running FL
    print("Initialization complete")
    client_model_accuracies = []
    client_model_datasizes = []
    client_ideal_returned_accuracies = []
    for r in range(num_rounds):
        print(f"Round {r}")
         
        # client update
        loss = 0
        for i in range(num_clients):
            loss += client_update(client_models[i], opt[i], train_loader[i], epoch=epochs)
            print(i)
        print("client update complete")
        
        # server aggregate
        server_aggregate(global_model, client_models)
        test_loss, acc = test(global_model, test_loader)
        print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_clients, test_loss, acc))

        client_shaped_datasize, client_ideal_returned_accuracy = update_client_models(global_model, client_models, datasize_per_client, optimal_data_per_client)
        client_model_datasizes.append(client_shaped_datasize)
        client_ideal_returned_accuracies.append(client_ideal_returned_accuracy)

        if PLOT_ACCURACY:
            print(client_shaped_datasize)
            client_model_accuracy = [None] * len(client_models)
            for (i,model) in enumerate(client_models):
                loss, client_model_accuracy[i] = test(model, test_loader)
            client_model_accuracies.append(client_model_accuracy)

    if PLOT_ACCURACY:
        for (i,d) in enumerate(client_model_datasizes):
            plt.plot(datasize_per_client, d, label=f"Round {i}", marker='o')
            
        plt.xlabel("Amount of data Contributed by Client")
        plt.ylabel("Amount of data used in Accuracy Shaped model")
        plt.title(f"Accuracy Shaped Datasize vs Client Datasize")
        plt.legend()
        plt.show()

        for (i,a) in enumerate(client_model_accuracies):
            plt.plot(datasize_per_client, a, label=f"Actual Accuracy Round {i}", marker='o')
        plt.xlabel("Amount of data Contributed by Client")
        plt.ylabel("Accuracy of the returned model")
        plt.title(f"Accuracy Shaping")
        plt.legend()
        plt.show()

        

