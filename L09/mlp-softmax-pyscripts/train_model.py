# imports from helper.py
from helper import get_dataloaders_mnist, set_all_seeds, set_deterministic
from helper import compute_accuracy, plot_training_loss, plot_accuracy

# standard library
import argparse
import logging
import os
import time

# installed libraries
import torch
from torchvision import transforms
import yaml  # conda install pyyaml


parser = argparse.ArgumentParser()
parser.add_argument('--settings_path',
                    type=str,
                    required=True)
parser.add_argument('--results_path',
                    type=str,
                    required=True)
args = parser.parse_args()
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
with open(args.settings_path) as file:
    SETTINGS = yaml.load(file, Loader=yaml.FullLoader)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logpath = os.path.join(args.results_path, 'training.log')
logger.addHandler(logging.FileHandler(logpath, 'a'))
print = logger.info


print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    device = torch.device(f'cuda:{SETTINGS["cuda device"]}')
else:
    device = torch.device('cpu')
print(f'Using {device}')

set_all_seeds(SETTINGS['random seed'])
set_deterministic()

train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    train_transforms=transforms.ToTensor(),
    test_transforms=transforms.ToTensor(),
    batch_size=SETTINGS['batch size'],
    num_workers=SETTINGS['num workers'],
    validation_fraction=SETTINGS['validation fraction'])


##########################
# ## MODEL
##########################

class MLP(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, num_hidden),  # Hidden Layer
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_hidden, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.classifier(x)


model = MLP(num_features=SETTINGS['input size'],
            num_hidden=SETTINGS['hidden layer size'],
            num_classes=SETTINGS['num class labels'])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning rate'])


start_time = time.time()
minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
for epoch in range(SETTINGS['num epochs']):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(device)
        targets = targets.to(device)

        # ## FORWARD AND BACK PROP
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()

        loss.backward()

        # ## UPDATE MODEL PARAMETERS
        optimizer.step()

        # ## LOGGING
        minibatch_loss_list.append(loss.item())
        if not batch_idx % 50:
            print(f'Epoch: {epoch+1:03d}/{SETTINGS["num epochs"]:03d} '
                  f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                  f'| Loss: {loss:.4f}')

    model.eval()
    with torch.no_grad():  # save memory during inference
        train_acc = compute_accuracy(model, train_loader, device=device)
        valid_acc = compute_accuracy(model, valid_loader, device=device)
        print(f'Epoch: {epoch+1:03d}/{SETTINGS["num epochs"]:03d} '
              f'| Train: {train_acc :.2f}% '
              f'| Validation: {valid_acc :.2f}%')
        train_acc_list.append(train_acc.item())
        valid_acc_list.append(valid_acc.item())

    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

test_acc = compute_accuracy(model, test_loader, device=device)
print(f'Test accuracy {test_acc :.2f}%')

# ######### MAKE PLOTS ######
plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=SETTINGS['num epochs'],
                   iter_per_epoch=len(train_loader),
                   results_dir=args.results_path)
plot_accuracy(train_acc_list=train_acc_list,
              valid_acc_list=valid_acc_list,
              results_dir=args.results_path)

results_dict = {'train accuracies': train_acc_list,
                'validation accuracies': valid_acc_list,
                'test accuracy': test_acc.item(),
                'settings': SETTINGS}

results_path = os.path.join(args.results_path, 'results_dict.yaml')
with open(results_path, 'w') as file:
    yaml.dump(results_dict, file)
