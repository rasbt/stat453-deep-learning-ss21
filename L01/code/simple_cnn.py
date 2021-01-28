# imports from helper.py and setting.py
from settings import SETTINGS
from helper import get_dataloaders_mnist, set_all_seeds, set_deterministic
from helper import compute_accuracy, plot_training_loss, plot_accuracy

# standard library
import time

# installed libraries
import torch
from torchvision import transforms

print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    device = torch.device(f'cuda:{SETTINGS["cuda device"]}')
else:
    device = torch.device('cpu')
print(f'Using {device}')

set_all_seeds(SETTINGS['random seed'])
set_deterministic()

resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])

train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    train_transforms=resize_transform,
    test_transforms=resize_transform,
    batch_size=SETTINGS['batch size'],
    num_workers=SETTINGS['num workers'],
    validation_fraction=SETTINGS['validation fraction'])


##########################
# ## MODEL
##########################

class LeNet5(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16*5*5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = torch.nn.functional.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(SETTINGS['num class labels'])
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
        logits, probas = model(features)
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
                   iter_per_epoch=len(train_loader))
plot_accuracy(train_acc_list=train_acc_list, valid_acc_list=valid_acc_list)

