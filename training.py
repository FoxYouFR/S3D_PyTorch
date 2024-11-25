import numpy as np
import torch
from tqdm import tqdm
import wandb

def train(model, trainloader, validloader, optim, scheduler, criterion, config, device='cpu'):
    epochs = config['epochs']
    # wandb.init(
    #     project=config['project'],
    #     config={
    #         'architecture': config['arch'],
    #         'dataset': config['dataset'],
    #         'epochs': epochs,
    #         'device': device
    #     }
    # )

    model = model.to(device)
    valid_loss_min = np.Inf
    print(f"Starting learning on device {device}...")
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()
        for data, target in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            out = model(data)
            _, preds = torch.max(out, dim=1)
            train_acc += torch.sum(preds == target).item()
            loss = criterion(out, target)
            loss.backward()
            optim.step()
            scheduler.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(trainloader.dataset)
        train_acc /= len(trainloader.dataset)

        if validloader is None:
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'lr': scheduler.get_last_lr()[0]})

        else:
            model.eval()
            with torch.no_grad():
                for data, target in tqdm(validloader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                    data, target = data.to(device), target.to(device)
                    out = model(data)
                    _, preds = torch.max(out, dim=1)
                    valid_acc += torch.sum(preds == target).item()
                    loss = criterion(out, target)
                    valid_loss += loss.item() * data.size(0)

            valid_loss /= len(validloader.dataset)
            valid_acc /= len(validloader.dataset)

            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f} '
                  f'\tValidation Loss: {valid_loss:.6f} \tValidation Acc: {valid_acc:.6f}')
            wandb.log({
                'train_loss': train_loss, 'train_acc': train_acc,
                'valid_loss': valid_loss, 'valid_acc': valid_acc,
                'lr': scheduler.get_last_lr()[0]
            })
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss

    wandb.finish()