import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import logging

def train(model, trainloader, validloader, optim, scheduler, criterion, config, device='cuda', skip_start_log=False):
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

    logging.basicConfig(
        filename='train.log',
        encoding='utf-8',
        filemode='w',
        format='{asctime} - {levelname} - {message}',
        style='{',
        datefmt='%Y-%m-%d %H:%M',
        level=logging.DEBUG
    )

    model = model.to(device)
    scaler = GradScaler(device=device)

    valid_loss_min = np.Inf
    print(f"Starting learning on device {device}...")
    logging.info(f"Starting learning on device {device}...")
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()
        for data, target in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            with autocast(device_type=device):
                out = model(data)
                print(out)
                print(target)
                loss = criterion(out, target)
                if torch.isnan(loss):
                    logging.warning('A LOSS IS NAN')
                    logging.debug(f'Out: {out}')
                    logging.debug(f'Target: {target}')
                    logging.debug(f'Loss: {loss}')
            
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optim)
            scaler.update()

            _, preds = torch.max(out, dim=1)
            train_loss += loss.item() * data.size(0)
            train_acc += torch.sum(preds == target).item()
            # if not (epoch == 0 and skip_start_log):
                # wandb.log({'train_loss': loss.item()})

        #scheduler.step() # Cut the scheduler for now
        train_loss /= len(trainloader.dataset)
        train_acc /= len(trainloader.dataset)
        wandb.log({'train_acc': train_acc, 'lr': scheduler.get_last_lr()[0]})

        if validloader is None:
            print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')
            logging.info(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')

        else:
            model.eval()
            with torch.no_grad():
                for data, target in tqdm(validloader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                    data, target = data.to(device), target.to(device)
                    out = model(data)
                    _, preds = torch.max(out, dim=1)
                    loss = criterion(out, target)
                    valid_loss += loss.item() * data.size(0)
                    valid_acc += torch.sum(preds == target).item()

            valid_loss /= len(validloader.dataset)
            valid_acc /= len(validloader.dataset)
            wandb.log({'valid_loss': valid_loss, 'valid_acc': valid_acc})

            print(f'Epoch: {epoch+1} \tValidation Loss: {valid_loss:.6f} \tValidation Acc: {valid_acc:.6f}')
            logging.info(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss

    wandb.finish()