import torch
from torch import nn
import os
import time
import warnings
import numpy as np
from tools import adjust_lr, EarlyStopping
from utils.metrics import metric
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

def train_ft(model, optimizer, device, train_loader, vali_loader, test_loader, path_ft, args):
    early_stopping = EarlyStopping(args.patience, verbose=True)
    criterion = nn.MSELoss()

    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=0.3,
                                        epochs=args.n_epochs_finetune,
                                        max_lr=args.lr)

    for epoch in range(args.n_epochs_finetune):
        start_time = time.time()
        model.train()
        train_loss = []

        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs[0], batch_y)

            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

        print(f'Epoch {epoch + 1} cost time: {time.time() - start_time}')

        train_loss_epoch = np.average(train_loss)
        if args.n_epochs_finetune > 1:
            vali_loss_epoch = vali_ft(model, vali_loader, criterion, device)
        else:
            vali_loss_epoch = 0.0
        test_loss_epoch = 0.0

        print('Epoch {} | train_loss {:.7f} | valid_loss {:.7f} | test_loss {:.7f}'.format(
            epoch + 1, train_loss_epoch, vali_loss_epoch, test_loss_epoch))

        early_stopping(vali_loss_epoch, model, path_ft)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_lr(optimizer, scheduler, epoch + 1, args)

    return


def vali_ft(model, vali_loader, criterion, device):
    model.eval()
    vali_loss = []
    with torch.no_grad():
        for (batch_x, batch_y) in vali_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            pred = outputs[0].detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            vali_loss.append(loss.item())

    model.train()

    return np.average(vali_loss)


def test_ft(test_loader, device, path_ft, settings_ft, model, settings_pt, dataset):
    preds = []
    trues = []

    print('loading model ...')
    best_model_path = path_ft + '/' + 'finetuned.pth'
    model.load_state_dict(torch.load(best_model_path))

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pred, _ = model(batch_x)

            pred = pred.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(batch_y)

    preds = np.array(preds)
    trues = np.array(trues)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    # result save
    if not os.path.exists(path_ft):
        os.makedirs(path_ft)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    # f = open("result_long_term_forecast_4.txt", 'a')
    f = open("./outputs/result_forecast/" + dataset + ".txt", 'a')
    f.write(settings_pt + ' | ' + settings_ft + '  \n')
    f.write('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()

    return
