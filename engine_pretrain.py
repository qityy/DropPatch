# import fontTools.t1Lib
import torch
import os
import time
import numpy as np
import pickle
from utils.showcase import dm_showcase
from torch.optim import lr_scheduler
from tools import adjust_lr


def train_pt(model, optimizer, device, train_loader, vali_loader, path, args):
    print("start training...")
    best_score = np.inf

    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=0.3,
                                        epochs=args.n_epochs_pretrain,
                                        max_lr=args.lr)

    train_loss_epoch_lst = []
    val_loss_epoch_lst = []

    for epoch in range(args.n_epochs_pretrain):
        model.train()

        train_loss = []

        start_time = time.time()

        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = outputs[0]
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

        train_loss_epoch = np.average(train_loss)

        vali_loss_epoch = vali(model, vali_loader, device)

        print('Epoch {} \n| train loss: {:.4f} | valid loss: {:.4f}'
              .format(epoch + 1, train_loss_epoch, vali_loss_epoch))

        print('| cost time: {:.2f}'.format(time.time() - start_time))

        # print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        train_loss_epoch_lst.append(train_loss_epoch)
        val_loss_epoch_lst.append(vali_loss_epoch)

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model
        }

        if vali_loss_epoch < best_score:
            print(f'Validation loss decreased ({best_score:.6f} --> {vali_loss_epoch:.6f}).')
            best_score = vali_loss_epoch
            print('saving best model...')
            torch.save(save_dict, path + '/' + 'checkpoint.pth')

        adjust_lr(optimizer, scheduler, epoch + 1, args)

    best_model_path = path + '/checkpoint.pth'
    print(f'best pretrain model path: {best_model_path}')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    train_loss_epoch = np.array(train_loss_epoch_lst)
    val_loss_epoch = np.array(val_loss_epoch_lst)

    np.save('outputs/pretrain_loss/' + args.setting + '_train.npy', train_loss_epoch)
    np.save('outputs/pretrain_loss/' + args.setting + '_val.npy', val_loss_epoch)

    return model


def vali(model, vali_loader, device):
    model.eval()
    vali_loss = []
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)

            outputs = model(batch_x)
            loss = outputs[0]

            vali_loss.append(loss.item())

        vali_loss_epoch = np.average(vali_loss)

    model.train()

    return vali_loss_epoch


def test_pt(test_loader, device, model, path, setting_pt, patch_len, showcase):
    best_model_path = path + '/checkpoint.pth'
    print(f'best pretrain model path: {best_model_path}')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    print("start testing...")
    accs = []
    losses = []
    visual_outs = []
    attns = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            loss_pos, loss_smooth, acc, model_visual_outs = model(batch_x, test=True)

            if showcase == 1:
                if (i+1) % 1000 == 0:
                    # visual_out = show_pos_rec(batch_x, model_visual_outs, patch_len, stride)
                    visual_out = dm_showcase(model_visual_outs, batch_x, patch_len)
                    visual_outs.append(visual_out)

                    if 'attn' in model_visual_outs.keys():
                        attns.append(model_visual_outs['attn'])
                        # print(len(model_visual_outs['attn']), model_visual_outs['attn'][0].shape)

            accs.append(acc)
            losses.append(loss_pos.detach().cpu().numpy())

    accuracy = np.array(accs)
    loss_ce = np.array(losses)  # num_batch * bs * l * c

    accuracy_avg = np.average(accuracy)
    loss_ce_avg = np.average(loss_ce)

    print('accuracy:{}, loss_ce:{}'.format(accuracy_avg, loss_ce_avg))
    f = open('./outputs/result_pretrain_test/pretrain_test.txt', 'a')
    f.write(setting_pt + ' \n')
    f.write('test acc: {:.2f} loss: {:.2f}\n'.format(accuracy_avg*100, loss_ce_avg))
    f.close()

    if showcase == 1:
        print('saving visualization outputs...')

        with open('./outputs/visualization_results/' + setting_pt + '.pkl', 'wb') as f:
            pickle.dump(visual_outs, f)

    if 'attn' in model_visual_outs.keys():
        print('saving attn outputs...')
        torch.save(attns, './outputs/visualization_results/attn_' + setting_pt + '.pth')

    return

def check_hidden_representations(model, train_loader, test_loader, device, data_classification):
    model.eval()
    train_instances = []
    test_instances = []
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            hidden_reps = model(batch_x)

            bs, _, d = hidden_reps.shape
            random_indices = np.random.choice(bs, size=5, replace=False)
            selected_samples = hidden_reps[random_indices]
            for selected_sample in selected_samples:
                train_instances.append(selected_sample.cpu().numpy())  # each sample: 1 * n * d_model

        print(f'total train samples: {len(train_instances)}')

        random_indices_test = np.random.choice(len(test_loader), size=500, replace=False)
        for i, (batch_x, _) in enumerate(test_loader):
            if i not in random_indices_test:
                continue

            batch_x = batch_x.float().to(device)
            hidden_reps = model(batch_x)

            bs, _, d = hidden_reps.shape
            random_indices_channel = np.random.choice(bs, size=1, replace=False)
            selected_samples = hidden_reps[random_indices_channel]
            for selected_sample in selected_samples:
                test_instances.append(selected_sample.cpu().numpy())
        print(f'total test samples: {len(test_instances)}')

    train_instances = np.array(train_instances)
    test_instances = np.array(test_instances)

    np.save('./outputs/hidden_reps/train_{}.npy'.format(data_classification), train_instances)
    np.save('./outputs/hidden_reps/test_{}.npy'.format(data_classification), test_instances)

    return

def check_hidden_representations_cka(model, test_loader, device, data_classification, status):
    model.eval()
    test_instances = []

    with torch.no_grad():
        for i, (batch_x, _) in enumerate(test_loader):
            if i < 2000:

                batch_x = batch_x.float().to(device)
                hidden_reps = model(batch_x)

                test_instances.append(hidden_reps.cpu().numpy())  # each sample: 1 * n * d_model

        print(f'total train samples: {len(test_instances)}')

    test_instances = np.array(test_instances)

    np.save('./outputs/hidden_reps_cka/{}_test_{}.npy'.format(status, data_classification), test_instances)

    return

def large_pt(model, optimizer, device, train_loader, path, args):
    print("start training...")
    best_score = np.inf

    lambda_smooth = args.lambda_smooth
    # if args.weight_decay == 2:
    #     lambda_smooth = 0.0

    # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #                                                              T_max=args.n_epochs_pretrain)
    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=0.3,
                                        epochs=args.n_epochs_pretrain,
                                        max_lr=args.lr)

    for epoch in range(args.n_epochs_pretrain):
        start_time = time.time()
        model.train()

        train_loss = []
        accs = []
        train_loss_smooth = []
        train_loss_pos = []

        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss_pos = outputs[0]
            loss_smooth = outputs[1]
            acc = outputs[2]

            loss = (1 - lambda_smooth) * loss_pos + lambda_smooth * loss_smooth

            loss.backward()
            train_loss.append(loss.item())
            accs.append(acc)
            train_loss_pos.append(loss_pos.item())
            train_loss_smooth.append(loss_smooth.item())
            optimizer.step()
            scheduler.step()

        train_loss_epoch = np.average(train_loss)
        train_acc_epoch = np.average(accs)
        train_loss_pos_epoch = np.average(train_loss_pos)
        train_loss_smooth_epoch = np.average(train_loss_smooth)

        # vali_loss_epoch, vali_acc_epoch, vali_loss_pos_epoch, vali_loss_smooth_epoch = vali(model, vali_loader, device, lambda_smooth)

        print('Epoch {} \n| train loss: {:.4f} | train loss mse: {:.4f} | train loss smooth: {:.4f}'
              .format(epoch + 1, train_loss_epoch, train_loss_pos_epoch, train_loss_smooth_epoch))
        # print('| valid loss: {:.4f} | valid loss mse: {:.4f} | valid loss smooth: {:.4f}'
        #       .format(vali_loss_epoch, vali_loss_pos_epoch, vali_loss_smooth_epoch))

        print('| cost time: {:.2f}'.format(time.time() - start_time))

        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model
        }

        torch.save(save_dict, path + '/' + 'checkpoint.pth')
        adjust_lr(optimizer, scheduler, epoch + 1, args)

    best_model_path = path + '/checkpoint.pth'
    print(f'best pretrain model path: {best_model_path}')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    return model
