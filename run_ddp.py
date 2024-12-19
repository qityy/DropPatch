import argparse
import torch
from torch import optim
import numpy as np
import random
import os
from models import dm
from data_provider.data_factory import data_provider
from engine_pretrain_ddp import train_pt, test_pt, large_pt
from accelerate import Accelerator
from torch.optim import lr_scheduler


def main():
    parser = argparse.ArgumentParser()
    # GENERAL
    parser.add_argument('--mode', type=str, default='pt', help='options: [pt, ft, randinit, eval]')
    parser.add_argument('--model', type=str, default='dm')
    parser.add_argument('--data', type=str, default='Large', help='dataset name')
    parser.add_argument('--data_path', type=str, default='solar_AL.txt', help='data file')
    parser.add_argument('--root_path', type=str, default='./dataset/Solar/', help='root path of data file')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seq_len', type=int, default=336, help='sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='forecast horizon')
    parser.add_argument('--patch_len', type=int, default=6, help='patch length')
    parser.add_argument('--stride', type=int, default=6, help='stride between patch')
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=2, help='number of Transformer heads')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
    parser.add_argument('--d_ff', type=int, default=128, help='Tranformer MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Transformer dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--c_in', type=int, default=7, help='number of input channels')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
    parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
    parser.add_argument('--target', type=str, default='OT', help='targe var in univariate task')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_gpu', type=bool, default=1)
    parser.add_argument('--distributed', type=bool, default=1)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # PRE-TRAIN
    parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
    parser.add_argument('--n_epochs_pretrain', type=int, default=50, help='number of pre-training epochs')
    parser.add_argument('--drop_ratio', type=float, default=0.4)
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--showcase', type=int, default=0)

    # FINE-TUNE
    parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of fine-tuning epochs')
    parser.add_argument('--setting', type=str, help='finetune model name')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--percent', type=int, default=100)

    # PITS
    parser.add_argument('--instance_CL', type=int, default=0)
    parser.add_argument('--temporal_CL', type=int, default=1)

    # EVAL PRETEXT TASK

    args = parser.parse_args()

    # --------random seed--------
    # fix_seed = 789
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # --------model args--------
    print('***Args***')
    print(args)

    # --------model options--------
    model_dict = {
                  'dm': dm,
                  }

    if args.mode == 'pt':
        settings_pt = args.setting
        path = os.path.join('./checkpoints', settings_pt)  # models are saved in ./checkpoints/xxxxx
        if not os.path.exists(path):
            os.mkdir(path)

        print('>>>>>>>>>>>>>>>>>>>>>>pre-training: {}>>>>>>>>>>>>>>>>>>>>>>'.format(settings_pt))

        # --------load data--------
        train_data, train_loader = data_provider(args, flag='train')
        vali_data, vali_loader = data_provider(args, flag='val')
        test_data, test_loader = data_provider(args, flag='test')

        # --------load model--------
        model = model_dict[args.model].Model(args).float()
        print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
        print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))
        # --------optimizer--------
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=len(train_loader),
                                            pct_start=0.3,
                                            epochs=args.n_epochs_pretrain,
                                            max_lr=args.lr)
        accelerator = Accelerator()
        device = accelerator.device
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        model.to(device)

        train_pt(model, optimizer, device, train_loader, vali_loader, scheduler, path, args, accelerator)

        if accelerator.is_local_main_process:
            model = model_dict[args.model].Model(args).float().to(device)
            test_pt(test_loader, device, model, path, args.setting, args.patch_len, args.showcase)

    elif args.mode == 'large':
        # --------settings & mkdir--------
        settings_pt = args.setting
        path = os.path.join(args.checkpoints, settings_pt)  # models are saved in ./checkpoints/xxxxx
        if not os.path.exists(path):
            os.mkdir(path)

        print('>>>>>>>>>>>>>>>>>>>>>>pre-training: {}>>>>>>>>>>>>>>>>>>>>>>'.format(settings_pt))

        # --------load data--------
        train_data, train_loader = data_provider(args, flag='train')
        vali_data, vali_loader = data_provider(args, flag='val')

        # --------load model--------
        model = model_dict[args.model].Model(args).float()
        print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
        print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))
        # --------optimizer--------
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=len(train_loader),
                                            pct_start=0.3,
                                            epochs=args.n_epochs_pretrain,
                                            max_lr=args.lr)
        accelerator = Accelerator()
        device = accelerator.device
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        model.to(device)

        large_pt(model, optimizer, device, train_loader, vali_loader, scheduler, path, args, accelerator)

if __name__ == '__main__':
    main()
