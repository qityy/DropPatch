import argparse
import torch
from torch import optim
import numpy as np
import random
import os
from models import dm
from data_provider.data_factory import data_provider
from engine_pretrain import train_pt, test_pt, large_pt, check_hidden_representations_cka
from engine_finetune import train_ft, test_ft
from tools import transfer_weights


parser = argparse.ArgumentParser()

# GENERAL
parser.add_argument('--mode', type=str, default='pt', help='options: [pt, ft, randinit, eval]')
parser.add_argument('--model', type=str, default='dm')
parser.add_argument('--data', type=str, default='Large', help='dataset name')
parser.add_argument('--data_path', type=str, default='solar_AL.txt', help='data file')
parser.add_argument('--root_path', type=str, default='./dataset/Solar/', help='root path of data file')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='forecast horizon')
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
parser.add_argument('--enc_layers', type=int, default=3)
parser.add_argument('--dec_layers', type=int, default=1)
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
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
parser.add_argument('--checkpoint', type=str, default='/checkpoint.pth')
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
parser.add_argument('--drop_ratio', type=float, default=0.6, help='dropping ratio for the input')
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
parser.add_argument('--showcase', type=int, default=0)

# FINE-TUNE
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of fine-tuning epochs')
parser.add_argument('--setting', type=str, help='finetune model name')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()

# --------random seed--------
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# --------model args--------
print('***Args***')
print(args)

# --------get device--------
gpu_id = args.gpu
if args.use_gpu and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f'Use GPU{gpu_id}')
else:
    device = torch.device('cpu')
    print('Use CPU')

# --------model options--------
model_dict = {
    'dm': dm
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
    model = model_dict[args.model].Model(args).float().to(device)
    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))
    # --------optimizer--------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = train_pt(model, optimizer, device, train_loader, vali_loader, path, args)
    test_pt(test_loader, device, model, path, args.setting, args.patch_len, args.showcase)

elif args.mode == 'ft':
    # locate finetune model
    path = os.path.join('./checkpoints', args.setting)
    if not os.path.exists(path):
        raise Exception(f'Unknown model: {args.setting}')

    # --------fine-tune settings & mkdir--------
    settings_ft = f'{args.model}_{args.pretrained_model_id}_{args.data}_sl{args.seq_len}_' \
                  f'predl{args.pred_len}_patchl{args.patch_len}_bs{args.batch_size}_lr{args.lr}_hdp{args.head_dropout}_epft{args.n_epochs_finetune}'

    path_ft = os.path.join(path, settings_ft)  # models are saved in ./checkpoints/pt_settings/xxxxx(ft_settings)
    if not os.path.exists(path_ft):
        os.mkdir(path_ft)

    print('>>>>>>>>>>>>>>>>>>>>>>fine-tuning: {}>>>>>>>>>>>>>>>>>>>>>>'.format(settings_ft))

    # --------load data--------
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # --------fine-tune model loading--------
    print('loading model...')
    model = model_dict[args.model].Model(args)

    pretrained_model_path = path + args.checkpoint
    model = transfer_weights(pretrained_model_path, model).float().to(device)

    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))

    # --------optimizer--------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_ft(model, optimizer, device, train_loader, vali_loader, test_loader, path_ft, args)
    test_ft(test_loader, device, path_ft, settings_ft, model, args.setting, args.data)

elif args.mode == 'randinit':
    # --------fine-tune settings & mkdir--------
    settings_ft = f'RandInit_{args.model}_ftdata{args.data}_sl{args.seq_len}_predl{args.pred_len}' \
                  f'_patchl{args.patch_len}_bs{args.batch_size}_lr{args.lr}'

    path_ft = os.path.join('./checkpoints', settings_ft)  # models are saved: ./checkpoints/RandInitxxxxx(ft_settings)
    if not os.path.exists(path_ft):
        os.mkdir(path_ft)

    print('>>>>>>>>>>>>>>>>>>>>>>fine-tuning random init model: {}>>>>>>>>>>>>>>>>>>>>>>'.format(settings_ft))
    # --------load data--------
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # --------fine-tune model loading--------
    print('loading model...')
    model = model_dict[args.model].Model(args).float().to(device)
    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))

    # --------optimizer--------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_ft(model, optimizer, device, train_loader, vali_loader, test_loader, path_ft, args)

    test_ft(test_loader, device, path_ft, settings_ft, model, 'RandomInit', args.data)  # results saved as: RandomInit|settings_ft

elif args.mode == 'eval':
    # locate eval pretrained model
    path = os.path.join('./checkpoints', args.setting)
    if not os.path.exists(path):
        raise Exception(f'Unknown pretrained model: {args.setting}')

    print('>>>>>>>>>>>>>>>>>>>>>>evaluate model: {}>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))

    test_data, test_loader = data_provider(args, flag='test')

    # --------model loading--------
    print('loading eval model...')
    model = model_dict[args.model].Model(args).float().to(device)
    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))

    pretrained_model_path = path + args.checkpoint
    pretrained_model = torch.load(pretrained_model_path, map_location='cpu')

    model.load_state_dict(pretrained_model['state_dict'], strict=False)
    print('eval model has been loaded!')

    test_pt(test_loader, device, model, path, args.setting, args.patch_len, args.showcase)

elif args.mode == 'hidden_rep':
    # locate eval pretrained model
    path = os.path.join('./checkpoints', args.setting)
    if not os.path.exists(path):
        raise Exception(f'Unknown pretrained model: {args.setting}')

    print('>>>>>>>>>>>>>>>>>>>>>>evaluate model: {}>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))

    # train_data, train_loader = data_provider(args, flag='train')
    is_shuffle = 1
    test_data, test_loader = data_provider(args, flag='test')

    # --------model loading--------
    print('loading eval model...')
    model = model_dict[args.model].Model(args).float().to(device)
    pretrained_model_path = path + args.checkpoint
    model = transfer_weights(pretrained_model_path, model).float().to(device)
    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))

    print('eval model has been loaded!')

    # test_pt(test_loader, device, model, path, args.setting, args.patch_len, args.showcase)

    # eval_pretext(test_loader, device, path, model, args.setting)

    data_classification_dic = {
        'ETTh1': 0,
        'ETTh2': 1,
        'ETTm1': 2,
        'ETTm2': 3,
        'Weather': 4,
        'ECL': 5,
        'Traffic': 6
    }

    data_classification = data_classification_dic[args.data]

    num_samples = [1, 1]

    check_hidden_representations_cka(model, test_loader, device, data_classification, num_samples, status='pt')

    # --------fine-tune settings & mkdir--------
    settings_ft = f'{args.model}_{args.pretrained_model_id}_{args.data}_sl{args.seq_len}_' \
                  f'predl{args.pred_len}_patchl{args.patch_len}_bs{args.batch_size}_lr{args.lr}_hdp{args.head_dropout}_epft{args.n_epochs_finetune}'

    path_ft = os.path.join(path, settings_ft)

    finetuned_model_path = path_ft + '/finetuned.pth'
    model = model_dict[args.model].Model(args).float().to(device)
    model.load_state_dict(torch.load(finetuned_model_path))
    check_hidden_representations_cka(model, test_loader, device, data_classification, num_samples, status='ft')

if args.mode == 'large':
    # --------settings & mkdir--------
    settings_pt = args.setting
    path = os.path.join('./checkpoints', settings_pt)  # models are saved in ./checkpoints/xxxxx
    if not os.path.exists(path):
        os.mkdir(path)

    print('>>>>>>>>>>>>>>>>>>>>>>pre-training: {}>>>>>>>>>>>>>>>>>>>>>>'.format(settings_pt))

    # --------load data--------
    train_data, train_loader = data_provider(args, flag='train')
    # vali_data, vali_loader = data_provider(args, flag='val')

    # --------load model--------
    model = model_dict[args.model].Model(args).float().to(device)
    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))
    # --------optimizer--------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    large_pt(model, optimizer, device, train_loader, path, args)

elif args.mode == 'fewshot':
    # locate finetune model
    path = os.path.join('./checkpoints', args.setting)
    if not os.path.exists(path):
        raise Exception(f'Unknown model: {args.setting}')

    # --------fine-tune settings & mkdir--------
    settings_ft = f'fewshot{args.percent}%_{args.model}_{args.pretrained_model_id}_{args.data}_sl{args.seq_len}_' \
                  f'predl{args.pred_len}_patchl{args.patch_len}_bs{args.batch_size}_lr{args.lr}_hdp{args.head_dropout}_epft{args.n_epochs_finetune}'

    path_ft = os.path.join(path, settings_ft)  # models are saved in ./checkpoints/pt_settings/xxxxx(ft_settings)
    if not os.path.exists(path_ft):
        os.mkdir(path_ft)

    print('>>>>>>>>>>>>>>>>>>>>>>fine-tuning: {}>>>>>>>>>>>>>>>>>>>>>>'.format(settings_ft))

    # --------load data--------
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # --------fine-tune model loading--------
    print('loading model...')
    model = model_dict[args.model].Model(args)

    pretrained_model_path = path + args.checkpoint

    model = transfer_weights(pretrained_model_path, model).float().to(device)

    print('total parameters: {:2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('total parameters: {:2f}K'.format(sum(p.numel() for p in model.parameters()) / 1000.0))

    # --------optimizer--------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_ft(model, optimizer, device, train_loader, vali_loader, test_loader, path_ft, args)
    test_ft(test_loader, device, path_ft, settings_ft, model, args.setting, args.data)
