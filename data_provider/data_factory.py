from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar,
                                       Dataset_Large, Dataset_PEMS, Dataset_Web)
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'Solar': Dataset_Solar,
    'Large': Dataset_Large,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Web': Dataset_Web,
}


def data_provider(args, flag):
    if args.data == 'Large':
        Data = Dataset_Large

        data_path = ['./dataset/large/airquality.csv', './dataset/large/metro.csv', './dataset/large/energy.csv',
                     './dataset/large/WTH.csv', './dataset/large/tcpc.csv', './dataset/large/wind_1_minute_dataset.csv',
                     './dataset/large/sunspot.csv', './dataset/large/electricity_demand.csv',
                     './dataset/large/solar_1_minute_dataset.csv', './dataset/large/river_flow.csv',
                     './dataset/electricity/electricity.csv', './dataset/PEMS/PEMS07.npz']
        data_set = Data(data_path, args.seq_len, flag, scale=True)
        print(len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True)
        return data_set, data_loader

    if args.data == 'Merge':
        Data = Dataset_Large

        data_path = ['./dataset/large/airquality.csv', './dataset/large/metro.csv', './dataset/large/energy.csv',
                     './dataset/large/WTH.csv', './dataset/large/tcpc.csv', './dataset/large/wind_1_minute_dataset.csv',
                     './dataset/large/sunspot.csv', './dataset/large/electricity_demand.csv',
                     './dataset/large/solar_1_minute_dataset.csv', './dataset/large/river_flow.csv']
        data_set = Data(data_path, args.seq_len, flag, scale=True)
        print(len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True)
        return data_set, data_loader

    few_shot = True if args.mode == 'fewshot' else False

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=args.percent,
        few_shot=few_shot,
        seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
