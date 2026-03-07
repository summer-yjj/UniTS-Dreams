from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, GLUONTSDataset
from data_provider.uea import collate_fn
from data_provider.dreams_pointseg import DreamsPointSegDataset, collate_pointseg
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    # 'm4': Dataset_M4,  Removed due to the LICENSE file constraints of m4.py
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    "dreams_pointseg": DreamsPointSegDataset,
    # datasets from gluonts package:
    "gluonts": GLUONTSDataset,
}


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def data_provider(args, config, flag, ddp=False):  # args,
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # working on one gpu
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if 'gluonts' in config['data']:
        # process gluonts dataset:
        data_set = Data(
            dataset_name=config['dataset_name'],
            size=(config['seq_len'], config['label_len'], config['pred_len']),
            path=config['root_path'],
            # Don't set dataset_writer
            features=config["features"],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    timeenc = 0 if config['embed'] != 'timeF' else 1

    if 'anomaly_detection' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            win_size=config['seq_len'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print("ddp mode is set to false for anomaly_detection", ddp, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if (ddp and dist.is_initialized()) else None,
            #sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
    elif config.get('data') == 'dreams_pointseg' or 'point_segmentation' in config.get('task_name', ''):
        drop_last = flag == 'train'
        split_files = config.get('split_files') or config.get('split_file')
        if isinstance(split_files, str):
            split_files = {flag: split_files} if split_files else None
        elif isinstance(split_files, dict):
            split_files = split_files.get(flag)
        data_set = Data(
            root_path=config['root_path'],
            flag=flag,
            window_T=config.get('window_T', config.get('seq_len', 256)),
            stride_T=config.get('stride_T', config.get('stride', 128)),
            fs=config.get('fs', 256),
            num_classes=config.get('num_classes', 2),
            split_files=split_files,
            split_list=config.get('split_list'),
            file_list=config.get('file_list'),
            debug=getattr(args, 'debug', '') == 'enabled',
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(data_set, args.subsample_pct, args.fix_seed)
        print(flag, 'dreams_pointseg', len(data_set))
        sampler = DistributedSampler(data_set) if (ddp and dist.is_initialized()) else None
        use_weighted_sampling = (flag == 'train') and bool(getattr(args, 'pointseg_weighted_sampling', 0))
        if use_weighted_sampling and hasattr(data_set, 'get_window_sample_weights'):
            pos_w = float(getattr(args, 'pointseg_pos_window_weight', 3.0))
            sample_weights = data_set.get_window_sample_weights(pos_window_weight=pos_w)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle_flag = False
            print('pointseg weighted sampling enabled:', 'pos_window_weight=', pos_w)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=collate_pointseg,
        )
        return data_set, data_loader
    elif 'classification' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            sampler=DistributedSampler(data_set) if (ddp and dist.is_initialized()) else None,
            #sampler=DistributedSampler(data_set) if ddp else None,
            collate_fn=lambda x: collate_fn(x, max_len=config['seq_len'])
        )
        return data_set, data_loader
    else:
        if config['data'] == 'm4':
            drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            data_path=config['data_path'],
            flag=flag,
            size=[config['seq_len'], config['label_len'], config['pred_len']],
            features=config['features'],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=config['seasonal_patterns'] if config['data'] == 'm4' else None
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if (ddp and dist.is_initialized()) else None,

            #sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
