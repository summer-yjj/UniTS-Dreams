import argparse
import os
import torch
from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
import random
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
#import wandb
from utils.ddp import is_main_process, init_distributed_mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='UniTS',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='exp/all_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='constant', help='adjust learning rate (constant=one lr per run, supervised=decay, cosine=cosine decay for point_seg)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='enabled', help='disabled')
    parser.add_argument('--project_name', type=str,
                        default='tsfm-multitask', help='wandb project name')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--prompt_num", type=int, default=5)
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # point_segmentation
    parser.add_argument("--seg_loss", type=str, default="ce_dice", choices=["ce", "ce_dice", "focal", "tversky"])
    parser.add_argument("--class_weight", type=str, default="auto", choices=["auto", "manual"])
    parser.add_argument("--seg_pos_weight", type=float, default=None, help="extra scale for positive class(es) in auto class_weight (e.g. 2.0 to emphasize spindle)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="gamma for focal loss in point segmentation")
    parser.add_argument("--tversky_alpha", type=float, default=0.7, help="Tversky alpha (FN penalty) for point segmentation")
    parser.add_argument("--tversky_beta", type=float, default=0.3, help="Tversky beta (FP penalty) for point segmentation")
    parser.add_argument("--bg_keep_prob", type=float, default=1.0, help="background keep probability for point-wise loss (<=1.0)")
    parser.add_argument("--pointseg_weighted_sampling", type=int, default=0, help="use WeightedRandomSampler for point-seg train windows (0/1)")
    parser.add_argument("--pointseg_pos_window_weight", type=float, default=3.0, help="extra sampling weight for windows containing positive labels")
    parser.add_argument("--pointseg_best_metric", type=str, default="spindle_f1", choices=["spindle_f1", "event_f1", "macro_f1", "spindle_event_combo"], help="metric used to select best point-seg checkpoint")
    parser.add_argument("--pointseg_best_spindle_weight", type=float, default=0.7, help="weight for spindle_f1 when pointseg_best_metric=spindle_event_combo")
    parser.add_argument("--pointseg_best_event_weight", type=float, default=0.3, help="weight for event_f1 when pointseg_best_metric=spindle_event_combo")
    parser.add_argument("--pointseg_best_pos_rate_guard", type=float, default=0.0, help="if >0, only save best when pred_pos_rate in [gt/guard, gt*guard]")
    parser.add_argument("--pointseg_use_threshold_search", type=int, default=0, help="enable val-set threshold search for binary point-seg test")
    parser.add_argument("--pointseg_threshold_metric", type=str, default="event_f1", choices=["event_f1", "spindle_f1", "combo"], help="selection metric for val threshold search")
    parser.add_argument("--pointseg_threshold_min", type=float, default=0.05, help="min threshold for cls1 probability scan")
    parser.add_argument("--pointseg_threshold_max", type=float, default=0.95, help="max threshold for cls1 probability scan")
    parser.add_argument("--pointseg_threshold_steps", type=int, default=19, help="number of threshold points in scan")
    parser.add_argument("--pointseg_threshold_spindle_weight", type=float, default=0.7, help="spindle weight when pointseg_threshold_metric=combo")
    parser.add_argument("--pointseg_threshold_event_weight", type=float, default=0.3, help="event weight when pointseg_threshold_metric=combo")
    parser.add_argument("--load_ckpt", type=str, default=None, help="path to checkpoint for test (e.g. checkpoints/<model_id>/best.pth)")

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    args = parser.parse_args()
    # point_segmentation: 默认梯度裁剪以稳定训练，避免 loss 剧烈震荡
    if args.task_name == "point_segmentation" and args.clip_grad is None:
        args.clip_grad = 1.0
    init_distributed_mode(args)
    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)

    print('Args in experiment:')
    print(args)
    exp_name = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des)

    if int(args.prompt_tune_epoch) != 0:
        exp_name = 'Ptune'+str(args.prompt_tune_epoch)+'_'+exp_name
        print(exp_name)

    if is_main_process():
        if wandb is not None:
            wandb.init(
                name=exp_name,
                # set the wandb project where this run will be logged
                project=args.project_name,
                # track hyperparameters and run metadata
                config=args,
                mode=args.debug,
            )

    if args.task_name == "point_segmentation":
        from exp.exp_pointseg import Exp_PointSeg
        Exp = Exp_PointSeg
    else:
        Exp = Exp_All_Task_SUP

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.d_model,
                args.e_layers,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if args.task_name == "point_segmentation":
                load_ckpt = os.path.join(args.checkpoints, setting, "best.pth")
                if not os.path.isfile(load_ckpt):
                    load_ckpt = os.path.join(args.checkpoints, setting, "last.pth")
                print('>>>>>>>auto testing (after training): {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, load_ckpt=load_ckpt)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.d_model,
            args.e_layers,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.task_name == "point_segmentation":
            load_ckpt = args.load_ckpt or os.path.join(args.checkpoints, setting, "best.pth")
            exp.test(setting, load_ckpt=load_ckpt)
        else:
            exp.test(setting, load_pretrain=True)
        torch.cuda.empty_cache()
