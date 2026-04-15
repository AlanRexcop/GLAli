import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet, set_mnist_loader
from utils.detection_util import get_and_print_results
import trainers.locoop
import trainers.zsclip_contra
import trainers.locproto_supc
import datasets.skin40
import datasets.ISIC
import datasets.Dermnet
from os import path as osp

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.lambda_value:
        cfg.lambda_value = args.lambda_value

    if args.topk:
        cfg.topk = args.topk

    cfg.in_dataset = args.in_dataset
    cfg.is_bonder = args.is_bonder
    cfg.is_dense = args.is_dense


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.LOCOOP = CN()
    cfg.TRAINER.LOCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.LOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.Adapter = CN()
    cfg.Adapter.Layer_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cfg.Adapter.Scale = 1.0
    cfg.Adapter.Down_Rate = 256
    cfg.Adapter.Attn = True
    cfg.Adapter.MLP = True
    cfg.Adapter.Visual = False
    cfg.Adapter.Text = False


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

    return cfg


def main(args):
    import clip_w_local
    cfg = setup_cfg(args)
    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if 'mnist' in args.in_dataset:
        id_data_loader, ood_loader = set_mnist_loader(cfg, args)
        out_datasets = [args.in_dataset]

    trainer = build_trainer(cfg)

    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    if trainer.cfg.is_bonder and trainer.cfg.TRAINER=="LocProto":
        trainer.model.text_prototypes = torch.load(osp.join(args.model_dir, 'proto.pth'))


    if args.in_dataset in ['skin40', 'ISIC', 'Dermnet']:
        out_datasets = [item for item in ['skin40', 'ISIC', 'Dermnet'] if item != args.in_dataset]
        id_data_loader = trainer.dm.id_loader

    trainer.test()
    in_score_mcm, in_score_gl, in_score_loc, in_score_gen = trainer.test_ood(id_data_loader, args.T)

    auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
    auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []
    auroc_list_loc, aupr_list_loc, fpr_list_loc = [], [], []
    auroc_list_gen, aupr_list_gen, fpr_list_gen = [], [], []

    for out_dataset in out_datasets:
        print(f"Evaluting OOD dataset {out_dataset}")
        if out_dataset in ['iNaturalist', 'SUN', 'places365', 'Texture', 'skin40', 'ISIC', 'Dermnet']:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
        elif out_dataset in ['eurosat', 'fgvc_aircraft', 'stanford_cars', 'skin40', 'oxford_flowers', 'food101', 'ISIC', 'Dermnet']:
            ood_loader = trainer.dm.ood_loader

        out_score_mcm, out_score_gl, out_score_loc, out_score_gen = trainer.test_ood(ood_loader, args.T)

        print("MCM score")
        get_and_print_results(args, in_score_mcm, out_score_mcm,
                              auroc_list_mcm, aupr_list_mcm, fpr_list_mcm)
        

    print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_mcm), np.mean(auroc_list_mcm), np.mean(aupr_list_mcm)))


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument('--in_dataset', default='imagenet', type=str,
                        help='in-distribution dataset')
    parser.add_argument("--output-dir", type=str, default="./png", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # augment for LoCoOp
    parser.add_argument('--lambda_value', type=float, default=1,
                        help='temperature parameter')
    parser.add_argument('--topk', type=int, default=50,
                        help='topk')
    # augment for MCM and GL-MCM
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=float, default=1,
                        help='temperature parameter')
    parser.add_argument('--is_mine', type=bool, default=False,
                        help='temperature parameter')
    parser.add_argument('--is_bonder', type=bool, default=False,
                        help='temperature parameter')
    parser.add_argument('--is_dense', type=bool, default=False,
                        help='temperature parameter')
    args = parser.parse_args()
    main(args)
