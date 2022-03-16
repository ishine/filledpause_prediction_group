from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# My Library
from fp_pred_group.module import MyLightningModel
from fp_pred_group.dataset import MyDataset
from fp_pred_group.util.train_util import collate_fn

def get_data_loaders(data_config, utt_list_paths, in_dir, out_dir, collate_fn):
    data_loaders = {}

    for phase in ["train", "dev"]:

        with open(utt_list_paths[phase], "r") as f:
            utts = [l.strip() for l in f if len(l.strip()) > 0]

        in_feats_paths = [
            in_dir / ("-".join(utt.split(":")[:3]) + "-feats.npy") 
            for utt in utts]
        out_feats_paths = [out_dir / in_path.name for in_path in in_feats_paths]

        dataset = MyDataset(in_feats_paths, out_feats_paths)
        data_loaders[phase] = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,
            shuffle=phase.startswith("train"),
        )

    return data_loaders    

def setup_each_model(
    config, out_dir, fp_list, 
    train_fp_rate_list_path, dev_fp_rate_list_path, 
    utt_list_paths, in_feat_dir, out_feat_dir, collate_fn,
    model, fine_tune, max_steps,
    load_ckpt_path=None,
    ):

    # Get fp rate
    train_fp_rate_dict = {}
    dev_fp_rate_dict = {}
    with open(train_fp_rate_list_path, "r") as f:
        for l in f:
            train_fp_rate_dict[l.strip().split(":")[0]] = float(l.strip().split(":")[1])
    with open(dev_fp_rate_list_path, "r") as f:
        for l in f:
            dev_fp_rate_dict[l.strip().split(":")[0]] = float(l.strip().split(":")[1])

    # Get loss weights
    loss_weights = [1 / (train_fp_rate_dict["no_fp"] + train_fp_rate_dict["others"])]
    for fp in fp_list:
        if train_fp_rate_dict[fp] == 0:
            loss_weights.append(0)
        else:
            loss_weights.append(1 / train_fp_rate_dict[fp])

    # data loaders
    data_loaders = get_data_loaders(config.data, utt_list_paths, in_feat_dir, out_feat_dir, collate_fn)

    # model
    lr_scheduler_params = config.train.optim.lr_scheduler.params
    lr_scheduler_params["step_size"] = int(config.train.optim.lr_scheduler.params.step_size / len(data_loaders["train"]))
    model_params = {
        "model": model,
        "fp_list": fp_list,
        "train_fp_rate_dict": train_fp_rate_dict,
        "dev_fp_rate_dict": dev_fp_rate_dict,
        "loss_weights": loss_weights,
        "optimizer_name": config.train.optim.optimizer.name,
        "optimizer_params": config.train.optim.optimizer.params,
        "lr_scheduler_name": config.train.optim.lr_scheduler.name,
        "lr_scheduler_params": lr_scheduler_params,
    }
    if fine_tune:
        pl_model = MyLightningModel.load_from_checkpoint(
            load_ckpt_path,
            **model_params,
        )        
    else:
        pl_model = MyLightningModel(
            **model_params,
        )

    # callbacks
    dirpath = out_dir / config.train.checkpoint.params.dirname
    dirpath.mkdir(parents=True, exist_ok=True)
    ckpt_params = {
        "monitor": config.train.checkpoint.params.monitor,
        "every_n_train_steps": config.train.checkpoint.params.every_n_train_steps,
        "save_top_k": config.train.checkpoint.params.save_top_k,
    }
    ckpt_callback = ModelCheckpoint(
        dirpath=dirpath,
        **ckpt_params,
    )
    lr_monitor = LearningRateMonitor("step")
    callbacks = [ckpt_callback, lr_monitor]

    # logging
    loggers = []
    for phase in ["train", "val"]:
        loggers.append(
            pl_loggers.TensorBoardLogger(
                save_dir=str(out_dir),
                name=config.train.logging.name,
                version=config.train.logging.version,
                sub_dir=phase)
        )

    trainer = pl.Trainer(
        # gpu
        gpus=config.train.gpus,
        auto_select_gpus=config.train.auto_select_gpus,
        # training
        max_steps=max_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        gradient_clip_val=config.train.gradient_clip_val,
        # checkpoint/logging
        default_root_dir=out_dir,
        callbacks=callbacks,
        logger=loggers,
        profiler="simple",
    )

    return data_loaders, pl_model, trainer


@hydra.main(config_path="conf/train", config_name="config")
def myapp(config: DictConfig):

    # Set output directory
    out_dir = Path(to_absolute_path(config.train.out_dir))
    if config.train.model_type == "non_personalized":
        out_dir_m = out_dir / "non_personalized"
    elif config.train.model_type == "group":
        group_id = config.train.group_id
        out_dir_m = out_dir / "group{}".format(str(group_id))
    exist_ok = True if config.train.resume else False
    out_dir_m.mkdir(parents=True, exist_ok=exist_ok)

    # Save config
    with open(out_dir_m / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Random seed
    pl.seed_everything(config.random_seed)

    # FPs
    fp_list_path = Path(to_absolute_path(config.data.fp_list))
    with open(fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Set feature directory
    in_feat_dir = Path(config.data.preprocessed_dir) / "infeats"
    out_feat_dir = Path(config.data.preprocessed_dir) / "outfeats"

    # Set model parameters
    model = hydra.utils.instantiate(config.model.netG)

    # Training settings
    max_steps = config.train.max_steps
    fine_tune = config.train.fine_tune

    # Trainig non-personalized model
    if config.train.model_type == "non_personalized":
        train_fp_rate_list_path = Path(config.data.preprocessed_dir) / "train_all_fp_rate.list"
        dev_fp_rate_list_path = Path(config.data.preprocessed_dir) / "dev_all_fp_rate.list"
        utt_list_paths = {}
        utt_list_paths["train"] = Path(config.data.preprocessed_dir) / "train_all.list"
        utt_list_paths["dev"] = Path(config.data.preprocessed_dir) / "dev_all.list"
        data_loaders, pl_model, trainer = setup_each_model(
            config, out_dir_m, fp_list, 
            train_fp_rate_list_path, dev_fp_rate_list_path, 
            utt_list_paths, in_feat_dir, out_feat_dir, collate_fn,
            model, fine_tune, max_steps,
            )
        trainer.fit(pl_model, data_loaders["train"], data_loaders["dev"])

    # Trainig group-dependent models
    elif config.train.model_type == "group":
        ckpt_dir = out_dir / "non_personalized" / config.train.checkpoint.params.dirname
        load_ckpt_path = list(ckpt_dir.glob(
            "*-step={}.ckpt".format(str(config.train.load_ckpt_step))
        ))[0]
        group_id = config.train.group_id
        train_fp_rate_list_path = \
            Path(config.data.preprocessed_dir) / "train_group{}_fp_rate.list".format(str(group_id))
        dev_fp_rate_list_path = \
            Path(config.data.preprocessed_dir) / "dev_group{}_fp_rate.list".format(str(group_id))
        utt_list_paths = {}
        utt_list_paths["train"] = Path(config.data.preprocessed_dir) / "train_group{}.list".format(str(group_id))
        utt_list_paths["dev"] = Path(config.data.preprocessed_dir) / "dev_group{}.list".format(str(group_id))
        data_loaders, pl_model, trainer = setup_each_model(
            config, out_dir_m, fp_list, 
            train_fp_rate_list_path, dev_fp_rate_list_path, 
            utt_list_paths, in_feat_dir, out_feat_dir, collate_fn,
            model, fine_tune, max_steps,
            load_ckpt_path=load_ckpt_path,
            )
        trainer.fit(pl_model, data_loaders["train"], data_loaders["dev"])
        
if __name__=="__main__":
    myapp()