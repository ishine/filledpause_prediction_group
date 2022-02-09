from pathlib import Path
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# My library
from dataset import MyDataset
from module import MyLightningModel
from util.train_util import collate_fn
from util.eval_util import calc_score_all, calc_score_each_fp

def evaluate(
    config, train_config, utt_list_path, trainer, model, out_dir, fp_list, eval_fp_rate_dict):

    # Load utt list
    with open(utt_list_path, "r") as f:
        sentence_list = [l.strip() for l in f]
        utt_name_list = ["-".join(sen.split(":")[:3]) for sen in sentence_list]

    # Dataset
    in_feat_dir = Path(train_config.data.preprocessed_dir) / "infeats"
    out_feat_dir = Path(train_config.data.preprocessed_dir) / "outfeats"

    in_feats_paths = [p for p in in_feat_dir.glob("*-feats.npy") if p.stem.split("-feats")[0] in utt_name_list]
    out_feats_paths = [out_feat_dir / in_path.name for in_path in in_feats_paths]

    dataset = MyDataset(in_feats_paths, out_feats_paths)
    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=config.data.num_workers,
        shuffle=False,
    )
    
    # Prediction
    out_utt_list = []
    prediction_list = []
    prediction_dict = {}
    target_list = []
    target_dict = {}
    outputs = trainer.predict(model, data_loader)
    for output in tqdm(outputs):
        batch_idx = output["batch_idx"]
        predictions = output["predictions"]
        targets = output["targets"]

        for in_feats_path, prediction, target in zip(
            in_feats_paths[batch_idx*config.data.batch_size : (batch_idx+1)*config.data.batch_size],
            predictions,
            targets
        ):
            speaker_id, koen_id, ipu_id = in_feats_path.stem.split("-")[:3]
            out_utt_list.append(f"{speaker_id}, {koen_id}, {ipu_id}:")
            for sen in sentence_list:
                if sen.startswith(f"{speaker_id}:{koen_id}:{ipu_id}:"):
                    ipu_text = sen.split(":")[-1]
            out_utt_list.append(f"\ttarget text: \t{ipu_text}")

            fp_prediction = [str(int(i)) for i in torch.argmax(prediction, dim=1)]
            fp_prediction = " ".join(fp_prediction)
            out_utt_list.append(f"\tpredicted text: \t{fp_prediction}")

            prediction_list.append(prediction)
            target_list.append(target)

            # Each speaker
            if speaker_id in prediction_dict.keys():
                prediction_dict[speaker_id].append(prediction)
            else:
                prediction_dict[speaker_id] = [prediction]
            if speaker_id in target_dict.keys():
                target_dict[speaker_id].append(target)
            else:
                target_dict[speaker_id] = [target]

    with open(out_dir / "fp_prediction.txt", "w") as f:
        print("writing prediction...")
        f.write("\n".join(out_utt_list))

    # Calc score
    print("calc score...")

    # FP position
    precision, recall, f_score, specificity = calc_score_all(
        prediction_list,
        target_list
    )
    out_text = \
        "--- fp position ---\nprecision:\t{}\nrecall:\t{}\nf_score:\t{}\nspecificity:{}\n\n".format(
            precision, recall, f_score, specificity
        )

    # Each fp word
    precision_word = 0
    recall_word = 0
    f_score_word = 0
    specificity_word = 0
    rate_sum = 0
    out_texts = []
    for i in tqdm(range(len(fp_list))):
        fp_rate = eval_fp_rate_dict[fp_list[i]]
        rate_sum += fp_rate

        precision, recall, f_score, specificity = calc_score_each_fp(
            prediction_list,
            target_list,
            i + 1
        )
        if precision is not None and torch.isnan(precision).sum() == 0:
            precision_word += precision * fp_rate
        if recall is not None and torch.isnan(recall).sum() == 0:
            recall_word += recall * fp_rate
        if f_score is not None and torch.isnan(f_score).sum() == 0:
            f_score_word += f_score * fp_rate
        if specificity is not None and torch.isnan(specificity).sum() == 0:
            specificity_word += specificity * fp_rate

        out_texts.append(
            "--- {} ---\nprecision:\t{}\nrecall:\t{}\nf_score:\t{}\nspecificity:{}\n".format(
                fp_list[i], precision, recall, f_score, specificity
            )
        )

    # FP word
    out_text += \
        "--- fp word ---\nprecision:\t{}\nrecall:\t{}\nf_score:\t{}\nspecificity:{}\n\n".format(
            precision_word / rate_sum, 
            recall_word / rate_sum, 
            f_score_word / rate_sum, 
            specificity_word / rate_sum
        ) + "\n".join(out_texts)

    # Each speaker
    out_text += "\n--- speakers ---\n"
    f_scores = []
    f_score_words = []
    for spk in prediction_dict.keys():
        prediction_list = prediction_dict[spk]
        target_list = target_dict[spk]
        _, _, f_score, _ = calc_score_all(
            prediction_list,
            target_list
        )
        f_scores.append(f_score)
        out_text += "{}: \t\t{},".format(spk, f_score)

        precision_word = 0
        recall_word = 0
        f_score_word = 0
        specificity_word = 0
        rate_sum = 0
        for i in tqdm(range(len(fp_list))):
            fp_rate = eval_fp_rate_dict[fp_list[i]]
            rate_sum += fp_rate

            precision, recall, f_score, specificity = calc_score_each_fp(
                prediction_list,
                target_list,
                i + 1
            )
            if precision is not None and torch.isnan(precision).sum() == 0:
                precision_word += precision * fp_rate
            if recall is not None and torch.isnan(recall).sum() == 0:
                recall_word += recall * fp_rate
            if f_score is not None and torch.isnan(f_score).sum() == 0:
                f_score_word += f_score * fp_rate
            if specificity is not None and torch.isnan(specificity).sum() == 0:
                specificity_word += specificity * fp_rate

        f_score_words.append(f_score_word)
        out_text += "\t{}\n".format(f_score_word / rate_sum)
        
    # Write scores
    with open(out_dir / "scores.txt", "w") as f:
        print("writing score...")
        f.write(out_text)

@hydra.main(config_path="conf/evaluate", config_name="config")
def main(config: DictConfig):

    # Phase
    phase = "eval"

    # Model type
    model_type = config[phase].model_type
    if model_type == "non_personalized":
        model_name = "non_personalized"
    elif model_type == "group":
        group_id = config[phase].group_id
        model_name = "group{}".format(str(group_id))

    # Input directory
    exp_dir = Path(to_absolute_path(config[phase].exp_dir))
    exp_dir_m = exp_dir / model_name
    ckpt_dir = exp_dir_m / "ckpt"

    # Output directory
    out_dir = Path(to_absolute_path(config[phase].out_dir))
    out_dir_m = out_dir / model_name
    out_dir_m.mkdir(parents=True, exist_ok=False)

    # Load config
    train_config = OmegaConf.load(exp_dir_m / "config.yaml")

    # Save config
    with open(out_dir_m / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Set rrandom seed
    pl.seed_everything(config.random_seed)

    # Utterance list
    if model_type == "non_personalized":
        utt_list_path = Path(train_config.data.preprocessed_dir) / "eval_all.list"
    elif model_type == "group":
        utt_list_path = Path(train_config.data.preprocessed_dir) / "eval_{}.list".format(model_name)

    # FPs
    fp_list_path = Path(train_config.data.fp_list)
    with open(fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Get fp rate
    eval_fp_rate_dict = {}
    if model_type == "non_personalized":
        eval_fp_rate_list_path = Path(train_config.data.preprocessed_dir) / "eval_all_fp_rate.list"
    elif model_type == "group":
        eval_fp_rate_list_path = \
            Path(train_config.data.preprocessed_dir) / "eval_{}_fp_rate.list".format(model_name)
    with open(eval_fp_rate_list_path, "r") as f:
        for l in f:
            eval_fp_rate_dict[l.strip().split(":")[0]] = float(l.strip().split(":")[1])

    # Get loss weights
    if config[phase].loss_weights:
        loss_weights = [1 / (eval_fp_rate_dict["no_fp"] + eval_fp_rate_dict["others"])]
        for fp in fp_list:
            if eval_fp_rate_dict[fp] == 0:
                loss_weights.append(0)
            else:
                loss_weights.append(1 / eval_fp_rate_dict[fp])
    else:
        loss_weights = None

    # Load model
    model = hydra.utils.instantiate(train_config.model.netG)
    ckpt_path = list(ckpt_dir.glob(
        "*-step={}.ckpt".format(str(config[phase].checkpoint.step))
    ))[0]
    pl_model = MyLightningModel.load_from_checkpoint(
        ckpt_path,
        model=model,
        fp_list=fp_list,
        loss_weights=loss_weights,
    )

    # Trainer
    trainer = pl.Trainer(
        # gpu
        gpus=config[phase].gpus,
        auto_select_gpus=config[phase].auto_select_gpus,
        default_root_dir=exp_dir_m,
        # profiler="simple",
    )

    # Predict
    evaluate(config, train_config, utt_list_path, trainer, pl_model, out_dir_m, fp_list, eval_fp_rate_dict)

    # elif config.corpus.name == "utokyo_naist_lecture":
    #     predict_utokyo_naist_lecture(config, phase, trainer, pl_model, out_dir, fp_list, eval_fp_rate_dict)

if __name__=="__main__":
    main()