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
    config, phase, trainer, model, out_dir, fp_list, eval_fp_rate_dict):

    # Load utt list
    with open(to_absolute_path(config.data[phase].utt_list), "r") as f:
        sentence_list = [l.strip() for l in f]
        utt_name_list = ["-".join(sen.split(":")[:3]) for sen in sentence_list]

    # Dataset
    in_feat_dir = Path(to_absolute_path(config.data.in_feat_dir))
    out_feat_dir = Path(to_absolute_path(config.data.out_feat_dir))

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

            if config.eval.each_speaker:
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
    if config.eval.each_speaker:
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
        
        # out_text += "var: \t\t{},\t{}".format(np.var(f_scores), np.var(f_score_words))

    # Write scores
    with open(out_dir / "scores.txt", "w") as f:
        print("writing score...")
        f.write(out_text)

    # # check
    # with open(Path(config.data.eval.sentence_list), "r") as f:
    #     utt_list = sorted(
    #         [l.strip() for l in f], 
    #         key=lambda u: (int(u.split(":")[0]), u.split(":")[1], int(u.split(":")[2]))
    #     )
    #     utt_text = "".join([re.sub(r"\(F.*?\)", "", utt.split(":")[3].replace(" ", "")) for utt in utt_list])
    # out_utt_text = "".join([utt.split(":")[1] for utt in out_utt_list if not utt.split(":")[1].startswith("(F)")])
    # assert utt_text == out_utt_text, f"utt_text should be equal to out_utt_text\nutt_text:\n{utt_text}\n\nout_utt_text:\n{out_utt_text}"

# def predict_utokyo_naist_lecture(
#     config, phase, trainer, model, out_dir, fp_list, eval_fp_rate_dict):
#     print("out directory: {}".format(out_dir))

#     # prediction mode
#     mode = config[phase].mode

#     # Paths
#     in_feat_dir = Path(config.data.in_dir)
#     out_feat_dir = Path(config.data.out_dir)
#     utt_list_path = Path(config.data.utt_list_path)

#     # Params
#     batch_size = config.data.batch_size

#     # Dataset
#     in_feats_paths = list(in_feat_dir.glob("*-feats.npy"))
#     out_feats_paths = [out_feat_dir / in_path.name for in_path in in_feats_paths]
#     dataset = NoFPDataset(in_feats_paths, out_feats_paths, utt_list_path)
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         collate_fn=dataset.collate_fn,
#         pin_memory=True,
#         num_workers=config.data.num_workers,
#         shuffle=False
#     )

#     # # fp rate
#     # fp_rate_list_path = data_dir / "traindeveval_fp_rate_list.txt"
#     # if config.eval.loss_weights:
#     #     fp_rate_dict = {}
#     #     with open(fp_rate_list_path, "r") as f:
#     #         for l in f:
#     #             fp_rate_dict[l.split(":")[0]] = float(l.strip().split(":")[1])
#     #     fp_rate_dict["no_fp_or_others"] = fp_rate_dict["no_fp"] + fp_rate_dict["others"]

#     #     # get loss weights
#     #     loss_weights = [1 / (fp_rate_dict["no_fp"] + fp_rate_dict["others"])]
#     #     for fp in fp_list:
#     #         loss_weights.append(1 / fp_rate_dict[fp])
#     # else:
#     #     loss_weights = None

#     # output directory
#     if mode == "pred_all":
#         pred_dir = out_dir / "prediction_all"
#     elif mode == "pred_type":
#         pred_dir = out_dir / "prediction_type"
#     elif mode == "random_all":
#         pred_dir = out_dir / "random_all"
#     elif mode == "random_type":
#         pred_dir = out_dir / "random_type"
#     pred_text_dir = pred_dir / "text"
#     pred_dir.mkdir(parents=True, exist_ok=True)
#     pred_text_dir.mkdir(parents=True, exist_ok=True)

#     # Prediction
#     out_utt_list = []
#     prediction_list = []
#     target_list = []
#     outputs = trainer.predict(model, data_loader)
#     for output in tqdm(outputs):
#         batch_idx = output["batch_idx"]
#         predictions = output["predictions"]
#         targets = output["targets"]
#         texts = output["texts"]

#         for in_feats_path, prediction, text, target in zip(
#             in_feats_paths[batch_idx*batch_size : (batch_idx+1)*batch_size],
#             predictions,
#             texts,
#             targets
#         ):
#             prediction_list.append(prediction)
#             target_list.append(target)

#             text_len = len([t for t in text.split(" ") if len(t) > 0]) if text != "" else 0
#             breath_para_name = in_feats_path.stem.replace("-feats", "")

#             # prediction without true position
#             if mode == "pred_all":
#                 fp_predictions = [int(i) for i in torch.argmax(prediction[:text_len+1], dim=1)]      
#                 i_utt = 0
#                 if fp_predictions[0] > 0:
#                     out_utt_list.append(f"{breath_para_name}-{i_utt}:(F)" + fp_list[fp_predictions[0]-1])
#                     with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                         f.write(fp_list[fp_predictions[0]-1])
#                     i_utt += 1

#                 out_texts = []
#                 for t, f_pred in zip(
#                     [t for t in text.split(" ") if len(t) > 0], 
#                     fp_predictions[1:]
#                 ):
#                     out_texts.append(t)
#                     if f_pred > 0:
#                         if len(out_texts) > 0:
#                             out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
#                             with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                                 f.write("".join(out_texts))
#                             i_utt += 1

#                         out_utt_list.append(f"{breath_para_name}-{i_utt}:(F)" + fp_list[int(f_pred)-1])
#                         with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                             f.write(fp_list[f_pred-1])
#                         i_utt += 1

#                         out_texts = []

#                 if len(out_texts) > 0:
#                     out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
#                     with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                         f.write("".join(out_texts))

#             # prediction with true position
#             elif mode == "pred_type":
#                 out_feat = np.load(data_dir / "outfeats" / f"{breath_para_name}.npy")
#                 fp_positions = [int(f > 0) for f in out_feat]
#                 fp_type_predictions = [str(int(i)) for i in torch.argmax(prediction[:text_len+1, 1:], dim=1)]
#                 assert len(fp_positions) == len(fp_type_predictions)

#                 i = 0
#                 if fp_positions[0] == 1:
#                     out_utt_list.append(f"{breath_para_name}-{i}:(F)" + fp_list[int(fp_type_predictions[0])-1])
#                     with open(pred_text_dir / f"{breath_para_name}-{i}.txt", "w") as f:
#                         f.write(fp_list[int(fp_type_predictions[0])-1])
#                     i += 1

#                 out_texts = []
#                 for t, f_pred, f_posi in zip(text.split(" "), fp_type_predictions[1:], fp_positions[1:]):
#                     out_texts.append(t)
#                     if f_posi == 1:
#                         if len(out_texts) > 0:
#                             out_utt_list.append(f"{breath_para_name}-{i}:" + "".join(out_texts))
#                             with open(pred_text_dir / f"{breath_para_name}-{i}.txt", "w") as f:
#                                 f.write("".join(out_texts))
#                             i += 1

#                         out_utt_list.append(f"{breath_para_name}-{i}:(F)" + fp_list[int(f_pred)-1])
#                         with open(pred_text_dir / f"{breath_para_name}-{i}.txt", "w") as f:
#                             f.write(fp_list[int(f_pred)-1])
#                         i += 1

#                         out_texts = []

#                 if len(out_texts) > 0:
#                     out_utt_list.append(f"{breath_para_name}-{i}:" + "".join(out_texts))
#                     with open(pred_text_dir / f"{breath_para_name}-{i}.txt", "w") as f:
#                         f.write("".join(out_texts))

#             # prediction random
#             elif mode == "random_all":
#                 prob = [fp_rate_dict["no_fp_or_others"]] + [fp_rate_dict[f] for f in fp_list]
                
#                 i_utt = 0
#                 j = np.random.choice(np.arange(14), p=prob)
#                 if j > 0:
#                     out_utt_list.append(f"{breath_para_name}-{i_utt}:(F){fp_list[j-1]}")
#                     with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                         f.write(fp_list[j-1])
#                     i_utt += 1

#                 out_texts = []
#                 for i in range(text_len):                        
#                     out_texts.append(text.split(" ")[i])
#                     j = np.random.choice(np.arange(14), p=prob)
#                     if j > 0:
#                         if len(out_texts) > 0:
#                             out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
#                             with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                                 f.write("".join(out_texts))
#                             i_utt += 1

#                         out_utt_list.append(f"{breath_para_name}-{i_utt}:(F){fp_list[j-1]}")
#                         with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                             f.write(fp_list[j-1])
#                         i_utt += 1

#                         out_texts = []

#                 if len(out_texts) > 0:
#                     out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
#                     with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                         f.write("".join(out_texts))

#             # prediction random with true position
#             elif mode == "random_type":
#                 out_feat = np.load(data_dir / "outfeats" / f"{breath_para_name}.npy")
#                 fp_positions = [int(f > 0) for f in out_feat]
#                 prob = [fp_rate_dict[f] for f in fp_list]
#                 prob = list(np.array(prob) / sum(prob))

#                 i_utt = 0
#                 if fp_positions[0] == 1:
#                     j = np.random.choice(np.arange(13), p=prob)

#                     out_utt_list.append(f"{breath_para_name}-{i_utt}:(F){fp_list[j]}")
#                     with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                         f.write(fp_list[j])
#                     i_utt += 1

#                 out_texts = []
#                 for i in range(text_len):                        
#                     out_texts.append(text.split(" ")[i])
#                     if fp_positions[i+1] == 1:
#                         j = np.random.choice(np.arange(13), p=prob)

#                         out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
#                         with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                             f.write("".join(out_texts))
#                         i_utt += 1

#                         out_utt_list.append(f"{breath_para_name}-{i_utt}:(F){fp_list[j]}")
#                         with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                             f.write(fp_list[j])
#                         i_utt += 1

#                         out_texts = []

#                 if len(out_texts) > 0:
#                     out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
#                     with open(pred_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
#                         f.write("".join(out_texts))

#     # Write predicted text
#     out_utt_list = sorted(
#         out_utt_list, 
#         key=lambda u: (u.split("-")[0], int(u.split("-")[1]), int(u.split("-")[2]), int(u.split(":")[0].split("-")[3]))
#     )
#     with open(pred_dir / "utt_fp_list.txt", "w") as f:
#         f.write("\n".join(out_utt_list))

#     # Calc score
#     if mode == "pred_all":
#         print("calc score...")
#         precision, recall, f_score, specificity = calc_score_all(
#             prediction_list,
#             target_list
#         )
#         out_text = \
#             "--- fp position ---\nprecision:\t{}\nrecall:\t{}\nf_score:\t{}\nspecificity:{}\n\n".format(
#                 precision, recall, f_score, specificity
#             )

#         precision_word = 0
#         recall_word = 0
#         f_score_word = 0
#         specificity_word = 0
#         rate_sum = 0
#         out_texts = []
#         for i in tqdm(range(len(fp_list))):
#             fp_rate = eval_fp_rate_dict[fp_list[i]]
#             rate_sum += fp_rate

#             precision, recall, f_score, specificity = calc_score_each_fp(
#                 prediction_list,
#                 target_list,
#                 i + 1
#             )
#             if precision is not None and torch.isnan(precision).sum() == 0:
#                 precision_word += precision * fp_rate
#             if recall is not None and torch.isnan(recall).sum() == 0:
#                 recall_word += recall * fp_rate
#             if f_score is not None and torch.isnan(f_score).sum() == 0:
#                 f_score_word += f_score * fp_rate
#             if specificity is not None and torch.isnan(specificity).sum() == 0:
#                 specificity_word += specificity * fp_rate

#             out_texts.append(
#                 "--- {} ---\nprecision:\t{}\nrecall:\t{}\nf_score:\t{}\nspecificity:{}\n".format(
#                     fp_list[i], precision, recall, f_score, specificity
#                 )
#             )

#         out_text += \
#             "--- fp word ---\nprecision:\t{}\nrecall:\t{}\nf_score:\t{}\nspecificity:{}\n\n".format(
#                 precision_word / rate_sum, 
#                 recall_word / rate_sum, 
#                 f_score_word / rate_sum, 
#                 specificity_word / rate_sum
#             ) + "\n".join(out_texts)

#         with open(pred_dir / "scores.txt", "w") as f:
#             print("writing score...")
#             f.write(out_text)

#     # check
#     with open(utt_list_path, "r") as f:
#         utt_list = sorted(
#             [l.strip() for l in f], 
#             key=lambda u: (u.split(":")[0], int(u.split(":")[1]), int(u.split(":")[2]))
#         )
#         utt_text = "".join([re.sub(r"\(F.*?\)", "", utt.split(":")[3].replace(" ", "")) for utt in utt_list])
#     out_utt_text = "".join([utt.split(":")[1] for utt in out_utt_list if not utt.split(":")[1].startswith("(F)")])
#     assert utt_text == out_utt_text, f"utt_text should be equal to out_utt_text\nutt_text:\n{utt_text}\n\nout_utt_text:\n{out_utt_text}"

@hydra.main(config_path="conf/evaluate", config_name="config")
def main(config: DictConfig):

    # Phase
    phase = "eval"

    # Out directory
    default_root_dir = Path(to_absolute_path(config[phase].default_root_dir))
    ckpt_dir = default_root_dir / "ckpt"
    out_dir = Path(to_absolute_path(config[phase].out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    train_config = OmegaConf.load(default_root_dir / "config.yaml")

    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Set rrandom seed
    pl.seed_everything(config.random_seed)

    # FPs
    fp_list_path = Path(to_absolute_path(config.data.fp_list))
    with open(fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Get fp rate
    eval_fp_rate_dict = {}
    eval_fp_rate_list_path = Path(to_absolute_path(config.data.eval.fp_rate_list))
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
        default_root_dir=default_root_dir,
        # profiler="simple",
    )

    # Predict
    evaluate(config, phase, trainer, pl_model, out_dir, fp_list, eval_fp_rate_dict)

    # elif config.corpus.name == "utokyo_naist_lecture":
    #     predict_utokyo_naist_lecture(config, phase, trainer, pl_model, out_dir, fp_list, eval_fp_rate_dict)

if __name__=="__main__":
    main()