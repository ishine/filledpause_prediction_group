# General
import random
from pathlib import Path
# Config
import hydra
from omegaconf import DictConfig, OmegaConf
# My library
from fp_pred_group.preprocessor import process_morph, extract_feats_test

@hydra.main(config_path="conf/preprocess_test", config_name="config")
def main(config: DictConfig):
    # Set random seed
    random.seed(config.random_seed)

    # Save config
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Preprocess
    print("process morphs...")
    process_morph(data_dir)
    print("extract features...")
    extract_feats_test(data_dir, config.fp_list_path, config.bert_model_dir, "utt_morphs")

if __name__ == "__main__":
    main()