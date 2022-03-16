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
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Preprocess
    print("process morphs...")
    process_morph(config)
    print("extract features...")
    extract_feats_test(config, "utt_morphs")

if __name__ == "__main__":
    main()