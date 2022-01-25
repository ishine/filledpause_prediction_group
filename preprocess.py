# General
import random
from pathlib import Path
# Config
import hydra
from omegaconf import DictConfig, OmegaConf
# My library
from preprocessor import extract_feats, split_data, process_tagtext, analyze_fp

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):
    # Set random seed
    random.seed(config.random_seed)

    # Save config
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Preprocess
    print("process tagtext...")
    process_tagtext(config)
    print("extract features...")
    extract_feats(config)
    print("split data...")
    split_data(config)
    print("analyze filler...")
    analyze_fp(config)

if __name__ == "__main__":
    main()