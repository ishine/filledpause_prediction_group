import numpy as np
import torch
from torch.utils.data import Dataset

from fp_pred_group.util.train_util import pad_1d, pad_2d

class MyDataset(Dataset):
    def __init__(self, in_paths, out_paths):
        self.in_paths = in_paths
        self.out_paths = out_paths

    def __getitem__(self, index):
        in_feat = np.load(self.in_paths[index]).astype(np.float32)
        out_feat = np.load(self.out_paths[index]).astype(np.float32)

        return in_feat, out_feat

    def __len__(self):
        return len(self.in_paths)

    def collate_fn(self, batch):
        lengths = [len(x[0]) for x in batch]
        max_len = max(lengths)
        x_batch = torch.stack([torch.from_numpy(pad_2d(x[0], max_len)) for x in batch])
        y_batch = torch.stack([torch.from_numpy(pad_1d(x[1], max_len)) for x in batch])
        return x_batch, y_batch

class NoFPDataset(Dataset):
    def __init__(self, in_paths, out_paths, utt_list_path=None):
        if utt_list_path is not None:
            self.text_dict = {}
            with open(utt_list_path, "r") as f:
                for l in f:
                    utt = l.strip()
                    if len(utt) > 0:
                        utt_name = "-".join(utt.split(":")[:-1])
                        text = " ".join([
                            w for w in utt.split(":")[-1].split(" ") 
                            if not w.startswith("(F")])
                        # text = re.sub(r"\(F.*?\)", "", utt.split(":")[-1])
                        self.text_dict[utt_name] = text

        self.in_paths = in_paths
        self.out_paths = out_paths

    def __getitem__(self, index):
        in_feat = np.load(self.in_paths[index]).astype(np.float32)
        out_feat = np.load(self.out_paths[index]).astype(np.float32)
        in_text = self.text_dict[self.in_paths[index].stem.replace("-feats", "")]
        sample = {
            "feat": in_feat, 
            "out_feat": out_feat, 
            "text": in_text,
        }
        return sample

    def __len__(self):
        return len(self.in_paths)

    def collate_fn(self, batch):
        lengths = [len(x["feat"]) for x in batch]
        max_len = max(lengths)
        x_batch = torch.stack([torch.from_numpy(pad_2d(x["feat"], max_len)) for x in batch])
        y_batch = torch.stack([torch.from_numpy(pad_1d(x["out_feat"], max_len)) for x in batch])
        text_batch = [x["text"] for x in batch]
        return x_batch, y_batch, text_batch