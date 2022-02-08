# Group-dependent filled pause prediction models and training scripts

This is a source implementation of group-dependent filled pause (FP) prediction on the basis of FP usage of speakers in Corpus of Spontaneous Japanese (CSJ). There are 8 group-dependent models on the basis of FP words and positions in ``model_files`` (detailed in [here](#models))

## Group-dependent filled pause prediction models

Explanation of clustering and model training. Coming soon...

### Models

Group-dependent filled pause prediction models are available at ``model_files/``. File names and model descriptions are listed below. Model files follows ``pytorch-lightning`` format. We recommend using ``predict.py`` to get prediction results (detailed in [here](#prediction)).

| filename (``model_files/``)  | description          |
| ---                    | ---                  |
| word_model1.ckpt       | model 1 (word)       |
| word_model2.ckpt       | model 2 (word)       |
| word_model3.ckpt       | model 3 (word)       |
| word_model4.ckpt       | model 4 (word)       |
| position_model1.ckpt   | model 1 (position)   |
| position_model2.ckpt   | model 2 (position)   |
| position_model3.ckpt   | model 3 (position)   |
| position_model4.ckpt   | model 4 (position)   |

## Preparation for training

### BERT
Install BERT model to the directory ``bert/`` from [here](https://nlp.ist.i.kyoto-u.ac.jp/?ku_bert_japanese). We use pytorch-pretrained-BERT with LARGE WWM version.

### CSJ
Install CSJ to the directory ``corpus/`` from [here](https://ccd.ninjal.ac.jp/csj/en/). We need transcription files of ``core`` and ``noncore`` data with ``Form1``.

## Training

### Step 1: Get CSJ information
The script ``get_csj_info.py`` get the list of the pairs of speaker and lecture id and the list of the core speakers.
```
python get_csj_info.py path/to/CSJ path/to/CSJ/fileList.csv
```

### Step 2: Preprocess
The script ``preprocess.py`` gets the list of utterances from the transcription files, segments them to morphemes, extracts features, splits them to training, validation, and evaluaitio data, and gets the frequency of FPs. This follows the setting written in ``conf/preprocess/config.yaml``. Change the setting accordingly.
```
python preprocess.py
```

### Step 3: Training
The script ``train.py`` train the non-personalized model or group-dependent models. This follows the setting written in ``conf/train/config.yaml``. Change the setting accordingly.
```
python train.py
```

## Evaluation
The script ``evaluate.py`` evaluate prediction performance of the models. This follows the setting written in ``conf/evaluate/config.yaml``. Change the setting accordingly.
```
python evaluate.py
```

## Prediction
The script ``predict.py`` predict FPs for new data. Prepare a file of utterance list and run script. This follows the setting written in ``conf/predict/config.yaml``. Change the setting accordingly.
```
python predict.py
```

## Contributors
- [Yuta Matsunaga](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/home) (The University of Tokyo, Japan) [main contributor]
- [Takaaki Saeki](https://takaaki-saeki.github.io/) (The University of Tokyo, Japan)
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (The University of Tokyo, Japan)
- Saruwatari Hiroshi (The University of Tokyo, Japan)

## Citation
```
@INPROCEEDINGS{yamazaki20blstmfillerprediction,  
    author={Yamazaki, Yoshihiro and Chiba, Yuya and Nose, Takashi and Ito, Akinori},  
    booktitle={Proc. GCCE},   
    title={Filler Prediction Based on Bidirectional LSTM for Generation of Natural Response of Spoken Dialog},   
    year={2020},  
    pages={360--361},  
    doi={10.1109/GCCE50665.2020.9291867}
}
@inproceedings{morita15jumanpp,
    title = "Morphological Analysis for Unsegmented Languages using Recurrent Neural Network Language Model",
    author = "Morita, Hajime  and
      Kawahara, Daisuke  and
      Kurohashi, Sadao",
    booktitle = "Proc. EMNLP",
    year = "2015",
    pages = "2292--2297",
    doi = "10.18653/v1/D15-1276",
}
@article{maekawa04csj,
    author="Maekawa, K.",
    title="Corpus of Spontaneous Japanese : its design and evaluation",
    journal="Proc. SSPR",
    year="2003",
    pages="7--12",
    URL="https://ci.nii.ac.jp/naid/10013308127/",
}
@article{devlin19bert,
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}, 
      author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
      year={2019},
      journal={arXiv},
      volume={abs/1810.04805},
}
```