# Group-dependent filled pause prediction models and training scripts

This is a source implementation of group-dependent filled pause (FP) prediction on the basis of FP usage of speakers in Corpus of Spontaneous Japanese (CSJ). There are 8 group-dependent models on the basis of FP words and positions in ``model_files`` (detailed in [here](#models))

## Requirements

- You can install the Python requirements with
    ```
    $ pip install -r requirements.txt
    ```
  - Our recommendation of the Python version is ``3.8``.

- Install BERT model to the directory ``bert/`` from [here](https://nlp.ist.i.kyoto-u.ac.jp/?ku_bert_japanese). We use pytorch-pretrained-BERT with LARGE WWM version.

## Group-dependent filled pause prediction models

We describe the detailed process of grouping speakers and training models in [paper](#Citation).

### Models

Group-dependent filled pause prediction models are available at ``model_files/``. File names and model descriptions are listed below. Model files follows ``pytorch-lightning`` format. We recommend using ``predict.py`` to get prediction results (detailed in [here](#prediction)).

| filename (``model_files/``)  | description          |
| ---                    | ---                  |
| word_group1.ckpt       | group 1 (word)       |
| word_group2.ckpt       | group 2 (word)       |
| word_group3.ckpt       | group 3 (word)       |
| word_group4.ckpt       | group 4 (word)       |
| position_group1.ckpt   | group 1 (position)   |
| position_group2.ckpt   | group 2 (position)   |
| position_group3.ckpt   | group 3 (position)   |
| position_group4.ckpt   | group 4 (position)   |

## Preparation for training

### CSJ
Install CSJ to the directory ``corpus/`` from [here](https://ccd.ninjal.ac.jp/csj/en/). We need transcription files of ``core`` and ``noncore`` data with ``Form1``.

## Training

### Step 1: Get CSJ information
The script ``get_csj_info.py`` get the list of the pairs of speaker and lecture id and the list of the core speakers.
```
$ python get_csj_info.py path/to/CSJ path/to/CSJ/fileList.csv
```

### Step 2: Preprocess
The script ``preprocess.py`` gets the list of utterances from the transcription files, segments them to morphemes, extracts features, splits them to training, validation, and evaluaitio data, and gets the frequency of FPs. This follows the setting written in ``conf/preprocess/config.yaml``. Change the setting accordingly.
```
$ python preprocess.py
```

### Step 3: Training
The script ``train.py`` train the non-personalized model or group-dependent models. This follows the setting written in ``conf/train/config.yaml``. Change the setting accordingly.
```
$ python train.py
```
1. Train the non-personalized model. Write the following in ``conf/train/config.yaml``.
```
train:
    model_type: non_personalized
    fine_tune: False
```
2. Train the group-dependent models. Write the following in ``conf/train/config.yaml``.
```
train:
    model_type: group
    group_id: <group_id>
    fine_tune: True
    load_ckpt_step: <step>
```

## Evaluation
The script ``evaluate.py`` evaluate prediction performance of the models. This follows the setting written in ``conf/evaluate/config.yaml``. Change the setting accordingly.
```
$ python evaluate.py
```
1. Evaluate the non-personalized model. Write the following in ``conf/evaluate/config.yaml``.
```
eval:
    model_type: non_personalized
```
2. Evaluate the group-dependent models. Write the following in ``conf/evaluate/config.yaml``.
```
eval:
    model_type: group
    group_id: <group_id>
```

## Prediction
The script ``predict.py`` predict FPs for new data.
1. Prepare a file of utterance list and run the script of preprocess. You can see an example of the utterance list to predict FPs in ``preprocessed_data/example``. This follows the setting written in ``conf/preprocess_test/config.yaml``. Change the setting accordingly.
```
$ python preprocess_test.py
```
2. Then, run the script of prediction. This follows the setting written in ``conf/predict/config.yaml``. Change the setting accordingly.
```
$ python predict.py
```

## Contributors
- [Yuta Matsunaga](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/home) (The University of Tokyo, Japan) [main contributor]
- [Takaaki Saeki](https://takaaki-saeki.github.io/) (The University of Tokyo, Japan)
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (The University of Tokyo, Japan)
- [Hiroshi Saruwatari](https://researchmap.jp/read0102891/) (The University of Tokyo, Japan)

## Citation
```
Coming soon...
```