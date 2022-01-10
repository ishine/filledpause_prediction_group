import pandas as pd

if __name__ == "__main__":
    corpus_info_path = "./corpus/CSJ/fileList.csv"
    speaker_koen_list_path = "./corpus/CSJ/speaker_koen.list"
    speaker_list_path = "./corpus/CSJ/speaker.list"

    df = pd.read_csv(corpus_info_path)

    # get list of pair (speaker id, koen ids) in CSJ
    speaker_koen_list = []
    for speaker_id, koen_ids in df.groupby(by=["講演者ID"])["講演ID"].unique().to_dict().items():
        speaker_koen_list.append(
            str(speaker_id) + ":" + ",".join(koen_ids))
    with open(speaker_koen_list_path, "w") as f:
        f.write("\n".join(speaker_koen_list))
        
    # get list of core persons in CSJ
    speaker_list = list(df[df["コア"]=="コア"]["講演者ID"].unique())
    speaker_list = sorted(speaker_list, key=lambda x: int(x))
    with open(speaker_list_path, "w") as f:
        f.write("\n".join([str(p) for p in speaker_list]))