import re
import pandas as pd


def structure_files(files, filepath):
    """removes whitespace lines and non character lines from the subtitle files and returns a list of all the sequences"""
    all_text = []
    for i in files:
        for x in i[2]:
            file = open(f"{filepath}/{x}", "r", encoding='cp1253')
            lines = file.readlines()
            file.close()

            text = []
            for line in lines:
                if re.search('^[0-9]+$', line) is None and re.search('^[0-9]{2}:[0-9]{2}:[0-9]{2}', line) is None and re.search('^$', line) is None:
                    text.append(line.rstrip('\n').lower())
            all_text += text[:-4]
    return all_text


def create_clean_list(text_list, character_names):
    """takes a list of strings and removes unwanted characters"""
    text = " ".join(text_list)
    data = []
    text = re.split('[.?;]', text)
    for i in text:
        i = re.sub(r"(\[.*?\])", "", i)
        i = re.sub(r"\d", "", i)
        for name in character_names:
            i = re.sub(f"{name}", "", i)
        i = i.strip()
        if i != "" and len(i) >= 5:
            data.append(i)
    return data


def get_vocab_size(data):
    """takes a list of clean strings and returns unique number of tokens"""
    return len(set(" ".join(data).split()))


def create_dataframe(data):
    """takes a list of clean strings and returns a dataframe of input_sequences and output_sequences"""
    q = []
    a = []

    for i, x in enumerate(data):
        if i % 2 == 0:
            q.append([x])
        else:
            a.append([x])

    questions_df = pd.DataFrame(data=q, columns=["input"])
    responses_df = pd.DataFrame(data=a, columns=["output"])
    df = pd.concat([questions_df, responses_df], axis=1)
    return df
