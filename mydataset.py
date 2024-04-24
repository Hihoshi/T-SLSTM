from torch.utils.data import Dataset, DataLoader
from tokenizer import tokenizer
import torch
import json
from tqdm import tqdm


def construct(corpus):
    data = []
    label = []
    process_of_constructing = tqdm(corpus["pairs"], bar_format='{l_bar}{bar:32}{r_bar}', leave=False)
    process_of_constructing.set_description(("Constructing Dataset"))
    for i in process_of_constructing:
        en = i["en"]
        zh = i["zh"]
        for idx, token_id in enumerate(zh):
            data.append(en + zh[:idx])
            label.append(token_id)
    return data, label


def tokenize(corpus):
    process_of_tokenizing = tqdm(corpus["pairs"], bar_format='{l_bar}{bar:32}{r_bar}', leave=False)
    process_of_tokenizing.set_description(("Tokenizing Data"))
    for i in process_of_tokenizing:
        i["en"] = tokenizer(i["en"])["input_ids"]
        i["zh"] = tokenizer(i["zh"])["input_ids"]
    return corpus


class MyDataset(Dataset):
    def __init__(self, path: str, cache_tokenized=True, use_cached=False):

        # load json file
        with open(path, "r", encoding='utf8') as f:
            corpus = json.load(f)

        # if use cached dataset, path should be the cached json file
        if use_cached is True:
            self.data, self.label = construct(corpus)
            return

        # tokenize the dataset
        corpus = tokenize(corpus)
        # save cached dataset
        if cache_tokenized is True:
            print("Saving tokenized data")
            with open("cached.json", "w", encoding="ascii") as f:
                json.dump(corpus, f)
            print("Tokenized data cached")

        self.data, self.label = construct(corpus)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


# def customized function to fetch batch
def custom_collate_fn(batch):
    inps = [data[0] for data in batch]
    tgts = [data[1] for data in batch]
    max_len = max([len(inp) for inp in inps])
    for i in range(len(inps)):
        if len(inps[i]) < max_len:
            inps[i] += [0 for j in range(max_len - len(inps[i]))]
    return torch.tensor(inps, dtype=torch.int32).transpose(0, 1).unsqueeze(2), torch.tensor(tgts, dtype=torch.int32)


if __name__ == '__main__':
    dataloader = DataLoader(
        MyDataset("cached.json", use_cached=True),
        batch_size=4,
        collate_fn=custom_collate_fn
    )
    for data, label, index in dataloader:
        print(data.shape)
        print(label.shape)
        print(index)
