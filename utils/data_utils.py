# Copyright 2022 Hyeon-Tae Seo, Su-Hyeon Kim, Sang-Ki Ko
# Kangwon National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, T5ForConditionalGeneration, RobertaTokenizer
import itertools

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

DRREPAIR_FMT = "./data/DrRepair_deepfix/{}.json"
DRREPAIR_Code2Align_FMT = "./data/DrRepair_deepfix_code2align/{}.json"

class DrRepairDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 split="val", data_len=None) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data = self.preprocessing(self.loas_dataset(DRREPAIR_FMT.format(split)))

    def loas_dataset(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    # def preprocessing(self, data):
    #     ret = []
    #     # max_length = 0
    #     # c_max_length = 0
    #     for key in tqdm(list(data.keys())):
    #         source = data[key]["source"]
    #         input_hf = self.tokenizer(source, padding=True, return_tensors='pt')
    #         # max_length = max(max_length, len(input_hf.input_ids[0]))
    #         for corrupt_source in data[key]["corrupt_source"]:
    #             target_hf = self.tokenizer(corrupt_source, padding=True, return_tensors='pt')
    #             # c_max_length = max(c_max_length, len(target_hf.input_ids[0]))
    #             ret.append({
    #                 "source": source,
    #                 "input":{
    #                     "input_ids": input_hf.input_ids,
    #                     "attention_mask": input_hf.attention_mask
    #                 },
    #                 "corrupt_source": corrupt_source,
    #                 "target": {
    #                     "input_ids": target_hf.input_ids,
    #                     "attention_mask": target_hf.attention_mask
    #                 },
    #             })
    #     # print("source max length: {}".format(max_length))
    #     # print("corrupt_source max length: {}".format(c_max_length))
    #     return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        return {
            "source": item["source"],
            "input": item["input"],
            "corrupt_source": item["corrupt_source"],
            "target": item["target"],
        }

    def get_collate_fn(self):
        def collate_fn(batch, pad_id):
            if len(batch) == 0:
                return None

            collated_batch = {
                # "source" : default_collate([ex["source"] for ex in batch]),
                "input":{
                    "input_ids": collate_tokens([ex["input"]["input_ids"] for ex in batch], pad_id),
                    "attention_mask": collate_tokens([ex["input"]["attention_mask"] for ex in batch], 0),
                },
                # "corrupt_source" : default_collate([ex["corrupt_source"] for ex in batch]),
                "target":{
                    "input_ids": collate_tokens([ex["target"]["input_ids"] for ex in batch], pad_id),
                    "attention_mask": collate_tokens([ex["target"]["attention_mask"] for ex in batch], 0),
                },
            }
            return collated_batch
        return functools.partial(collate_fn, pad_id=self.tokenizer.pad_token_id)


class DrRepairDatasetForCode2Code(DrRepairDataset):
    def __init__(self,
                 tokenizer,
                 split="val", data_len=None) -> None:
        super().__init__(tokenizer, split, data_len)
        self.data = self.preprocessing(self.loas_dataset(DRREPAIR_FMT.format(split)), data_len)

    def preprocessing(self, data, data_len):
        ret = []
        # max_length = 0
        # c_max_length = 0
        for key in tqdm(list(data.keys())[:data_len]):
            source = data[key]["source"]
            target_hf = self.tokenizer(source, padding=True, return_tensors='pt')
            # max_length = max(max_length, len(input_hf.input_ids[0]))
            for corrupt_source in data[key]["corrupt_source"]:
                input_hf = self.tokenizer(corrupt_source, padding=True, return_tensors='pt')
                # c_max_length = max(c_max_length, len(target_hf.input_ids[0]))
                ret.append({
                    "source": source,
                    "input":{
                        "input_ids": input_hf.input_ids,
                        "attention_mask": input_hf.attention_mask
                    },
                    "corrupt_source": corrupt_source,
                    "target": {
                        "input_ids": target_hf.input_ids,
                        "attention_mask": target_hf.attention_mask
                    },
                })
        # print("source max length: {}".format(max_length))
        # print("corrupt_source max length: {}".format(c_max_length))
        return ret



class DrRepairDatasetForCode2Align(DrRepairDataset):
    def __init__(self,
                 tokenizer,
                 split="val", data_len=None) -> None:
        super().__init__(tokenizer, split, data_len)
        # self.data =  dict(itertools.islice(self.loas_dataset(DRREPAIR_Code2Align_FMT.format(split)).items(), data_len))
        self.data = self.loas_dataset(DRREPAIR_Code2Align_FMT.format(split))[:data_len]

        # with open(output_file.format(split), 'r') as outfile:
        #     self.data = json.load(outfile)[:data_len]



# class DrRepairDatasetForCode2Align(DrRepairDataset):
#     def __init__(self,
#                  tokenizer,
#                  split="val", data_len=None) -> None:
#         super().__init__(tokenizer, split, data_len)
#         self.data = self.preprocessing(self.loas_dataset(DRREPAIR_FMT.format(split)), data_len)

#     def get_target(self, corrupt_program, program):
#         corrupt_program, program = corrupt_program.tolist()[0], program.tolist()[0]
#         log = getTrace(corrupt_program, program, getEditDistance(corrupt_program, program))
#         unchange_id, insert_id, delete_id = tuple(self.tokenizer.convert_tokens_to_ids(["[unchange]", "[insert]", "[delete"]))

#         target = [unchange_id for _ in range(len(corrupt_program))]
#         for l in log:
#             if l[0] == "i":
#                 target.insert(l[1], l[2])
#                 target.insert(l[1], insert_id)
#             elif l[0] == "r":
#                 target[l[1]] = l[2]
#             elif l[0] == "d":
#                 target[l[1]] = delete_id

#         return target

#     def preprocessing(self, data, data_len):
#         ret = []
#         for key in tqdm(list(data.keys())[:data_len]):
#             source = data[key]["source"]
#             source_token = self.tokenizer(source, padding=True, return_tensors='pt')
#             for corrupt_source in data[key]["corrupt_source"]:
#                 corrupt_source_token = self.tokenizer(corrupt_source, padding=True, return_tensors='pt')

#                 target = self.get_target(corrupt_source_token['input_ids'], source_token['input_ids'])
#                 target_ids = torch.tensor([target])
#                 target_mask = torch.tensor([[1 for _ in range(len(target))]])

#                 ret.append({
#                     "source": source,
#                     "input":{
#                         "input_ids": corrupt_source_token.input_ids,
#                         "attention_mask": corrupt_source_token.attention_mask
#                     },
#                     "corrupt_source": corrupt_source,
#                     "target": {
#                         "input_ids": target_ids,
#                         "attention_mask": target_mask
#                     },
#                 })
#         # print("source max length: {}".format(max_length))
#         # print("corrupt_source max length: {}".format(c_max_length))
#         return ret



if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok_name = "Salesforce/codet5-base"
    tokenizer = RobertaTokenizer.from_pretrained(tok_name)

    # train_ds = DrRepairDatasetForCode2Code(tokenizer, data_len=100)
    special_tokens_dict = {"additional_special_tokens": ['[u]', '[i]', '[d]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.vocab_files_names)
    exit()
    train_ds = DrRepairDatasetForCode2Align(tokenizer, data_len=10)

    print(len(train_ds))

    for idx, ex in zip(range(10), train_ds):
        print(ex['source'])
        print(ex['corrupt_source'])
        print(ex['input']['input_ids'])
        print([tokenizer.decode(tseq, skip_special_tokens=True) for tseq in ex['input']["input_ids"]])
        # print(tokenizer.decode(ex['input']['input_ids']), skip_special_tokens=True)
        print(ex['target']['input_ids'])
        print([tokenizer.decode(tseq, skip_special_tokens=False) for tseq in ex['target']["input_ids"]])
        # print(tokenizer.decode(ex['target']['input_ids']), skip_special_tokens=True)


        print(ex.keys())
        # print(ex)
