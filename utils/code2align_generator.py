import torch
from tqdm import tqdm

from helpers import getTrace, getEditDistance
from transformers import AutoTokenizer, RobertaTokenizer
import json


def get_target(corrupt_program, program, tokenizer):
    corrupt_program, program = corrupt_program.tolist()[0], program.tolist()[0]
    log = getTrace(corrupt_program, program, getEditDistance(corrupt_program, program))
    unchange_id, insert_id, delete_id = tuple(tokenizer.convert_tokens_to_ids(["[u]", "[i]", "[d]"]))

    target = [unchange_id for _ in range(len(corrupt_program))]
    for l in log:
        if l[0] == "i":
            target.insert(l[1], l[2])
            target.insert(l[1], insert_id)
        elif l[0] == "r":
            target[l[1]] = l[2]
        elif l[0] == "d":
            target[l[1]] = delete_id

    return target

def preprocessing(data, tokenizer):
    ret = []
    for key in tqdm(list(data.keys())):
        source = data[key]["source"]
        source_token = tokenizer(source, padding=True, return_tensors='pt')

        for corrupt_source in data[key]["corrupt_source"]:
            corrupt_source_token = tokenizer(corrupt_source, padding=True, return_tensors='pt')

            target = get_target(corrupt_source_token['input_ids'], source_token['input_ids'], tokenizer)
            target_ids = torch.tensor([target])
            target_mask = torch.tensor([[1 for _ in range(len(target))]])

            ret.append({
                "source": source,
                "input":{
                    "input_ids": corrupt_source_token.input_ids.tolist(),
                    "attention_mask": corrupt_source_token.attention_mask.tolist()
                },
                "corrupt_source": corrupt_source,
                "target": {
                    "input_ids": target_ids.tolist(),
                    "attention_mask": target_mask.tolist()
                },
            })
    return ret

def loas_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def code2align_generator(split, tokenizer):
    split = split
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
    special_tokens_dict = {"additional_special_tokens": ['[u]', '[i]', '[d]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    input_file = "./data/DrRepair_deepfix/{}.json"
    output_file = "./data/DrRepair_deepfix_code2align/{}.json"
    data = preprocessing(loas_dataset(input_file.format(split)), tokenizer)
    with open(output_file.format(split), 'w') as outfile:
        json.dump(data, outfile)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",
                        default="Salesforce/codet5-base", type=str)
    parser.add_argument("--split",
                        default="train", type=str)
    ## train or val
    args = parser.parse_args()


    code2align_generator(args.split, args.tokenizer)