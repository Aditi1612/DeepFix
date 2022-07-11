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


import sys
import os
import numpy as np
import json
import copy
from tqdm import tqdm
import glob
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.helpers import make_dir_if_not_exists

def remove_line_numbers(source_dict):
    ret = []
    for _, v in source_dict.items():
        ret.append(v)
    return ' '.join(ret)

def generate_training_data(bins, validation_users):
    data_path = "./data_processing/DrRepair_deepfix/"
    result = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    for problem_id in tqdm(problem_list):
        for data_file in glob.glob(data_path+problem_id+"/*"):
            try:
                data = json.loads(open(data_file).read())
            except:
                exceptions_in_mutate_call += 1
                continue

            code_id = data["meta"]["subid"].split("-")[0]
            user_id = data["meta"]["subid"].split("-")[1]
            key = 'validation' if user_id in validation_users[problem_id] else 'train'
            code_list = dict()
            for lines in data["lines"]:
                code_list[lines["line"]] = lines["code"]

            source = remove_line_numbers(code_list)

            if problem_id not in result[key].keys():
                result[key][problem_id+user_id+code_id] = {"source": source}

            # Mutate
            for iter_i in range(len(data["errors"])):
                temp = copy.deepcopy(code_list)
                for mod_line, mod_code in zip(data["errors"][iter_i]['mod_line'],
                        data["errors"][iter_i]['mod_code']):
                    temp[mod_line] = mod_code

                corrupt_source = remove_line_numbers(temp)

                try:
                    result[key][problem_id+user_id+code_id]["corrupt_source"].append(corrupt_source)
                except:
                    result[key][problem_id+user_id+code_id]["corrupt_source"] = [corrupt_source]

    print("Exceptions in mutate() call: {}".format(exceptions_in_mutate_call))
    return result

if __name__ == '__main__':
    validation_users = np.load(os.path.join('./data', 'deepfix_raw_data', 'validation_users.npy'), allow_pickle=True).item()
    bins = np.load(os.path.join('./data', 'deepfix_raw_data', 'bins.npy'), allow_pickle=True)

    output_dir = os.path.join('./data', 'DrRepair_deepfix')
    make_dir_if_not_exists(os.path.join(output_dir))

    result = generate_training_data(bins, validation_users)

    with open(output_dir+'/train.json', 'w') as f:
        json.dump(result['train'], f, indent=4)
    with open(output_dir+'/val.json', 'w') as f:
        json.dump(result['validation'], f, indent=4)

    print('\n\n--------------- all outputs written to {} ---------------\n\n'.format(output_dir))
