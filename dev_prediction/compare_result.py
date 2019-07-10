import json

result_file_name = 'compare_result.txt'

dataset_name_lst = ["BioASQ", "DROP", "DuoRC", "RACE", "RelationExtraction", "TextbookQA"]
f1_score = [62.7, 34.5, 54.6, 41.4, 83.8, 53.9]

with open(result_file_name, 'w', encoding='utf-8') as f_w:
    for idx, dataset_name in enumerate(dataset_name_lst):
        with open("eval_{}.json".format(dataset_name), 'r', encoding='utf-8') as f_r:
            result = json.load(f_r)
            f_w.write("[{}] orig: {} result: {:.1f}\n".format(dataset_name, f1_score[idx], result["f1"]))
