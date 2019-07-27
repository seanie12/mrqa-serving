import argparse

import flask
import torch
from pytorch_pretrained_bert import BertForQuestionAnswering, BertTokenizer, BertConfig
from model import DomainQA
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from iterator import convert_examples_to_features, write_predictions, SquadExample
import json
import collections

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def predict(model, tokenizer, item):
    """
    param model: pytorch pre-trained model
    param tokenizer : tokenizer for convert_examples_to_features
    param item : json_obj
    """
    # for the given passage, there are multiple questions (1 ~ many)
    doc_tokens = []
    eval_examples = []
    for token in item['context_tokens']:
        # BERT has only [SEP] in it's word piece vocabulary. because we keps all separators char length 5
        # we can replace all of them with [SEP] without modifying the offset
        if token[0] in ['[TLE]', '[PAR]', '[DOC]']:
            token[0] = '[SEP]'
        doc_tokens.append(token[0])

    # 2. qas
    for qa in item['qas']:
        qas_id = qa['qid']  # NOTE: 모든 데이터셋에 qid는 존재하고, unique하다
        question_text = qa['question']

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None)

        eval_examples.append(example)

    eval_features = convert_examples_to_features(eval_examples, tokenizer, max_seq_length=384,
                                                 doc_stride=128, max_query_length=64, is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_seg_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_seg_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=all_input_ids.size(0))
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    all_results = []

    for batch in eval_dataloader:
        input_ids, input_mask, seg_ids, example_indices = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        seg_ids = seg_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, seg_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    pred = write_predictions(eval_examples, eval_features, all_results,
                             n_best_size=20, max_answer_length=30, do_lower_case=True)

    print("prediction_file:", pred)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("port", type=int)
    # TODO: Model 파일이 매번 달라질 것이므로, 입력으로 받던지, 매번 동일한 모델이름으로 넣을 것인지 결정하기
    weight_file_name = os.listdir('./config/save/')[0]
    parser.add_argument("--model_path", type=str, default="./config/save/{}".format(weight_file_name),
                        help="pre-trained model path")
    parser.add_argument("--vocab_file", type=str, default="./config/large_vocab.txt", help="vocab file path")
    parser.add_argument("--config_file", type=str, default="./config/bert_large_config.json", help="config file path")
    parser.add_argument("--use_adv", default=True, type=bool, help="whether to use adversarially regularized model")
    parser.add_argument("--use_conv", action="store_true", help="whether to use conv discriminator")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda")
    else:
        print("Using CPU...")
        device = torch.device("cpu")

    config = BertConfig(args.config_file)
    if "large" in args.config_file:
        print("large model")
        hidden_size = 1024
    else:
        print("base model")
        hidden_size = 768

    if args.use_adv:
        model = DomainQA(config,
                         hidden_size=hidden_size,
                         use_conv=args.use_conv)
    else:
        model = BertForQuestionAnswering(config)

    # This part also have to be checked whether we are using the gpu or nots
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')

    model.load_state_dict(state_dict)
    # check whether there's avaible gpu device
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer(args.vocab_file)
    app = flask.Flask(__name__)


    @app.route('/', methods=['POST'])
    def index():
        json_obj = flask.request.get_json()
        pred = predict(model, tokenizer, json_obj)
        return pred


    app.run(port=args.port)
