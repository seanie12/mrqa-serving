import argparse

import flask
import torch
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from pytorch_pretrained_bert import BertForQuestionAnswering, BertTokenizer, BertConfig
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from iterator import convert_examples_to_features, write_predictions, SquadExample
import json
import collections


def predict(model, tokenizer, item):
    """
    param model: pytorch pre-trained model
    param tokenzier : tokenizer for convert_examples_to_features
    param item : json_obj
    """
    doc_tokens = []
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
    eval_examples = [example]  # item has only one instance
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
    parser.add_argument("--model_path", type=str, default="../config/save/base_0_1.230",help="pretrained model path")
    parser.add_argument("--vocab_file", type=str, default="../config/vocab.txt", help="vocab file path")
    parser.add_argument("--config_file", type=str, default="../config/bert_base_config.json", help="config file path")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    config = BertConfig(args.config_file)
    model = BertForQuestionAnswering(config)
    state_dict = torch.load(args.model_path)
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
