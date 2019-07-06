from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering, BertConfig

config_file = "../config/bert_base_config.json"
vocab_file = "../config/vocab.txt"
config = BertConfig(config_file)
model = BertForQuestionAnswering(config)

tokenizer = BertTokenizer(vocab_file)
print(tokenizer.vocab["i"])

for k,v in model.state_dict().items():
    print(k)
