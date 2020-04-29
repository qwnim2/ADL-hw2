import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import os
from pathlib import Path
from argparse import Namespace,ArgumentParser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

output_dir = ('./model_save_3/')
parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()

model = BertForQuestionAnswering.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

model.to(device)
batch_size = 4

class EarlyDataset(Dataset):
  def __init__(self, path: str, tokenizer: BertTokenizer) -> None:
    self.tokenizer = tokenizer
    self.data = []
    with open(path) as f:
      for article in json.load(f)['data']:
        parapraphs = article['paragraphs']
        for para in parapraphs:
          context = para['context']
          for qa in para['qas']:
            qa_id = qa['id']
            question = qa['question']
            self.data.append((qa_id, context, question))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question = self.data[index]
    return qa_id, context, question

test_dataset = EarlyDataset(args.test_data_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_predictions = {}

model.eval()
with torch.no_grad():
  pbar=tqdm(test_loader)
  for batch in pbar:
    ids, contexts, questions = batch
    eval_input = []
    for i in range(len(ids)):
      context = ()
      #print(i)
      question_len = len(questions[i])
      #print(questions[i])
      context_max_len = 509 - question_len
      #print(context_max_len)
      if len(contexts[i])>context_max_len:      #truncate
        context=contexts[i][:context_max_len]
      else:
        context=contexts[i]
      eval_input.append([context, questions[i]])
    input_dict = tokenizer.batch_encode_plus(eval_input,
                                              max_length=tokenizer.max_len, 
                                              pad_to_max_length=True,
                                              return_tensors='pt')
    input_dict = {k: v.to(device) forㄋ k, v in input_dict.items()}
    start_list ,end_list = model(**input_dict, start_positions=None, end_positions=None)
    
    for i in range(len(ids)):
      if start_list[i].argmax() < 0 or end_list[i].argmax():
        start = 0
        end = 0
      else:
        start = start_list[i].argmax()
        end = end_list[i].argmax()
      if end <= start or end-start>30:
        answer = ""
      else:
        answer = "".join(tokenizer.convert_ids_to_tokens(input_dict['input_ids'][i][start:end+1]))
      answer = answer.replace("#","")
      for i in range(len(answer)):
        if answer[i]=="『":
          if "』" not in answer[i:]:
            answer = answer.replace("『","")
        if answer[i]=="「":
          if "」"not in answer[i:]:
            answer = answer.replace("「","")
      all_predictions[ids[i]]=answer


Path(args.output_path).write_text(json.dumps(all_predictions))