import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import os

max_epoch = 3
batch_size = 4
lr = 5e-6
weight_decay = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert_pretrain_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_pretrain_name , do_lower_case=True)
model = BertForQuestionAnswering.from_pretrained(bert_pretrain_name).to(device)
optim = AdamW(model.parameters(), lr)

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
            # answers = qa['answers']   #dict array  [{'id': '1', 'text': '10秒鐘', 'answer_start': 84}],
            text = qa['answers'][0]['text']
            #start = int(qa['answers'][0]['answer_start'])
            #end = start+len(text)-1
            answerable = qa['answerable']
            self.data.append((qa_id, context, question, text, answerable))
  
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question, text, answerable = self.data[index]
    return qa_id, context, question, text, int(answerable)

if __name__ == "__main__":

  def startfinder(mylist, pattern):
    for i in range(len(mylist)):
      if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
        return i, i+len(pattern)-1
    return 0,0

  train_dataset = EarlyDataset("../train.json", tokenizer)
  valid_dataset = EarlyDataset("../dev.json", tokenizer)

  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

  version = 1
  for epoch in trange(max_epoch):
    output_dir = '../model_save_'+str(version)
    print(output_dir)
    pbar = tqdm(train_loader)
    for batch in pbar:
      ids, contexts, questions, text, answerable = batch
      train_input = []
      for i in range(batch_size):
        context = ()
        question_len = len(questions[i])
        context_max_len = 509 - question_len
        if len(contexts[i])>context_max_len:      #truncate
          context=contexts[i][:context_max_len]
        else:
          context=contexts[i]
        train_input.append([context, questions[i]])     

      input_dict = tokenizer.batch_encode_plus(train_input,
                                              max_length=tokenizer.max_len, #512
                                              pad_to_max_length=True,
                                              return_tensors='pt')
      input_dict = {k: v.to(device) for k, v in input_dict.items()}

      start_list = []
      end_list = []
      for i in range(batch_size):
        text_encode = tokenizer.encode(text[i])
        text_encode = text_encode[1:-1]
        if text_encode!=[]:  
          start, end = startfinder(input_dict["input_ids"][i].tolist(), text_encode)
          start_list.append(start)
          end_list.append(end)
        else:
          start_list.append(0)
          end_list.append(0)

      loss, start_scores, end_scores = model(**input_dict,
                                            start_positions=torch.tensor(start_list).to(device),
                                            end_positions=torch.tensor(end_list).to(device)
                                            )
      loss.backward()
      optim.step()
      optim.zero_grad()
      
      pbar.set_description(f"loss: {loss.item():.4f}")

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    version+=1
  print("DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEs")