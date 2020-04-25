import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from tqdm.auto import trange, tqdm
import os
from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

output_dir = ('./model_save')
model = BertForNextSentencePrediction.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

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
            #answerable = qa['answerable']
            self.data.append((qa_id, context, question))# answerable))
  
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    qa_id, context, question = self.data[index]  #, answerable
    return qa_id, context, question#, int(answerable)

test_dataset = EarlyDataset("./dev.json", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

best_valid_loss = float('inf')
all_predictions = {}

model.eval()
with torch.no_grad():
    pbar=tqdm(test_loader)
    for batch in pbar:
        ids, contexts, questions = batch
        input_dict = tokenizer.batch_encode_plus(contexts, questions, 
                                                 max_length=tokenizer.max_len, 
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        with torch.no_grad():
            logits = model(**input_dict)[0]
            probs = logits.softmax(-1)[:, 1]
            print(probs)
            all_predictions.update(
               {
                    uid: 'answer' if prob > 0.66 else ''

                    for uid, prob in zip(ids, probs)
               }
             )

Path("./predict.json").write_text(json.dumps(all_predictions))