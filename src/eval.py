import torch
from tqdm import tqdm
import sys
import re
from collections import Counter
import string
from torch.utils.data import DataLoader

#Taken from https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate_v1.1.py
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

from torch.cuda.amp import autocast
#-------------------------------------------------------------------------------------------------
import torch.nn as nn
from src.utils.trainer_utils import load_model_checkpoint, load_soft_prompt
from typing import Union
def evaluate_one_hop_wiki(model : Union[nn.Module],
                          tokenizer,
                          test_dataloader : DataLoader,
                          model_checkpoint_path : str,
                          device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    load_model_checkpoint(model, model_checkpoint_path, device)
        
        
    model.to(device)
    model.eval()
    total_em = 0
    total_f1 = 0
    prediction_vs_label = {}
    progress_bar = tqdm(test_dataloader, leave=True, desc=f"Test - Knowledge Integration", file = sys.stdout)
    with torch.no_grad():
        for batch_idx, (input_str, label) in enumerate(progress_bar):
            input_ids = tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
            attention_mask = tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['attention_mask'].to(device)
            
            outputs = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_length=50,
                                    num_beams = 5,
                                    early_stopping=True
                                    )

            
            decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
            _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, label)])
            em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, label)])
            
            total_em += em_score
            total_f1 += _f1_score
            prediction_vs_label[batch_idx] = f'Prediction: {decoded_predictions[0]} \n Label: {label[0]}'
        avg_em_perc = total_em / len(test_dataloader.dataset)
        avg_f1_perc = total_f1 / len(test_dataloader.dataset)
    print(f"Test - AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
    return prediction_vs_label
from models import SoftPromptModel, HyperbolicSoftPromptModel

def evaluate_random_walk_training(finetuned_model : SoftPromptModel,
                                  tokenizer,
                                  test_dataloader : DataLoader,
                                  #model_checkpoint_path : str,
                                  #hopping_soft_prompt_checkpoint_path : str,
                                  device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    #model.knit5 = load_model_checkpoint(model.knit5, model_checkpoint_path, device)
    #model.soft_prompt = load_soft_prompt(model.soft_prompt, hopping_soft_prompt_checkpoint_path, device)
    
    
    finetuned_model.to(device)
    finetuned_model.eval()
    total_em = 0
    total_f1 = 0
    prediction_vs_label = {}
    progress_bar = tqdm(test_dataloader, leave=True, desc=f"Test - Random Walk Training", file=sys.stdout)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            incomplete_sequence, complete_sequence = batch
            inputs = tokenizer(incomplete_sequence, padding=True, truncation=True, return_tensors = 'pt').to(device)
           
            outputs = finetuned_model.generate(input_ids=inputs.input_ids,
                                               attention_mask=inputs.attention_mask,
                                                max_length=50,
                                                num_beams = 5,
                                                early_stopping=True)
                
            decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
            

            

            _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, complete_sequence)])
            em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, complete_sequence)])
            
            
            total_em += em_score
            total_f1 += _f1_score
            
            progress_bar.set_description(f'Test - Random Walk Training - EM: {total_em / ((batch_idx+1) * test_dataloader.batch_size)} | F1: {total_f1 / ((batch_idx+1) * test_dataloader.batch_size)}')
            
            for idx, (pred, label) in enumerate(zip(decoded_predictions, complete_sequence)):
                index = batch_idx*test_dataloader.batch_size+idx
                prediction_vs_label[index] = {}
                prediction_vs_label[index]['prediction'] = pred
                prediction_vs_label[index]['label'] = label
            if batch_idx <= 5:
                print('\n', f'Prediction: {decoded_predictions[0]} \n Label: {complete_sequence[0]}')
            
        avg_em_perc = total_em / len(test_dataloader.dataset)
        avg_f1_perc = total_f1 / len(test_dataloader.dataset)
        
        
        
        print(f"Test - AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
        
        return prediction_vs_label
        
import random
def evaluate_parse_then_hop_training(parsing_model: SoftPromptModel,
                                     hopping_model: SoftPromptModel,
                                     tokenizer,
                                     test_dataloader : DataLoader,
                                     model_checkpoint_path : str,
                                     hopping_soft_prompt_checkpoint_path: str,
                                     parsing_soft_prompt_checkpoint_path: str,
                                     device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    parsing_model.knit5 = load_model_checkpoint(parsing_model.knit5, model_checkpoint_path, device, with_model_state_dict=False)
    parsing_model.soft_prompt = load_soft_prompt(parsing_model.soft_prompt, parsing_soft_prompt_checkpoint_path, device)
    
    hopping_model.knit5 = load_model_checkpoint(hopping_model.knit5, model_checkpoint_path, device, with_model_state_dict=False)
    hopping_model.soft_prompt = load_soft_prompt(hopping_model.soft_prompt, hopping_soft_prompt_checkpoint_path, device)
    
    parsing_model.to(device)
    hopping_model.to(device)
    parsing_model.eval()
    hopping_model.eval()
    total_em = 0
    total_f1 = 0
    progress_bar = tqdm(test_dataloader, leave=True, desc=f"Test - Parse Then Hop", file=sys.stdout)
    prediction_vs_label = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            question, complete_sequence = batch
            
            parts = complete_sequence.split(' ; ') # Strip each part to remove extra spaces
            if len(parts) > 5:
                parts[0] = '| '.join([parts[0], parts[1]]) # Some parts have ; in the first entity we replace it with a | 
            parts = [part.strip() for part in parts] #Some have double whitespaces we remove them and join with one whitespace
            complete_sequence = ' ; '.join(parts) 
            
            inputs_question = tokenizer(question, padding=True, truncation=True, return_tensors = 'pt').to(device)

            incomplete_sequence = parsing_model.generate(inputs=inputs_question)
            
            decoded_incomplete_sequence = tokenizer.batch_decode(incomplete_sequence, skip_special_tokens=True) 
            
            inputs_incomplete_sequence = tokenizer(decoded_incomplete_sequence, padding=True, truncation=True, return_tensors='pt').to(device)
            
            predictions = hopping_model.generate(inputs=inputs_incomplete_sequence)
            
            decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True) 

            _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, complete_sequence)])
            em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, complete_sequence)])
            
            total_em += em_score
            total_f1 += _f1_score
            prediction_vs_label[batch_idx] = f'Prediction: {decoded_predictions[0]} \n Label: {complete_sequence[0]}'
            
        avg_em_perc = total_em / len(test_dataloader.dataset)
        avg_f1_perc = total_f1 / len(test_dataloader.dataset)
        print(f"Test - AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")

            
        return prediction_vs_label

