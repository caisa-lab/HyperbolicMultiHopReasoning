import torch
from tqdm import tqdm

def compute_exact_match(pred, truth):
    return int(pred.strip() == truth.strip())

def evaluate_single_hop(one_hop_wiki_test_dataloader, model, tokenizer, device):
    model.eval()
    total_em = 0
    total_count = 0
    progress_bar = tqdm(one_hop_wiki_test_dataloader, leave=True, desc=f"Evaluation - One Hop Wiki")
    with torch.no_grad():
        for (questions, answers) in progress_bar:
            inputs = tokenizer(questions, padding=True, truncation=True, return_tensors='pt').to(device)
            
            outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length = 512)
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            for pred, truth in zip(predictions, answers):
                total_em += compute_exact_match(pred, truth)
                total_count += 1
            
    return total_em / total_count