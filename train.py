import torch.nn as nn
import torch

def train_single_hop(model, tokenizer, optimizer, dataloader, epochs):
    """Trains the Knowledge Integration. The model becomes e1 ; r1 as an input and tries to predict e2.
        For the Dataset we will use the SingleHopDataset which gets a list of [e1, r1, e2] an turns it into input_ids and attention mask for "e1 ; r1" and the label input_ids for "e2"
        In this part the Model will be Finetuned.
    Tracks only the Loss for now.
    """
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_str, label = batch
            input_ids = tokenizer(input_str, return_tensors='pt')['input_ids']
            attention_mask = tokenizer(input_str, return_tensors='pt')['attention_mask']
            labels = tokenizer(label, return_tensor='pt')['input_ids']
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
def random_walk_training(model, hp_length, embedding_size, tokenizer, optimizer, dataloader, epochs):
    """Trains the Random Walk Part. Random Walk takes in a sequence "e1 ; r1 ; r2" and should predict "e1 ; r1 ; e2 ; r2 ; e3"
       In this part the Soft Prompt HP will be Finetuned and the Model will be frozen.
       
       For the Dataset we will use the RandomWalk Dataset which contains gives as a complete and an incomplete path
    """
    
    #Freeze Model in Random Walk Training
    for param in model.parameters():
        param.required_grad = False
        
    #HP Soft Prompt will be tuned
    hp_embeddings = nn.Parameter(torch.randn(hp_length, embedding_size))
    hp_embeddings.requires_grad = True
    
    for epoch in range(epochs):
        total_loss = 0
        for incomplete_sequence, complete_sequence in dataloader:
            inputs = tokenizer(incomplete_sequence, return_tensors = 'pt')
            labels = tokenizer(complete_sequence, return_tensors = 'pt')['input_ids']
            
            #Generate HP Embedding and concatenate with input IDs
            hp_input = hp_embeddings.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1)
            concatenated_input = torch.cat([hp_input, inputs['input_ids']], dim=1)
            
            #Adjust attention mask (take all of the soft prompt tokens should be attented)
            hp_attention_mask = torch.ones((inputs['attention_mask'].size(0), hp_length), device=inputs['attention_mask'])
            concatenated_attention_mask = torch.cat((hp_attention_mask, inputs['attention_mask']), dim=1)
            
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                outputs = model(input_ids=concatenated_input, attention_mask=concatenated_attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

#TODO: How to do both steps parsing and hopping in one training loop or should it be in one training loop?
def parse_then_hop(model, hp_embeddings, pp_length, pp_embedding_size, tokenizer, optimizer, dataloader, epochs):
    """
    Finetunes a Soft Prompt Parsing Prompts (PP) everything else is kept frozen. Takes in a Question and the PP gives it to the model which should output an incomplete path.
    This incomplete path should be concatonated with the Hopping Prompt (HP) and then fed into the model which should output the complete path.
    """
    
    #Freeze model for PaTH Training
    for param in model.parameters():
        param.required_grad = False
    
    #Freeze HP For Path Training
    hp_embeddings.requires_grad = False
    
    #PP Soft Prompt will be tuned
    pp_embeddings = nn.Parameter(torch.randn(pp_length, pp_embedding_size))
    pp_embeddings.requires_grad = True
    
    for epoch in range(epochs):
        total_parsing_loss = 0
        total_hopping_loss = 0
        for question, incomplete_sequence, complete_sequence in dataloader:
            
            #Parsing Step
            
            inputs = tokenizer(question, return_tensors = 'pt')
            labels = tokenizer(incomplete_sequence, return_tensors = 'pt')['input_ids']
            
            #Generate HP Embedding and concatenate with input IDs
            pp_input = pp_embeddings.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1)
            concatenated_input = torch.cat([pp_input, inputs['input_ids']], dim=1)
            
            #Adjust attention mask (take all of the soft prompt tokens should be attented)
            pp_attention_mask = torch.ones((inputs['attention_mask'].size(0), pp_length), device=inputs['attention_mask'].device)
            concatenated_attention_mask = torch.cat((pp_attention_mask, inputs['attention_mask']), dim=1)
            
            
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(input_ids=concatenated_input, attention_mask=concatenated_attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_parsing_loss += loss.item()
            
            
            
            #Hopping Step
            incomplete_paths = outputs.logits.argmax(dim=-1)
            
            labels = tokenizer(complete_sequence, return_tensors = 'pt')['input_ids']
            
            hp_input = hp_embeddings.unsqueeze(0).expand(incomplete_paths.size(0), -1, -1)
            concatenated_input_hp = torch.cat([hp_input, incomplete_paths], dim=1)
            
            # Adjust attention mask for HP
            hp_attention_mask = torch.ones((incomplete_paths.size(0), pp_length), device=incomplete_paths.device)
            concatenated_attention_mask_hp = torch.cat((hp_attention_mask, torch.ones_like(incomplete_paths)), dim=1)
            
            optimizer.zero_grad()
            outputs = model(input_ids=concatenated_input_hp, attention_mask=concatenated_attention_mask_hp, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_hopping_loss += loss.item()
                    
        avg_parsing_loss = total_parsing_loss / len(dataloader)
        avg_hopping_loss = total_hopping_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Parsing: {avg_parsing_loss:.4f}, Hopping: {avg_hopping_loss:.4f}")
    
            
            
            
            
            
    
    
    
    
    