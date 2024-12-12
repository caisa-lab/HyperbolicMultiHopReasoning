from src.utils.util import load_dataset, load_musique_dataset
from src.train import *
from src.datasets import ParseDataset
from src.datasets.musique import ParseMusiqueDataset
import pandas as pd
from src.knowledge_graph import create_knowledge_graph_wikimultihop
from src.config import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from src.models import *
import argparse

  

def _train_parse_then_hop(hyperbolic: bool, dataset, lr):
    if dataset == 'musique':
        train_dataset, dev_dataset = load_musique_dataset('dataset/musique', replace_answer_with_gt=False)

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")

        parse_train = ParseMusiqueDataset(train_dataset)
        parse_dev = ParseMusiqueDataset(dev_dataset)

    elif dataset in ['2wikimultihop', 'wikimultihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")

        parse_train = ParseDataset(train_dataset)
        parse_dev = ParseDataset(dev_dataset)

    print(f"Number of Parse Questions Train: {len(parse_train)}")
    print(f"Number of Parse Questions Dev: {len(parse_dev)}")

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    #google/t5-large-lm-adapt
    parse_then_hop_dataloader_train = DataLoader(parse_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.parse_then_hop_training.num_workers)
    parse_then_hop_dataloader_dev = DataLoader(parse_dev,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.parse_then_hop_training.num_workers)

    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if config.parse_then_hop_training.hopping_prompt_checkpoint_path is not None:
        soft_prompt = nn.Parameter(torch.randn(100, 1024))
        soft_prompt = load_soft_prompt(soft_prompt, config.parse_then_hop_training.hopping_prompt_checkpoint_path)
    else:
        soft_prompt = None
    
    print("Loading Model...")
    if hyperbolic:
        hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type='hyperbolic', curvature=config.parse_then_hop_training.curvature, checkpoint_hyperbolic_knit5=config.parse_then_hop_training.model_checkpoint_path, with_model_state_dict=True)
        model = SoftPromptModel(curvature=config.parse_then_hop_training.curvature, knit5=hyperbolic_knit5_model, knit5_checkpoint_path=config.parse_then_hop_training.model_checkpoint_path, model_name='hyperbolic_hopping_prompt', soft_prompt=soft_prompt, with_model_state_dict=True)
        print(f"Train with hyperbolic Soft Prompt Model with curvature {config.parse_then_hop_training.curvature} and Exponential Mapping at encoder layer {config.t5_model.map_encoder_layers} and at decoder layer {config.t5_model.map_decoder_layers}")
    else:
        euclidean_knit5_model = T5ModelWithAdditionalLayer(layer_type='euclidean', checkpoint_hyperbolic_knit5=config.parse_then_hop_training.model_checkpoint_path, with_model_state_dict=True)
        model = SoftPromptModel(curvature=config.parse_then_hop_training.curvature, knit5=euclidean_knit5_model, knit5_checkpoint_path=config.parse_then_hop_training.model_checkpoint_path, model_name='hyperbolic_hopping_prompt', soft_prompt=soft_prompt, with_model_state_dict=True)

    print("Model type before passing to SoftPromptTrainer:", type(model))

    config.parse_then_hop_training.learning_rate = lr
    trainer = SoftPromptTrainer(model,
                      tokenizer,
                      parse_then_hop_dataloader_train,
                      parse_then_hop_dataloader_dev,
                      config,
                      device=device,
                      method='parse_then_hop_training',
                      checkpoint_path=config.parse_then_hop_training.hopping_prompt_checkpoint_path,
                      tboard_checkpoint_path=config.parse_then_hop_training.tboard_checkpoint_path,
                      retrain = True
                      )




    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')

    print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.parse_then_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.parse_then_hop_training.optimizer}')   

    trainer.train(epochs=config.parse_then_hop_training.epochs)
    
    
if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Parse Then Hop Training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    parser.add_argument('--lr', type=float, default=0.001, help='Specify the learning rate (e.g., 0.001)')
    parser.add_argument('remainder', type=str, nargs='?', default=None, help='Specify the remainder (e.g., musique, 2wikimultihop)')
    args = parser.parse_args()
    

    _train_parse_then_hop(hyperbolic = args.hyperbolic, dataset=args.remainder, lr = args.lr)
