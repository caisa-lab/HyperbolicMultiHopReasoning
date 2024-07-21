class Config:
    
    class BaseTrainingConfig:
        def __init__(self, optimizer = 'AdaFactor',
                     learning_rate = 0.001,
                     optimizer_param = 1e-5,
                     epochs = 50,
                     scheduler = None,
                     scheduler_factor = 1,
                     log_dir = 'default_log',
                     model_save_path = 'default_model_save'):
            self.optimizer = optimizer #[Adam, AdamW, AdaFactor]
            self.learning_rate = learning_rate #Same as in the paper
            self.optimizer_param = optimizer_param #weight_decay for AdamW Check which Optimizer
            self.epochs = epochs # 100K steps with early Stopping 
            self.scheduler = scheduler  
            self.scheduler_factor = scheduler_factor
            self.log_dir = log_dir
            self.model_save_path = model_save_path
    
    class SingleHopTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(log_dir='tboard_logs/knowledge_integration', model_save_path='checkpoints/knowledge_integration')

                
    class OneHopWikiTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(log_dir='tboard_logs/one_hop_wiki_finetuning', model_save_path='checkpoints/one_hop_wiki_finetuning', epochs=160)

    class RandomWalkTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(learning_rate=0.3, epochs=250, log_dir='tboard_logs/random_walk_training', model_save_path='checkpoints/random_walk_training')
            self.prompt_length = 100
            self.hopping_prompt_checkpoint_path = 'checkpoints/random_walk_training/large_adapt_bsize64_c4_part1/hopping_soft_prompt_epoch_25_val_loss_0.2693.pth'
            self.model_checkpoint_path = 'checkpoints/knowledge_integration/large_adapt_bsize64_c4/model_epoch_16_val_loss_0.0336.pth'

    class ParseThenHopTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(learning_rate=0.3, epochs=250, log_dir='tboard_logs/parse_then_hop_training', model_save_path='checkpoints/parse_then_hop_training')
            self.prompt_length = 100
                
    class T5_Model:
        def __init__(self):
            self.batch_size = 64#128 # From the Paper or 32 if xxl
            self.model_name = "google/t5-large-lm-adapt"            
            self.tokenizer_max_length = 128


            
    def __init__(self):
        self.t5_model = self.T5_Model()
        self.one_hop_wiki_training = self.OneHopWikiTraining()
        self.single_hop_training = self.SingleHopTraining()
        self.random_walk_training = self.RandomWalkTraining()
        self.parse_then_hop_training = self.ParseThenHopTraining()
