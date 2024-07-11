class Config:
    
    class BaseTrainingConfig:
        def __init__(self, optimizer = 'AdaFactor',
                     learning_rate = 0.001,
                     optimizer_param = 1e-5,
                     epochs = 40,
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
            super().__init__(log_dir='knowledge_integration', model_save_path='knowledge_integration')

                
    class OneHopWikiTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(log_dir='one_hop_wiki_finetuning', model_save_path='one_hop_wiki_finetuning')

    class RandomWalkTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(learning_rate=0.3, epochs=40, log_dir='random_walk_training', model_save_path='random_walk_training')
            self.prompt_length = 100

    class ParseThenHopTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(learning_rate=0.3, epochs=40, log_dir='parse_then_hop_training', model_save_path='parse_then_hop_training')
            self.prompt_length = 100
                
    class T5_Model:
        def __init__(self):
            self.batch_size = 64#128 # From the Paper or 32 if xxl
            self.model_name = "google/t5-v1_1-large"            
            self.tokenizer_max_length = 128


            
    def __init__(self):
        self.t5_model = self.T5_Model()
        self.one_hop_wiki_training = self.OneHopWikiTraining()
        self.single_hop_training = self.SingleHopTraining()
        self.random_walk_training = self.RandomWalkTraining()
        self.parse_then_hop_training = self.ParseThenHopTraining()
