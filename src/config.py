class Config:
    
    class SingleHopTraining:
        def __init__(self):
            self.optimizer = 'AdaFactor' #[Adam, AdamW, AdaFactor]
            self.learning_rate = 0.001 #Same as in the paper
            self.optimizer_param = 1e-5 #weight_decay for AdamW Check which Optimizer
            self.epochs = 40 # 100K steps with early Stopping 
            self.scheduler = None
            self.scheduler_factor = 1
            self.log_dir = 'tboard_logs'
            self.model_save_path = 'checkpoints'
    
    class PromptTraining:
        def __init__(self):
            self.optimizer = 'AdaFactor' #[Adam, AdamW, AdaFactor]
            self.optimizer_param = 0.01 #weight_decay for AdamW Check which Optimizer
            self.learning_rate = 0.3
            self.prompt_length = 100
            self.epochs = 200 # 200K steps with early 
            self.scheduler = None
            self.scheduler_factor = 1
            self.log_dir = 'tboard_logs'
            self.model_save_path = 'checkpoints'
            
    class T5_Model:
        def __init__(self):
            self.batch_size = 64#128 # From the Paper or 32 if xxl
            self.model_name = "google/t5-v1_1-large"            
            self.tokenizer_max_length = 128


            
    def __init__(self):
        self.t5_model = self.T5_Model()
        self.single_hop_training = self.SingleHopTraining()
        self.prompt_training = self.PromptTraining()
