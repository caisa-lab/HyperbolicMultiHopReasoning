class Config:
    
    class SingleHopTraining:
        def __init__(self):
            self.optimizer = 'AdaFactor' #[Adam, AdamW, AdaFactor]
            self.learning_rate = 0.001 #Same as in the paper
            self.optimizer_param = 0.01 #weight_decay for AdamW Check which Optimizer
            self.epochs = 100 # 100K steps with early Stopping needs to be implemented
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
            self.epochs = 200 # 200K steps with early Stopping needs to be implemented
            self.scheduler = None
            self.scheduler_factor = 1
            self.log_dir = 'tboard_logs'
            self.model_save_path = 'checkpoints'
            
    class T5_Large_Model:
        def __init__(self):
            self.batch_size = 128#128 # From the Paper
            
            
    class T5_XXL_Model:
        def __init__(self):
            self.batch_size = 32
            
    def __init__(self):
        self.t5_xxl_model = self.T5_XXL_Model()
        self.t5_large_model = self.T5_Large_Model()
        self.single_hop_training = self.SingleHopTraining()
        self.prompt_training = self.PromptTraining()
