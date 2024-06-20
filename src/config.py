class Config:
    
    class Training:
        def __init__(self):
            self.optimizer = 'AdamW' #[Adam, AdamW]
            self.learning_rate = 3e-5
            self.optimizer_param = 0.01 #weight_decay for AdamW
            self.scheduler = None
            self.scheduler_factor = 1
            self.log_dir = 'tboard_logs'
            self.model_save_path = 'checkpoints'
    def __init__(self):
        
        self.training = self.Training()