# Enhancing Multi-Hop Reasoning with Hyperbolic Representations

We used this Paper as our Base Method:
[Triggering Multi-Hop Reasoning for Question Answering in Language Models using Soft Prompts and Random Walks](https://arxiv.org/pdf/2306.04009)

## Content
### [src](./src)

Contains the actual code implementations which consist of:
1. [datasets](./src/datasets/) has all necessary datasets that should be created like KnowledgeIntegrationDataset, RandomWalkDataset etc.
2. [train](./src/train/) consists of 2 training methods. Model Training (for pretraining) and SoftPrompt Training for training the parsing and hopping soft prompt. 
3. [eval](./src/eval.py) has evaluation functions which capture EM and F1 scores.
4. [config](./src/config.py) manages hyperparameters
5. [models](./src/models/) consists of the Hyperbolic T5 and Soft Prompt Model. Soft Prompt Model takes a Knit5 Model and a Soft Prompt.

### [tboard_logs](./tboard_logs)

Contains tensorboard log files for all training experiments.

### [load_c4_dataset](./load_c4_dataset.sh)

Loads the C4 dataset. With `--files` you can specify how many files should be downloaded since we don't want the complete dataset.

### [train_knowledge_integration](./train_knowledge_integration.py)

Training script for the Knowledge Integration part. Hyperparameters and model type can be adjusted in the [config](./src/config).
For example for MetaQA:
```bash
python train_knowledge_integration.py \
    --c4 \
    --dataset metaqa \
    --epochs 50 \
    --checkpoint_save_path checkpoints/metaqa/knowledge_integration/ \
    --tboard_logs_save_path tboard_logs/metaqa/knowledge_integration/ \
    --batch_size 64 \
    --learning_rate 0.001
```

### [train_random_walk](./train_random_walk.py)

Training script for the Random Walk part. Hyperparameters and model type can be adjusted in the [config](./src/config).
For example for MetaQA:
```bash
python train_random_walk.py \
    --additional_layer hyperbolic \
    --learning_rate 0.3 \
    --dataset metaqa \
    --batch_size 64 \
    --epochs 100 \
    --curvature 0.1 \
    --knit5_checkpoint_path Path/To/Knowledge_Integrated/Model/Of/MetaQA \
    --checkpoint_save_path Path/For/Checkpoints \
    --tboard_logs_save_path Path/For/Tensorboard_Logs

```

### [train_random_walk](./train_parse_then_hop.py)

Training script for the Parsing part. Hyperparameters and model type can be adjusted in the [config](./src/config).
For example for MetaQA:
```bash
python train_parse_then_hop.py \
    --additional_layer hyperbolic \
    --learning_rate 0.3 \
    --curvature 0.3728873144946715 \
    --dataset metaqa \
    --epochs 50 \
    --batch_size 64 \
    --knit5_checkpoint_path Path/To/Knowledge_Integrated/Model/Of/MetaQA \
    --checkpoint_save_path Path/For/Checkpoints \
    --tboard_logs_save_path Path/For/Tensorboard_Logs
```

### [test_parse_then_hop](./test_parse_then_hop.py)
Testing Script for Parse Then Hop Method.
```bash
python test_parse_then_hop.py \
    --additional_layer_parse hyperbolic \
    --additional_layer_hop hyperbolic \
    --dataset metaqa \
    --batch_size 64 \
    --knit5_checkpoint_path Path/To/Knowledge_Integrated/Model \
    --parsing_prompt_checkpoint_path Path/To/Parsing_Prompt \
    --hopping_prompt_checkpoint_path Path/To/Hopping_Prompt
```

### [compute_delta_hyperbolicity](./compute_delta_hyperbolicity.py)
Functions to Compute Delta Hyperbolicity and Curvature for each Dataset.

### [compute_distance_accuracy](./compute_distance_accuracy.py)
Functions to Compute the Distance Accuracy for Hyperbolic and Euclidean Embeddings.