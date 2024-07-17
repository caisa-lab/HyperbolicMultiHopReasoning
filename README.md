# Hyperbolic Soft Prompting for Multi-Hop Reasoning

We will implement and combine the methods of these papers:
[Triggering Multi-Hop Reasoning for Question Answering in Language Models using Soft Prompts and Random Walks](https://arxiv.org/pdf/2306.04009)
[Hyperbolic Representations for Prompt Learning](https://aclanthology.org/2024.lrec-main.744.pdf)

## Content

### [notebooks](./notebooks)

Contains mostly notebooks for testing purposes.

### [outputs](./outputs)

Contains the output and error files from the jobs.

### [src](./src)

Contains the actual code implementations which consist of:
1. [datasets](./src/datasets.py) has all necessary datasets that should be created like KnowledgeIntegrationDataset, RandomWalkDataset, OneHopWikiDataset, etc.
2. [train](./src/train.py) has all training methods like KnowledgeIntegration, RandomWalkTraining, ParseThenHop.
3. [eval](./src/eval.py) has evaluation functions which capture EM and F1 scores.

### [tboard_logs](./tboard_logs)

Contains tensorboard log files for all training experiments.

### [load_c4_dataset](./load_c4_dataset.sh)

Loads the C4 dataset. With `--files` you can specify how many files should be downloaded since we don't want the complete dataset.

### [train_knowledge_integration](./train_knowledge_integration.py)

Training script for the Knowledge Integration part. Hyperparameters and model type can be adjusted in the [config](./src/config).

### [train_finetuning_single_hop](./train_finetuning_single_hop.py)

Finetunes the Knowledge Integrated Model on the One Hop Wiki Dataset. Hyperparameters and model type can be adjusted in the [config](./src/config).

### [train_random_walk](./train_random_walk.py)

Training script for the Random Walk part. Hyperparameters and model type can be adjusted in the [config](./src/config).