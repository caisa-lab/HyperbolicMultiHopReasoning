# Hyperbolic Soft Prompting for Multi-Hop Reasoning

We will implement and combine the methods of these papers:
[Triggering Multi-Hop Reasoning for Question Answering in Language Models using Soft Prompts and Random Walks](https://arxiv.org/pdf/2306.04009)
[Hyperbolic Representations for Prompt Learning](https://aclanthology.org/2024.lrec-main.744.pdf)

## Dataset

You need to download the Dataset from here: [2WikiMultiHopQADataset](https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1).
You need to have a directory dataset/2wikimultihop/ where the dev.json, test.json, train.json lies.

## Content

### [notebooks](./notebooks)

Contains mostly notebooks for testing purposes.

### [outputs](./outputs)

Contains the output and error files from the jobs.

### [src](./src)

Contains the actual code implementations which consist of:
1. [datasets](./src/datasets/) has all necessary datasets that should be created like KnowledgeIntegrationDataset, RandomWalkDataset, OneHopWikiDataset, etc.
2. [train](./src/train/) consists of 2 training methods. ModelTraining (for pretraining) and SoftPromptTraining for training the parsing and hopping soft prompt. 
3. [eval](./src/eval.py) has evaluation functions which capture EM and F1 scores.
4. [config](./src/config.py) manages hyperparameters
5. [models](./src/models/) consists of the Hyperbolic T5 and Soft Prompt Model. Soft Prompt Model takes a Knit5 Model and a Soft Prompt.

### [tboard_logs](./tboard_logs)

Contains tensorboard log files for all training experiments.

### [load_c4_dataset](./load_c4_dataset.sh)

Loads the C4 dataset. With `--files` you can specify how many files should be downloaded since we don't want the complete dataset.

### [train_knowledge_integration](./train_knowledge_integration.py)

Training script for the Knowledge Integration part. You can do `--c4` to train with pretraining objective on the c4 dataset and you can do `--hyperbolic` to train the hyperbolic t5 model. Hyperparameters and model type can be adjusted in the [config](./src/config).

### [train_finetuning_single_hop](./train_finetuning_single_hop.py)

Finetunes the Knowledge Integrated Model on the One Hop Wiki Dataset. Hyperparameters and model type can be adjusted in the [config](./src/config).

### [train_random_walk](./train_random_walk.py)

Training script for the Random Walk part. You can do `--hyperbolic` to train the HyperbolicSoftPromptModel. Hyperparameters and model type can be adjusted in the [config](./src/config).