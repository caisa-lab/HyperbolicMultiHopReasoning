# Enhancing Multi-Hop Reasoning with Hyperbolic Representations

We used this Paper as our Base Method:
[Triggering Multi-Hop Reasoning for Question Answering in Language Models using Soft Prompts and Random Walks](https://arxiv.org/pdf/2306.04009)

## Datasets:
All datasets live under the `dataset/` directory.  
### 2WikiMultiHopQA
1. **Download**:
Download the 2WikiMultiHopQA Dataset from here: [2WikiMultiHopQADataset](https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1).
2. **Extract**:
Make sure you have:
- `dataset/2wikimultihop/train.json`
- `dataset/2wikimultihop/dev.json`
- `dataset/2wikimultihop/test.json`

### MetaQA
1. **Download**:
Download the MetaQA Dataset from here: [MetaQA](https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g). You only need the following files:
- `dataset/metaqa/2-hop/vanilla/qa_train.txt, qa_dev.txt & qa_test.txt`
- `dataset/metaqa/2-hop/qa_train_qtype.txt, qa_dev_qtype.txt & qa_test_qtype.txt`
- `dataset/metaqa/kb.txt`
2. **Preprocess**:
Use the metaqa_preprocess.py file to preprocess the MetaQA dataset. You can adjust the input and output paths in the python file at the bottom.
1. **Extract**:
Make sure you have the following files after preprocessing:
- `dataset/metaqa/vanilla/metaqa_train_evidences.json`
- `dataset/metaqa/vanilla/metaqa_dev_evidences.json`
- `dataset/metaqa/vanilla/metaqa_test_evidences.json`

### MLPQ
1. **Download**:
Download the MLPQ Dataset from here: [MLPQ](https://github.com/seu-kse/MLPQ/tree/master)
2. **Preprocess**
Use the mlpq_preprocess.py file to preprocess the MLPQ dataset. For example like this: \
    ```bash 
    python mlpq_preprocess.py --en_fr dataset/mlpq/Questions/fr-en/2-hop/r2r_en_fr_question_en --fr_en dataset/mlpq/Questions/fr-en/2-hop/r2r_fr_en_question_en --out_dir ./mlpq_json
    ```
3. **Extract**
Make sure you have the following files after preprocessing:
- `dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json`
- `dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json`
- `dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json`


### PQ
1. **Download**: Download the PQ Dataset from here [PathQuestion](https://github.com/zmtkeke/IRN).
2. **Extract**: Make sure you have
- `dataset/pathquestion/PQ-2H.txt`




## Content
### [src](./src)

Contains the actual code implementations which consist of:
1. [datasets](./src/datasets/) has all necessary datasets that should be created like KnowledgeIntegrationDataset, RandomWalkDataset etc.
2. [train](./src/train/) consists of 2 training methods. Model Training (for pretraining) and SoftPrompt Training for training the parsing and hopping soft prompt. 
3. [eval](./src/eval.py) has evaluation functions which capture EM and F1 scores.
4. [config](./src/config.py) manages hyperparameters
5. [models](./src/models/) consists of the Hyperbolic T5 and Soft Prompt Model. Soft Prompt Model takes a Knit5 Model and a Soft Prompt.

### [load_c4_dataset](./load_c4_dataset.sh)

Loads the C4 dataset. With `--files` you can specify how many files should be downloaded since we don't want the complete dataset.

### [train_knowledge_integration](./train_knowledge_integration.py)

Training script for the Knowledge Integration part. Hyperparameters and model type can be adjusted in the [config](./src/config).
For example for MetaQA:
```bash
python train_knowledge_integration.py \
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

### [train_parse_then_hop](./train_parse_then_hop.py)

Training script for the Parsing part. Hyperparameters and model type can be adjusted in the [config](./src/config).
For example for MetaQA:
```bash
python train_parse_then_hop.py \
    --additional_layer hyperbolic \
    --learning_rate 0.3 \
    --curvature 0.37 \
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


## How to get started
- Install the c4 dataset by using the ./load_c4_dataset.sh --files 5
- Download and Preprocess the Dataset as described above
- Use the train_knowledge_integration.py script to finetune your model and save the Knowledge Integrated Model
- Finetune the Soft Prompts (and optionally the additional layer) with the train_random_walk.py and train_parse_then_hop.py 
- You can adjust parameters and save path in the config. Some hyperparameters you can also adjust in the arguments of the python scripts like the batch size, number of epochs, learning rate, which additional layer to choose and the initial curvature.