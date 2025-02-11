# RoBERTa Finetuning on Covid-QA Dataset

This repository contains code for finetuning RoBERTa models on the Covid-QA dataset for question answering tasks. It supports both full model finetuning and parameter-efficient finetuning (PEFT) using LoRA.

## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Finetuning

To finetune the RoBERTa model, run:

```bash
python scripts/finetune_roberta.py
```

Key configurations in the script:
- Base model: `deepset/roberta-base-squad2`
- Hyperparameters:
  - Learning rate: 2e-6
  - Batch size: 32
  - Epochs: 15
  - Weight decay: 0.0001

The script supports two training modes:
- Full model finetuning
- Parameter-efficient finetuning (PEFT) using LoRA (set `TRAIN_PEFT = True`)

### 2. Generating Predictions

To generate predictions on the validation and test sets:

```bash
python scripts/generate_predictions.py
```

The script will save predictions in JSON format:
- Test predictions: `test_predictions_finetuned.json` or `test_predictions_finetuned_peft.json`
- Validation predictions: `val_predictions_finetuned.json` or `val_predictions_finetuned_peft.json`

### 3. Evaluation

The evaluation script compares model predictions against the gold standard answers. You'll need:

1. **Gold File** (`$gold_file`): 
   - Use `covid-qa-dev.json` for validation set
   - Use `covid-qa-test.json` for test set

2. **Prediction File** (`$pred_file`):
   - JSON file containing a dictionary where:
     - Keys: question IDs
     - Values: predicted answer text
   - Generated using `generate_predictions.py` script

3. **Evaluation Output** (`$eval_file`):
   - JSON file where evaluation metrics will be saved

To run the evaluation:

```bash
python scripts/evaluate.py $gold_file $pred_file --out-file $eval_file
```

Example:
```bash
# For validation set
python scripts/evaluate.py covid-qa/covid-qa-dev.json results/val_predictions_finetuned.json --out-file results/val_metrics.json

# For test set
python scripts/evaluate.py covid-qa/covid-qa-test.json results/test_predictions_finetuned.json --out-file results/test_metrics.json
```

The evaluation script computes:
- Exact Match (EM) score
- F1 score

## Model Outputs

The finetuning process saves:
- Model checkpoints in the specified output directory
- Training logs in the `logs` directory
- Evaluation metrics for both validation and test sets
- Prediction files in JSON format
