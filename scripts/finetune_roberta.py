from functools import partial
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from datatools import preprocess_function, preprocess_covid_format

# set hyperparameters
MODEL_NAME = "deepset/roberta-base-squad2"
MAX_LENGTH = None
PADDING = True
LEARNING_RATE = 2e-6
BATCH_SIZE = 32
EPOCHS = 15
WEIGHT_DECAY = 0.0001
TRAIN_PEFT = True
OUTPUT_DIR = "roberta-base-squad2-covid-qa_no_lr_scheduler"
if TRAIN_PEFT:
    OUTPUT_DIR += "_peft"


model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if TRAIN_PEFT:
    config = LoraConfig(
        task_type="QUESTION_ANS",
        r=8,
        lora_alpha=32,
        lora_dropout=0.01,
    )
    model = get_peft_model(model, config)

dataset = load_dataset(
    "json",
    data_files={
        "train": "covid-qa/covid-qa-train.json",
        "validation": "covid-qa/covid-qa-dev.json",
        "test": "covid-qa/covid-qa-test.json",
    },
)

dataset = preprocess_covid_format(dataset)

dataset = dataset.map(
    partial(
        preprocess_function, tokenizer=tokenizer, max_length=MAX_LENGTH, padding=PADDING
    ),
    batched=True,
)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
)
dataset.cleanup_cache_files()

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

model.print_trainable_parameters()
print(
    "Num Trainable Params: ",
    sum([torch.numel(p.data) for p in model.parameters() if p.requires_grad]),
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    push_to_hub=False,
    logging_dir="logs",
    logging_strategy="epoch",
    save_strategy="best",
    metric_for_best_model="loss",
    save_total_limit=2,
    seed=42,
    evaluation_strategy="epoch",
    lr_scheduler_type="constant",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
)


trainer.train()
