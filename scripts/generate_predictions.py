from functools import partial
from peft import PeftModel
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from datatools import preprocess_covid_format, preprocess_function
import json

MODEL_NAME = (
    "deepset/roberta-base-squad2"
    # "roberta-base-squad2-covid-qa_no_lr_scheduler/checkpoint-270"
)

TEST_PREDICTIONS_FILE = "test_predictions_finetuned.json"
VAL_PREDICTIONS_FILE = "val_predictions_finetuned.json"
PEFT_MODEL_NAME = "roberta-base-squad2-covid-qa_no_lr_scheduler_peft/checkpoint-675"


if PEFT_MODEL_NAME is not None:
    TEST_PREDICTIONS_FILE = "test_predictions_finetuned_peft.json"
    VAL_PREDICTIONS_FILE = "val_predictions_finetuned_peft.json"
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(model, PEFT_MODEL_NAME)
    model.eval()

else:
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_dataset(
    "json",
    data_files={
        "train": "covid-qa/covid-qa-train.json",
        "validation": "covid-qa/covid-qa-dev.json",
        "test": "covid-qa/covid-qa-test.json",
    },
)

dataset = preprocess_covid_format(dataset)

test_ids = dataset["test"]["id"]
val_ids = dataset["validation"]["id"]

dataset.set_format(columns=["context", "question"])

dataset.cleanup_cache_files()

nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

predictions = {}

# Test predictions
for pid, out in zip(
    dataset["test"]["id"], nlp(dataset["test"], batch_size=8, truncation="only_first")
):
    predictions.update({pid: out["answer"]})


with open(TEST_PREDICTIONS_FILE, "w") as f:
    json.dump(predictions, f)

# Validation predictions
for pid, out in zip(
    dataset["validation"]["id"],
    nlp(dataset["validation"], batch_size=8, truncation="only_first"),
):
    predictions.update({pid: out["answer"]})

with open(VAL_PREDICTIONS_FILE, "w") as f:
    json.dump(predictions, f)
