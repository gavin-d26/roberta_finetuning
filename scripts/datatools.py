import pandas as pd


def _covert_to_squad_format(row):
    row = row["data"]["paragraphs"][0]
    context = row["context"]
    questions = row["qas"]

    question_texts = []
    answers = []
    qids = []
    for question in questions:
        question_text = question["question"]
        qid = question["id"]
        answer_text = [question["answers"][0]["text"]]
        answer_start = [question["answers"][0]["answer_start"]]
        answer = {"text": answer_text, "answer_start": answer_start}
        # answer_end = answer_start + len(answer_text)

        question_texts.append(question_text)
        answers.append(answer)
        qids.append(qid)
    # create a dataframe
    df = pd.DataFrame(
        {"id": qids, "question": question_texts, "answers": answers, "context": context}
    )
    return df


def preprocess_covid_format(dataset):
    import pandas as pd
    from datasets import Dataset, DatasetDict

    train_dataset = dataset["train"].to_pandas().apply(_covert_to_squad_format, axis=1)
    val_dataset = (
        dataset["validation"].to_pandas().apply(_covert_to_squad_format, axis=1)
    )
    test_dataset = dataset["test"].to_pandas().apply(_covert_to_squad_format, axis=1)

    train_dataset = pd.concat(train_dataset.to_list(), axis=0)
    val_dataset = pd.concat(val_dataset.to_list(), axis=0)
    test_dataset = pd.concat(test_dataset.to_list(), axis=0)

    train_dataset = Dataset.from_pandas(
        train_dataset[["id", "question", "answers", "context"]]
    )
    val_dataset = Dataset.from_pandas(
        val_dataset[["id", "question", "answers", "context"]]
    )
    test_dataset = Dataset.from_pandas(
        test_dataset[["id", "question", "answers", "context"]]
    )

    dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    return dataset


def preprocess_function(examples, tokenizer, max_length=384, padding="max_length"):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding=padding,
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
