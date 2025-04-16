from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import pandas as pd
import os
from models import columns,vectorize_data
from resultsAnalyse import drawConfusionMatrix
from datasets import load_dataset
def evaluateMultiLabelClassifier( dataset_type):
    train_df = pd.read_csv(f"data/{dataset_type}/train_clean.csv")
    val_df = pd.read_csv(f"data/{dataset_type}/validation_clean.csv")
    test_df = pd.read_csv(f"data/{dataset_type}/test_clean.csv")
    #predictions = model.predict(test_x)
    #os.makedirs(f"plots/ConfusionMatrix/Test",exist_ok=True)
    #drawConfusionMatrix(predictions,test_df[columns],model.__class__.__name__)
    ds = load_dataset("higopires/RePro-categories-multilabel")
    print(ds.features)
    #run_blitr(train_df,val_df,test_df)
    run_blitr(ds)


def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int) 
    return metric.compute(predictions=predictions, references=labels,average="micro")

def preprocess_function(sample):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize text
    tokenized = tokenizer(sample["review_text"], truncation=True, padding=True)
    
    # Convert binary label columns to a list (e.g., [1, 0, 1, 0, 0, 0])
    labels = [
        sample["ENTREGA"],
        sample["OUTROS"],
        sample["PRODUTO"],
        sample["CONDICOESDERECEBIMENTO"],
        sample["INADEQUADA"],
        sample["ANUNCIO"]
    ]
    
    tokenized["labels"] = labels
    return tokenized

def run_blitr(raw_ds):
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6,problem_type="multi_label_classification")
    
    tokenized_dataset = raw_ds.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./resultsNew",
        learning_rate=1.5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    #trainer.evaluate()


if __name__ == "__main__":
    evaluateMultiLabelClassifier( "datasets_removeNulls")