from sklearn.metrics import classification_report

# Load different models for comparison (XLM-Roberta, mBERT, etc.)
model_1 = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base")
model_2 = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased")

trainer_1 = Trainer(model=model_1)
trainer_2 = Trainer(model=model_2)

predictions_1 = trainer_1.predict(validation_dataset)
predictions_2 = trainer_2.predict(validation_dataset)

print(classification_report(validation_labels, predictions_1.predictions.argmax(-1)))
print(classification_report(validation_labels, predictions_2.predictions.argmax(-1)))
