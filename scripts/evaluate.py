from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the IMDb dataset
dataset = load_dataset('imdb')
test_dataset = dataset['test']

# Load the Oracle-n tokenizer and model
tokenizer = BertTokenizer.from_pretrained('../oracle-n-tokenizer')
model = TFBertForSequenceClassification.from_pretrained('../oracle-n-model')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize the test dataset
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert test dataset to TensorFlow format
test_dataset = test_dataset.to_tf_dataset(columns=['input_ids', 'attention_mask', 'label'],
                                           shuffle=False, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.2f}")

# Get predictions
predictions = model.predict(test_dataset)
pred_labels = tf.argmax(predictions.logits, axis=1).numpy()
true_labels = [label for label in test_dataset['label']]

# Calculate metrics
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("Evaluation complete.")
