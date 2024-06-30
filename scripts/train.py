import os
from datasets import load_dataset
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import tensorflow as tf

# Load the IMDb dataset from Parquet files
train_dataset = load_dataset('parquet', data_files={'train': './aclImdb/train-00000-of-00001.parquet'}, split='train')
print(f"Train dataset loaded: {len(train_dataset)} examples")
test_dataset = load_dataset('parquet', data_files={'test': './aclImdb/test-00000-of-00001.parquet'}, split='test')

# Print a sample of the raw dataset
print("Raw dataset sample:")
print(train_dataset[0])

# Create a new tokenizer named Oracle-n
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('../oracle-n-tokenizer')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert datasets to TensorFlow format
def format_dataset(dataset):
    def to_tf_data(examples):
        return {
            'input_ids': examples['input_ids'],
            'attention_mask': examples['attention_mask'],
            'label': examples['label']  # Keep it as 'label' here
        }
    
    dataset = dataset.map(to_tf_data, batched=True)
    return dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],  # Use 'label' here
        shuffle=True, 
        batch_size=32
    )

train_dataset = format_dataset(train_dataset)
test_dataset = format_dataset(test_dataset)

# Print a sample of the formatted dataset
print("\nFormatted dataset sample:")
for batch in train_dataset.take(1):
    print("Input IDs shape:", batch[0]['input_ids'].shape)
    print("Attention Mask shape:", batch[0]['attention_mask'].shape)
    print("Labels shape:", batch[1].shape)
    print("Sample labels:", batch[1])

# Define a smaller Oracle-n model configuration
config = BertConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    num_labels=2  # Specify the number of labels
)

# Initialize the model with the custom configuration
model = TFBertForSequenceClassification(config)

# Compile the model
from tensorflow.keras.optimizers.legacy import Adam
model.compile(optimizer=Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define TensorBoard callback
log_dir = "../logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Train the model
try:
    history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=[tensorboard_callback])
except Exception as e:
    print(f"An error occurred during training: {e}")
    # Print the first batch of the dataset
    for batch in train_dataset.take(1):
        print("Input shape:", {k: v.shape for k, v in batch[0].items()})
        print("Label shape:", batch[1].shape)
        print("Sample inputs:", {k: v[:1] for k, v in batch[0].items()})
        print("Sample labels:", batch[1][:5])
else:
    # Save the trained model
    model.save_pretrained('../oracle-n-model')
    print("Training complete and model saved.")