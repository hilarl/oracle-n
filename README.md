# Oracle-n Model

Oracle-n is an AI model based on the BERT architecture, designed for text and sentiment analysis. Using the foundational strengths of BERT, Oracle-n is designed to to optimize performance for specific needs. This repository includes the model code, tokenizer, and training scripts.

## Features

- **Customized BERT Configuration**: Tailored configurations to fit text and sentiment analysis tasks.
- **Oracle-n Tokenizer**: Custom tokenizer designed for preprocessing text data efficiently.
- **Sentiment Analysis**: Trained on the IMDb dataset to perform sentiment analysis tasks.

## Directory Structure

- `aclImdb/`: Directory containing the IMDb dataset files in Parquet format.
- `dataset.py`: Script for handling the dataset loading and preprocessing.
- `logs/`: Directory for TensorBoard logs during training.
- `oracle-n-model/`: Directory containing the saved model.
- `oracle-n-tokenizer/`: Directory containing the saved tokenizer.
- `scripts/`: Directory containing additional scripts for training and evaluation.
- `.gitignore`: Git ignore file to exclude unnecessary files from the repository.
- `requirements.txt`: File listing the dependencies required for the project.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hilarl/oracle-n.git
   cd oracle-n
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Ensure you have the IMDb dataset files in the `aclImdb/` directory. If needed, you can download them from the IMDb dataset page and convert them to Parquet format.

## Usage

### Training the Model

1. **Prepare the Dataset**:
   Ensure the dataset files are in the `aclImdb/` directory in Parquet format.

2. **Run the Training Script**:
   ```bash
   python scripts/train_model.py
   ```

3. **Monitor Training with TensorBoard**:
   ```bash
   tensorboard --logdir logs
   ```

### Evaluating the Model

1. **Run the Evaluation Script**:
   ```bash
   python scripts/evaluate_model.py
   ```

## Customizing the Model

1. **Modify the Configuration**:
   Edit the `dataset.py` script to change the model configuration parameters such as hidden size, number of layers, and attention heads.

2. **Add Your Own Tokenizer**:
   Customize the tokenizer by editing the `oracle-n-tokenizer` directory.

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **BERT**: This model is based on the BERT architecture developed by Google.
- **Hugging Face**: Leveraging the Hugging Face Transformers library for model development.

## Contact

For any questions or suggestions, please open an issue on GitHub or contact us at hilal@tenzro.com.

---

This README provides an overview of the Oracle-n model, its features, and how to set up and use it. Feel free to customize it further based on your specific requirements and details.