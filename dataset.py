from datasets import get_dataset_config_names, get_dataset_split_names

# Get available configurations for the IMDb dataset
configs = get_dataset_config_names('imdb')
print("Configurations:", configs)

# Get available splits for the 'plain_text' configuration
splits = get_dataset_split_names('imdb', 'plain_text')
print("Splits for 'plain_text' configuration:", splits)
