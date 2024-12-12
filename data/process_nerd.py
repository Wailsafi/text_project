from datasets import load_dataset
import json

from tqdm import tqdm


def process_huggingface_dataset(dataset_name, split, output_file, entity_file):
    """
    Downloads a dataset from Hugging Face, processes it into a specific format,
    and extracts unique entity types.

    Args:
    - dataset_name (str): The name of the Hugging Face dataset.
    - split (str): The dataset split (e.g., 'train', 'test').
    - output_file (str): Path to save the converted dataset.
    - entity_file (str): Path to save the unique entity types.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name,"supervised" ,split=split)

    converted_data = []
    unique_entities = set()

    for entry in tqdm(dataset):
        words = entry["tokens"]
        ner_tags = entry["ner_tags"]
        entities = []

        current_entity = None
        for idx, tag in enumerate(ner_tags):
            if tag != 0:  # Non-zero tags indicate an entity
                entity_type = dataset.features["ner_tags"].feature.names[tag]  # Map tag ID to entity name
                unique_entities.add(entity_type)
                if current_entity is None:
                    current_entity = {"start": idx, "end": idx, "type": entity_type}
                else:
                    current_entity["end"] = idx  # Extend the entity span
            else:
                if current_entity:
                    # Save the completed entity
                    entities.append([current_entity["start"], current_entity["end"], current_entity["type"]])
                    current_entity = None

        # If there's an entity being processed at the end of the loop, add it
        if current_entity:
            entities.append([current_entity["start"], current_entity["end"], current_entity["type"]])

        # Prepare the formatted output
        formatted_entry = {
            "tokenized_text": words,
            "ner": entities
        }
        converted_data.append(formatted_entry)

    # Save the converted dataset
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=4)

    # Save the unique entity types
    with open(entity_file, "w") as f:
        json.dump(list(unique_entities), f, indent=4)

# Example usage
dataset_name = "DFKI-SLT/few-nerd"  # Replace with your Hugging Face dataset name
split = "test"  # Replace with the desired split (e.g., "train", "test", "validation")
output_file = "few-nerd.json"  # Replace with your desired output file path
entity_file = "few-nerd_entity_types.json"  # File to store detected unique entity types

process_huggingface_dataset("DFKI-SLT/few-nerd", split, output_file, entity_file)
# dataset = load_dataset("DFKI-SLT/few-nerd","supervised")
#
# convert_ner_dataset(dataset["train"],"few-nerd.json")