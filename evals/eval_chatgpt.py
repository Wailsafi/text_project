import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock
from openai import OpenAI
import os

def ask_chat_gpt_json(chat_input):
    # Add your OpenAI API key here
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = chat_input["messages"]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"}
    )
    answer = completion.choices[0].message.content
    # print(answer)
    return answer


def format_chat(chat,prompt=None):
    chat = chat.split("\n")
    formated_chat = {"messages": []}
    messages = [{"role": "system", "content": prompt}]
    for c in chat:
        if "You :" in c:
            messages.append({"role": "assistant", "content": c.replace("You :", "")})
        else:
            messages.append({"role": "user", "content": c})
    formated_chat["messages"] = messages
    return formated_chat

prompt = """
You are an advanced Named Entity Recognition (NER) system. Your task is to extract entities from the provided text and categorize them into specific types. 

### Entity types: {entity_types}

### Output format:
Provide the result as a list of tuples, where each tuple contains:
1. The entity as a string.
2. The category of the entity.

Example:
Input text: The B-52 pilot, Major Larry G. Messinger, later recalled
Output: 
{{"results" : [[
            [
                "B-52",
                "product"
            ],
            [
                "Larry G. Messinger",
                "person"
            ]
        ]]
}}
Now process the following text:
-Detect the entities and their types.
-Return the result in the specified format.
- where it's a list of tuples, each the entity and its entity type.
- return the result as json as a list of lists, stick to this exact format and do not add any additional information.
"""
data_set = "assets/few-nerd.json"
entity_types = "assets/few-nerd_entity_types.json"
with open(data_set, "r") as file:
    data = json.load(file)
with open(entity_types, "r") as file:
    entity_types = json.load(file)
prompt_formatted = prompt.format(entity_types=entity_types)


def find_word_indices(words, entity):
    """
    Finds the word start and end indices of an entity in a given text.

    Parameters:
        text (str): The full input text.
        entity (str): The entity to find.

    Returns:
        tuple: (word_start_index, word_end_index) of the entity in the text, or (-1, -1) if not found.
    """
    entity_words = entity.split()

    # Loop through the words in the text to find the entity
    for i in range(len(words)):
        if words[i:i + len(entity_words)] == entity_words:
            word_start_index = i
            word_end_index = i + len(entity_words) - 1
            return word_start_index, word_end_index

    # Return (-1, -1) if the entity is not found
    return -1, -1


# Create a global lock
file_lock = Lock()


def process_item(item):
    """
    Process a single item to perform NER and save results in a JSONL file.
    """
    input_text = item["tokenized_text"]
    input_text_unified = " ".join(input_text)

    # Example placeholder functions (replace with actual implementations)
    formated_chat = format_chat(chat=input_text_unified, prompt=prompt_formatted)
    answer = ask_chat_gpt_json(formated_chat)

    # Parse the answer
    answer = json.loads(answer)["results"][0]
    predictions = []

    for entity in answer:
        start, end = find_word_indices(input_text, entity[0])  # Find word indices
        predictions.append([start, end, entity[1]])  # Append as [start, end, entity type]

    # Write results to the file with thread safety
    with file_lock:
        with open("chatgpt_40_mini.jsonl", "a+") as file:
            json.dump({"item": item, "predictions": predictions}, file)
            file.write("\n")


def run_with_threadpool(data, max_workers=2):
    """
    Run the process_item function using a thread pool with progress tracking.

    Parameters:
        data (list): List of items to process.
        max_workers (int): Number of threads in the thread pool.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in data}

        # Track progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Items"):
            try:
                future.result()  # Wait for the task to complete
            except Exception as e:
                print(f"Error processing item: {e}")


# data = data[11497:]
run_with_threadpool(data, max_workers=5)