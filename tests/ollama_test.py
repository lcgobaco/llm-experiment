#!/usr/bin/env python3
import pandas as pd
import requests
import time


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"
# change to actual data path
DATA_PATH = "/home/dustinmathia/UCD_RSE_stuff/llm-experiment/tests/ollama_llm_output_test_1.csv"




# only load the columns we care about (based on highlighted columns in Excel file)
columns_to_use = [
    "Chromosome", "ChromosomePosition", "Ref", "RefAA", "Alt", "AltAA",
    "Type", "Coverage", "Gene", "GeneStrand", "ExonNumber", "Transcript",
    "Protein", "CodingBase", "CodonPosition", "AAPosition",
    "HGVSGenomic", "HGVSCodingTranscript", "HGVSCoding",
    "HGVSTranslationProtein", "HGVSProtein"
]

# load only the specified columns
df = pd.read_csv(DATA_PATH, usecols=columns_to_use)
df = df.iloc[[0]]

outputs = []

def batch_iteration(df, batch_size): # to yield batches of DataFrame rows
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

def build_prompt(row_dict):
    lines = [f"* {col}: {row_dict.get(col, '')}" for col in columns_to_use] # format each column as "* ColumnName: Value"
    input_data = "\n".join(lines)
    
    return f"""You are an expert system.
    Analyze the following somatic variant data and predict its pathogenicity
    (do not rely on any previous examples or context):

    ### INPUT DATA
    {input_data}
    ### TASK
    Based on this information, please classify the variant as 'Benign', 'Likely Benign', 'Likely Pathogenic',
    'Pathogenic', or 'Unknown Significance'. Answer with only one type."""



for batch in batch_iteration(df, batch_size=5): # loop through DataFrame in batches
    for idx, row in batch.iterrows(): # loop through rows in the batch
        prompt = build_prompt(row.to_dict()) # turn row into dict to build prompt

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=60
        )

        result = response.json()["response"] # extract LLM response from JSON object

        outputs.append({ # store row index and LLM output
            "row_id": idx,
            "llm_output": result.strip()
        })

    time.sleep(0.1)  # polite pacing


output_df = pd.DataFrame(outputs)
output_df.to_csv("ollama_llm_output_test_1.csv", index=False) # write outputs to CSV
# df_with_llm = df.join(output_df.set_index("row_id"), how="left") # merge LLM outputs back to original DataFrame

