from llmwrapper import LlamaModel
import pandas as pd
from extract_data import results_format
import os 
def preprocess(source):
    for files in sorted(os.listdir(source)):
        path_file=os.path.join(source,files)
        data=results_format(path_file)
        data.extract_metada()
        data.metadata_per_model()
        data.extract_topics_keywords()
        os.makedirs("json",exist_ok=True)
        data.save_json(f"json/{os.path.splitext(files)[0]}.json")
def main():
    llm=LlamaModel("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    llm.request_interpretation("json/huatulco-I_insta.json")
    #llm.get_interpretations()
    #preprocess("resultados")
main()
