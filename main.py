from llmwrapper import LlamaModel
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
def generate_batch(source):
    for models in sorted(os.listdir(source)):
        path_model=os.path.join(source,models)
        llm=LlamaModel(path_model)
        os.makedirs(os.path.join("output", models),exist_ok=True)
        for files in sorted(os.listdir("json")):
            os.makedirs(f"output/{models}/{os.path.splitext(files)[0]}",exist_ok=True)
            dire=f"output/{models}/{os.path.splitext(files)[0]}"
            if os.path.exists(dire) and len(os.listdir(dire))==3:
                print(f"Saltando, ya existe {dire}")
                continue
            path_file=os.path.join("json",files)
            print(f"Procesando {path_file} con {models}")
            llm.request_global_interpretation(path_file,f"output/{models}/{os.path.splitext(files)[0]}")

def main():
    #llm=LlamaModel("models/Aitana-s2b-c0dc7-f16.gguf")
    generate_batch("models")   
#    llm=LlamaModel("models/google_gemma-3-4b-it-Q8_0.gguf")
  #  llm=LlamaModel("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
  #  llm=LlamaModel("models/chatgpt-5-q8_0.gguf")
  #  llm=LlamaModel("models/Claude3.5-4B-Instruct.Q8_0.gguf")
    #llm.request_global_interpretation("json/huatulco-I_insta.json")
    #llm.get_interpretations()
    #preprocess("resultados")
main()
