from llama_cpp import Llama
import re
import sys, os
from contextlib import contextmanager
import json
@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
class LlamaModel:
    """Clase que posee la iteracción con un modelo de LLM. Este modelo puede ser llama u otro tipo de modelo de GGUF
    model_name=nombre del modelo
    model=instancia de llama que inicia el modelo
    freq: las palabras que se usarán para dar un inferencia
    generetaed_titles : titutlos generados 
    generated_interpretations: Interpretaciones generadas 
    """
    def __init__(self,model:str):
        """Método constructor que inicializa todos los atributos
        Args:
            model (str): Nombre del modelo
        """
        self.model_name=model
        with suppress_stderr():
            self.model=Llama(model_path=model,
                                n_ctx=4096,          
                                n_gpu_layers=-1,  
                                n_threads=8,
                                verbose=False)
        self.freq = None
        self.generated_interpretations = None
    def build_prompt(self,topic_id,keywords,documents):
        keywords_str=", ".join(keywords)
        docs_str="\n".join([f"{i+1}. {d}" for i,d in enumerate(documents)])
        return f"""
            You are a data analyst specialized in topic modeling evaluation.
            Your task is to analyze the output of a topic modeling process based strictly and exclusively on the information provided.
    You will receive:
    1. A list of keywords extracted for a topic.
    2. A set of representative tourist reviews associated with that topic.

    IMPORTANT RULES:

    1. You must base your analysis ONLY on the keywords and documents provided.
    2. You must NOT assume, infer, or introduce any geographical locations, destinations, cities, landmarks, or contextual information that are not explicitly mentioned in the provided text.
    3. If the documents mention a location explicitly, you may refer to it. Otherwise, do not introduce any location.
    4. If the keywords and documents do not appear semantically coherent or do not share a clear common theme, you must classify the topic as: "INCOHERENT – DISCARD".
    5. Do not speculate.
    6. Do not add external knowledge.
    7. Be concise, analytical, and objective.

    For each topic, provide:

    - Topic ID:
    - Decision: (VALID / INCOHERENT – DISCARD)
    - Relevance Score (1–5): a numeric score indicating topic quality:
        1 = completely incoherent or noise
        2 = weak / mostly incoherent
        3 = partially coherent but noisy
        4 = coherent topic with minor noise
        5 = highly coherent and well-defined topic
    - Proposed Topic Label (max 6 words, neutral and descriptive):
    - Short Analytical Description (2–3 sentences strictly grounded in the provided text)

    After analyzing all topics:

    - Identify if any VALID topics are semantically overlapping.
    - If so, propose a merged topic label and indicate which Topic IDs should be merged.
    - Do not propose to merge topics unless there is strong semantic overlap explicitly visible in the data.
    - If you are uncertain, propose to DISCARD rather than inventing a thematic interpretation.

    Output must follow this structure exactly.
    {{
    "topic_id": integer,
    "decision": "VALID" | "INCOHERENT – DISCARD",
    "score": integer (1-5),
    "label": string,
    "description": string
    }}

    below are the input information from the topic modeling process:

    Topic ID: {topic_id}

        Keywords:
        {keywords_str}

        Documents:
        {docs_str}
    """
    def request_interpretation(self,input_data,params=None):
        if params is None:
            params = {
                "max_tokens": 400,
                "temperature": 0,
                "top_p": 0.1,
                "repeat_penalty": 1.2
            }
        data=read_json(input_data)
        data_models=data["models"]
        for model in data_models:
            modelo=model["model"]
            info=model["topics"]
            for topico in info:
                
                prompt=self.build_prompt(topico["topic_id"],topico["keywords"],topico["pertenece"])
                output=self.model(prompt=prompt,**params)
                print(output)
                break
            break
    def get_interpretations(self)->list[str]:
        """Obtiene las interpretaciones almacenadas

        Returns:
            list[str]: conjunto de strings con las interpretaciones por tópico
        """
        return self.generated_interpretations
def read_json(src):
    with open(src,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data
