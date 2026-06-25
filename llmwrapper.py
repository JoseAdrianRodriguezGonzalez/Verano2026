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
                                n_ctx=16981,          
                                n_gpu_layers=-1,  
                                n_threads=8,
                                verbose=False)
        
        self.generated_interpretations = []
    def build_prompt(self,topic_id,keywords,documents):
        keywords_str=", ".join(keywords)
        docs_str="\n".join([f"{i+1}. {d}" for i,d in enumerate(documents)])
        return f"""
            You are a strict JSON generator for topic modeling evaluation in TOURISM DATA ANALYSIS.

            You are a data analyst specialized in tourism reviews, travel experiences, and visual tourism content.

            You will analyze ONLY ONE topic at a time.

            CRITICAL RULES:
            - Output ONLY a valid JSON object.
            - Do NOT output explanations.
            - Do NOT output markdown.
            - Do NOT output multiple topics.
            - Do NOT include any text before or after the JSON.
            - Do NOT perform merging or cross-topic analysis.
            - Ignore any instruction about multiple topics or global analysis.

            VERY IMPORTANT SEMANTIC RULE:
            - Your evaluation MUST be based ONLY on DIRECT TOURISM-RELATED COHERENCE.
            - The topic must reflect clear, literal relationships in tourism contexts such as:
            - places, landscapes, cities, attractions
            - tourist activities (photography, walking, sightseeing, festivals, beaches, museums)
            - travel-related visual descriptions

            DO NOT consider as valid coherence:
            - poetic interpretations
            - metaphors or symbolic meaning
            - artistic or abstract associations
            - literary, philosophical, or metaphorical connections
            - any non-literal or indirect semantic link

            If the topic only makes sense through metaphor, symbolism, or poetic interpretation, it MUST be classified as INCOHERENT – DISCARD.

            TASK:
            Evaluate whether the given topic is coherent as a REAL-WORLD TOURISM TOPIC based only on the provided keywords and documents.

            You must decide if the topic is coherent or not in a tourism context.

            Return ONLY this JSON schema:

            {{
            "topic_id": {topic_id},
            "decision": "VALID" | "INCOHERENT – DISCARD",
            "score": 1,
            "label": "short tourism-related label (max 6 words)",
            "description": "2–3 sentences strictly grounded in tourism content"
            }}

            SCORING (tourism coherence only):
            1 = no tourism meaning, pure noise or random words
            2 = weak or unclear tourism relation
            3 = partially related to tourism but noisy or inconsistent
            4 = clear tourism topic with minor noise
            5 = highly coherent and strongly tourism-focused topic

            INPUT:

            Topic ID: {topic_id}

            Keywords:
            {keywords_str}

            Documents:
            {docs_str}

            OUTPUT (JSON ONLY):
            """
    
    def format_llm_input_string(self,cluster_keywords_dict, representative_docs_dict):
        strin = ""
        # Get unique sorted cluster IDs that are present in both dictionaries
        cluster_ids = sorted(list(set(cluster_keywords_dict.keys()) & set(representative_docs_dict.keys())))

        for topic_id in cluster_ids:
            # Convert np.int64 to standard int for f-string formatting if necessary
            strin += f"Topic ID: {int(topic_id)}:\n"

            # Get keywords for this topic_id
            # cluster_keywords_dict[topic_id] is a list of tuples: [('keyword', score), ...]
            keywords_list = [item[0] for item in cluster_keywords_dict[topic_id]]
            strin += f"Keywords:\n{', '.join(keywords_list)}\n"

            strin += f"Documents:\n"
            # representative_docs_dict[topic_id] is a list of tuples: [('document_text', score), ...]
            for i, doc_text in enumerate(representative_docs_dict[topic_id]):
                strin += f"{i+1}: {doc_text}\n"
            strin += "\n" # Add an extra newline for better separation

        return strin
    def build_global_prompt(self, formatted_topics_str):
        return f"""You are a data analyst specialized in topic modeling evaluation.

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
        - Proposed Topic Label (max 6 words, neutral and descriptive):
        - Short Analytical Description (2–3 sentences strictly grounded in the provided text)

        After analyzing all topics:

        - Identify if any VALID topics are semantically overlapping.
        - If so, propose a merged topic label and indicate which Topic IDs should be merged.
        - Do not propose to merge topics unless there is strong semantic overlap explicitly visible in the data.
        - If you are uncertain, propose to DISCARD rather than inventing a thematic interpretation.

        Output must follow this structure exactly.

        below are the input information from the topic modeling process:

        {formatted_topics_str}
        """

    def extract_json(self, text):
        text = text.strip()
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
        return None
    def request_interpretation(self,input_data,params=None):
        if params is None:
            params = {
                "max_tokens": 1000,
                "temperature": 0.8,
                "top_p": 0.9,
                "repeat_penalty": 1.2,

            }
        data=read_json(input_data)
        results=[]
        data_models=data["models"]
        for model in data_models:
            modelo=model["model"]
            info=model["topics"]
            for topico in info:
                
                prompt=self.build_prompt(topico["topic_id"],topico["keywords"],topico["pertenece"])
                output=self.model(prompt=prompt,**params)
                raw = output["choices"][0]["text"]
                parsed = self.extract_json(raw)
                results.append(parsed)
                self.generated_interpretations.append(parsed)
        return results
    def request_global_interpretation(self,input_data,output_data,params=None):
        if params is None:
            params = {
                "max_tokens": 500,
                "temperature": 0.4,
                "top_p": 0.9,
                "repeat_penalty": 1.2,

            }
        data=read_json(input_data)
        results=[]
        data_models=data["models"]
        for model in data_models:
            modelo=model["model"]
            info=model["topics"]
            cluster_keywords_dict = {topico["topic_id"]: topico["keywords"] for topico in info}
            representative_docs_dict = {topico["topic_id"]: topico["pertenece"] for topico in info}
            formatted_topics_str = self.format_llm_input_string(cluster_keywords_dict, representative_docs_dict)
            prompt=self.build_global_prompt(formatted_topics_str)
            output=self.model(prompt=prompt,**params)
            raw = output["choices"][0]["text"]
            results.append(raw)
            self.generated_interpretations.append(raw)
            os.makedirs(f"{output_data}/{modelo}",exist_ok=True)
            with open(f"{output_data}/{modelo}/global.txt", "w", encoding="utf-8") as f:
                f.write(raw)
        return results
    def export_jsonl(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in self.generated_interpretations:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get_interpretations(self)->list[str]:
        """Obtiene las interpretaciones almacenadas

        Returns:
            list[str]: conjunto de strings con las interpretaciones por tópico
        """
        return self.generated_interpretations
    def export_txt(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in self.generated_interpretations:
                f.write(item + "\n\n")
    
def read_json(src):
    with open(src,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data