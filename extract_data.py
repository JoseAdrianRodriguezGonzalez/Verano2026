import re 
import json
class results_format:
    def __init__(self,src=""):
        self.src=src
        self.file=self.read_file(src)
    def read_file(self,src=""):
        with open(self.src,"r",encoding="utf-8") as f:
            return f.readlines()
    def extract_metada(self):
        line = self.file[1]
        data_line=line.strip().split("|")
        dict_metadata={section.split(":")[0].strip(): section.split(":")[1].strip() for section in data_line}
        self.metadata=dict_metadata
        return dict_metadata
    def metadata_per_model(self,src=""):
        pattern=r"---\s*(.*?)\s*---"
        results=[]
        complete_file=self.file
        for i,line in enumerate(complete_file):
            match=re.search(pattern,line)
            model_name=""
            dict_metadata=dict()
            model_name=match.group(1) if match else ""
            if i+2<len(complete_file) and match:
                metadata_line=complete_file[i+1].strip() 
                data_line=metadata_line.split("|")
                dict_metadata={
                    section.split(":")[0].strip():section.split(":")[1].strip()
                    for section in data_line 
                }
                results.append({"model":model_name,
                                "metadata":dict_metadata})
        self.metadata_models=results
        return results

    def extract_topics_keywords(self):
        pattern_model = r"---\s*(.*?)\s*---"
        pattern_topic = r"TOPICO\s+(\d+)\s+--"
        i = 0
        models = []
        current_model = None
        while i < len(self.file):
            line = self.file[i].strip()
            model_match = re.search(pattern_model, line)
            if model_match:
                if current_model:
                    models.append(current_model)
                current_model = {
                    "model": model_match.group(1),
                    "topics": []
                }
                i += 1
                continue
            topic_match = re.search(pattern_topic, line)
            if topic_match and current_model:
                topic_id = int(topic_match.group(1))
                keywords_line = self.file[i + 1]
                keywords = []
                if "Keywords:" in keywords_line:
                    keywords = keywords_line.split(":")[1].strip().split(",")
                    keywords = [k.strip() for k in keywords]
                i += 2
                if i < len(self.file) and self.file[i].strip() == "":
                    i += 1
                pertenece = []
                contraste = []
                # PERTENECE
                if i < len(self.file) and  self.file[i].strip() == "[PERTENECE]":
                    i += 1
                    while i < len(self.file) and "[CONTRASTE]" not in self.file[i]:
                        pertenece.append(self.file[i].strip())
                        i += 1

                # CONTRASTE
                if i < len(self.file) and self.file[i].strip()=="[CONTRASTE]":
                    i += 1
                    while (
                        i < len(self.file)
                        and not re.search(pattern_topic, self.file[i])
                        and not re.search(pattern_model, self.file[i])
                        and not re.match(r"^=+$", self.file[i].strip())
                    ):
                        contraste.append(self.file[i].strip())
                        i += 1

                current_model["topics"].append({
                    "topic_id": topic_id,
                    "keywords": keywords,
                    "pertenece": pertenece[:-1],
                    "contraste": contraste[:-1]
                })
                continue
            i += 1
        #  ultimo modelo
        if current_model:
            models.append(current_model)
        self.models = models
        return models   
    def save_json(self, path="output.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_info": self.metadata,
                "models_info":self.metadata_models,
                "models": self.models
            }, f, indent=4, ensure_ascii=False)    
    def change_file(self,src):
        self.file=self.read_file(src)
