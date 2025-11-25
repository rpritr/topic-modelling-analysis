"""
Class for Data Processing
    - init # init DataProcess, args: texts
    - preprocess_text # Basic text preprocessing, return: texts, slovenian_stopwods
"""
import classla
import os
class DataLemmatize:
    
    def __init__(self, texts):
        self.texts = texts
        
        # Load CLASSLA for lemmatization support
        resource_dir = os.environ.get("CLASSLA_RESOURCES_DIR")

        # SECONDARY fallback (local users)
        if not resource_dir:
            resource_dir = os.path.expanduser("~/.classla")

        # Check if classla resources exist

        def has_classla_resources(path: str) -> bool:
            if not os.path.isdir(path):
                return False
            for fname in os.listdir(path):
                if fname.startswith("resources") and fname.endswith(".json"):
                    return True
            return False

        if os.environ.get("CLASSLA_RESOURCES_DIR"):
            if not has_classla_resources(resource_dir):
                raise RuntimeError(
                    f"CLASSLA_RESOURCES_DIR={resource_dir}, "
                    "classla resources not found. "
                    "Check build (Apptainer %post)."
                )
            else:
                print(f"[INFO] Using prebuilt Classla models in container: {resource_dir}")
        else:
            # Local env ~/.classla
            if not has_classla_resources(resource_dir):
                print(f"[INFO] Classla models not found in {resource_dir}. Downloading...")
                os.makedirs(resource_dir, exist_ok=True)
                classla.download(
                    "sl",
                    processors="tokenize,pos,lemma",
                    dir=resource_dir
                )
            else:
                print(f"[INFO] Using existing Classla models in: {resource_dir}")

        self.nlp = classla.Pipeline(
            "sl",
            processors="tokenize,pos,lemma",
            tokenize_no_ssplit=True,
            dir=resource_dir,
        )
        
    def lemmatize_texts(self):
        lem_texts = []
        for txt in self.texts:
            doc = self.nlp(txt)
            lemmas = [w.lemma for s in doc.sentences for w in s.words]
            lem_texts.append(" ".join(lemmas))
        return lem_texts
