import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class semantic_similarity:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        #self.model = SentenceTransformer(model_name)
        # 获取当前脚本所在的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(script_dir, "..", "models", "all-MiniLM-L6-v2")
        local_model_path = os.path.normpath(local_model_path)  # 标准化路径
        if not os.path.exists(local_model_path):
            raise RuntimeError(f"❌ Model not found at {local_model_path}. Please download it first!")
        self.model = SentenceTransformer(local_model_path)

    def sentence_similarity(self, text1, text2):
        # 编码为向量
        embeddings = self.model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

from difflib import SequenceMatcher
def word_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
if __name__ == "__main__":
    semantic_similarity = semantic_similarity()
    # 测试
    print(semantic_similarity.sentence_similarity('Leslie Road', 'Leslie RD.'))  # 输出：0.85
    # print(semantic_similarity.sentence_similarity('car', 'vehicle'))  # 输出：0.10
    print(word_similarity('car', 'Cars'))  # 输出：0.85
    print(word_similarity('Leslie Road', 'Leslie RD.'))  # 输出：0.85
