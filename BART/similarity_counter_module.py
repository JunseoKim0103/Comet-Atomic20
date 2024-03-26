# similarity_counter_module.py
from comet_module import Comet
from sentence_transformers import SentenceTransformer, util

class CometWithSimilarityCounter(Comet):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.results_cache = {}
        self.similarity_threshold = 0.7

    def add_and_count_similar_results(self, results):
        for result in results:
            is_new = True
            result_embedding = self.similarity_model.encode([result], convert_to_tensor=True)
            
            for existing_result, data in self.results_cache.items():
                existing_embedding = data['embedding']
                similarity = util.pytorch_cos_sim(result_embedding, existing_embedding)[0][0]
                
                if similarity >= self.similarity_threshold:
                    self.results_cache[existing_result]['count'] += 1
                    is_new = False
                    break
            
            if is_new:
                self.results_cache[result] = {'count': 1, 'embedding': result_embedding}

        # 결과 출력
        for result, data in self.results_cache.items():
            print(f"Result: {result}, Count: {data['count']}")
