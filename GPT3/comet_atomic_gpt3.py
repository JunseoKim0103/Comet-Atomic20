from transformers import AutoTokenizer, AutoModelForCausalLM

# Hugging Face Hub에서 모델을 불러올 경우
model = AutoModelForCausalLM.from_pretrained("mismayil/comet-gpt2-ai2")
tokenizer = AutoTokenizer.from_pretrained("mismayil/comet-gpt2-ai2")

# 로컬 파일 시스템에서 모델을 불러올 때
# model = AutoModelForCausalLM.from_pretrained("./gpt2xl-comet-atomic-2020")
# tokenizer = AutoTokenizer.from_pretrained("./gpt2xl-comet-atomic-2020")

## tf-idf 적용
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class GenerateModel:
    def __init__(self, model, tokenizer, head_event: str, relation: str):
        self.head_event = head_event
        self.relation = relation
        self.prompt = f"{head_event} {relation} [GEN]"
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self):
        inputs = self.tokenizer.encode(self.prompt, return_tensors="pt", add_special_tokens=True)
        # outputs = self.model.generate(inputs, max_length=50, num_beams=4, num_return_sequences=4)
        outputs = self.model.generate(
            inputs, 
            do_sample=True,
            max_length=50, 
            num_beams=1, 
            num_return_sequences=1,
            temperature=0.9,  # 무작위성을 조금 더 도입
            top_k=10,         # 상위 50개 토큰만 고려
            top_p=0.95,        # 상위 92% 누적 확률 분포를 고려
        )
        prompt_length = len(self.tokenizer.encode(self.prompt, add_special_tokens=True)) - 1
        result_texts = [self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip() for output in outputs]
        return result_texts

    def filter_similar_responses(self, responses, similarity_threshold=0.8):
        # TF-IDF 벡터화
        tfidf_vectorizer = TfidfVectorizer().fit_transform(responses)
        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_vectorizer, tfidf_vectorizer)
        # 유사도에 따라 결과 필터링
        filtered_indices = []
        for i in range(cosine_sim.shape[0]):
            if i not in filtered_indices:
                # i번째 문서와 유사한 문서들을 찾음
                similar_indices = np.where(cosine_sim[i] > similarity_threshold)[0]
                # 첫 번째 유사한 문서만 유지
                filtered_indices.append(similar_indices[0])
        
        filtered_responses = [responses[i] for i in filtered_indices]
        return filtered_responses

if __name__ == "__main__":
    input_head_event = input("1. Input head_event: ")
    input_relation = input("2. Input relation: ")
    atomic_model = GenerateModel(model, tokenizer, input_head_event, input_relation)
    result_list = atomic_model.generate_response()
    filtered_result_list = atomic_model.filter_similar_responses(result_list)
    print(filtered_result_list)
    for i, result in enumerate(filtered_result_list):
        print(f"Output {i+1}: {result}")

