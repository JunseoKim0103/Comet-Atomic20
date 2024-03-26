# main.py
from similarity_counter_module import CometWithSimilarityCounter

if __name__ == "__main__":
    print("Model loading ...")
    comet = CometWithSimilarityCounter("mismayil/comet-bart-ai2")
    print("Model loaded!")
    queries = ["PersonX drinks water everyday xEffect [GEN]"]
    print("Query: ", queries)
    results = comet.generate(queries, decode_method="beam", num_generate=3)
    print("Initial results: ", results)
    comet.add_and_count_similar_results(results)

    # You can add more code here to generate and count additional results as needed.
