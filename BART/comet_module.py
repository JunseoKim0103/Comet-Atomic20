# comet_module.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.batch_size = 1

    def generate(self, queries, decode_method="beam", use_sampling=True, temperature=1.0, num_generate=5, max_new_tokens=50):
        gen_kwargs = {
            "num_return_sequences": num_generate,
            "num_beams": num_generate if decode_method == "beam" else 1,
            "early_stopping": True if decode_method == "beam" else None,
            "max_new_tokens": max_new_tokens
        }

        if use_sampling:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
            })
            gen_kwargs.pop("num_beams", None)
            gen_kwargs.pop("early_stopping", None)
        else:
            gen_kwargs.update({
                "num_beams": num_generate if decode_method == "beam" else 1,
                "early_stopping": True if decode_method == "beam" else None,
            })

        decs = []
        with torch.no_grad():
            for batch in self.chunks(queries, self.batch_size):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                summaries = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
                dec = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                decs.extend(dec)
        return decs

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
