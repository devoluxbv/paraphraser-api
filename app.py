import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.tokenize import sent_tokenize
from typing import Optional

# Download both punkt and punkt_tab
nltk.download('punkt')
nltk.download('punkt_tab')

app = FastAPI()

class ParaphraseRequest(BaseModel):
    text: str
    lex_diversity: Optional[int] = 80
    order_diversity: Optional[int] = 0
    prefix: Optional[str] = ""  # Default prefix is now empty
    do_sample: Optional[bool] = True
    top_p: Optional[float] = 0.75
    top_k: Optional[int] = None
    max_length: Optional[int] = 512


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model."""
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        # Handle both single and multiple paragraphs
        paragraphs = [input_text] if '\n' not in input_text else [p.strip() for p in input_text.split('\n') if p.strip()]
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            paragraph_output = ""

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
                final_input_text = f"lexical = {lex_code}, order = {order_code}"
                if prefix:
                    final_input_text += f" {prefix}"
                final_input_text += f" <sent> {curr_sent_window} </sent>"

                final_input = self.tokenizer([final_input_text], return_tensors="pt")
                final_input = {k: v.cuda() for k, v in final_input.items()}

                with torch.inference_mode():
                    outputs = self.model.generate(**final_input, **kwargs)
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += " " + outputs[0]
                paragraph_output += " " + outputs[0]

            output_text += paragraph_output.strip() + "\n\n"  # Add paragraph breaks between outputs

        return output_text.strip()


# Initialize the paraphraser
dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

@app.post("/paraphrase")
def paraphrase_text(request: ParaphraseRequest):
    start_time = time.time()
    
    output = dp.paraphrase(
        request.text,
        lex_diversity=request.lex_diversity,
        order_diversity=request.order_diversity,
        prefix=request.prefix,
        do_sample=request.do_sample,
        top_p=request.top_p,
        top_k=request.top_k,
        max_length=request.max_length
    )
    
    processing_time = time.time() - start_time
    
    return {
        "paraphrased_text": output,
        "processing_time_seconds": processing_time
    }

# To run the FastAPI server, use this command in terminal:
# uvicorn your_script_name:app --host 0.0.0.0 --port 5000