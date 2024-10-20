from beam import Image, endpoint, env

if env.is_remote():
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    import nltk
    import sentencepiece

# Download nltk resources (only in remote environment)
if env.is_remote():
    nltk.download('punkt')
    nltk.download('punkt_tab')

# DipperParaphraser class definition (same as you had before)
class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')  # Use T5 tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        paragraphs = [input_text] if '\n' not in input_text else [p.strip() for p in input_text.split('\n') if p.strip()]
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            paragraph_output = ""

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
                final_input_text = f"lexical = {lex_code}, order = {order_code} <sent> {curr_sent_window} </sent>"
                final_input = self.tokenizer([final_input_text], return_tensors="pt").input_ids.cuda()

                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids=final_input,
                        do_sample=kwargs.get('do_sample', True),
                        top_p=kwargs.get('top_p', 0.75),
                        top_k=kwargs.get('top_k', None),
                        max_length=kwargs.get('max_length', 512)
                    )
                output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                paragraph_output += " " + output

            output_text += paragraph_output.strip() + "\n\n"

        return output_text.strip()

    def regenerate_if_too_short(self, input_text, output_text, min_ratio=0.5, **kwargs):
        input_len = len(input_text)
        output_len = len(output_text)

        while output_len < min_ratio * input_len:
            print("Regenerating because the output is too short...")
            output_text = self.paraphrase(input_text, **kwargs)
            output_len = len(output_text)

        return output_text

# Define the endpoint using Beam's decorator
@endpoint(
    name="paraphrase-inference",
    cpu=4,
    memory="32Gi",
    gpu="A100-40",
    image=Image(
        python_version="python3.9",
        python_packages=[
            "transformers",
            "torch",
            "nltk",  # Add nltk for tokenization
            "sentencepiece"  # Add sentencepiece for T5 tokenizer
        ],
    ),
)
def predict(request):
    # Initialize the DipperParaphraser model with your custom model
    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")  # This line loads your custom model

    # Parse the input text from the request
    text = request.get("text", "")
    lex_diversity = request.get("lex_diversity", 80)
    order_diversity = request.get("order_diversity", 0)
    prefix = request.get("prefix", "")
    do_sample = request.get("do_sample", True)
    top_p = request.get("top_p", 0.75)
    top_k = request.get("top_k", None)
    max_length = request.get("max_length", 512)
    min_ratio = request.get("min_ratio", 0.5)

    # Generate the paraphrase and ensure it meets the min_ratio requirement
    output = dp.paraphrase(
        text,
        lex_diversity=lex_diversity,
        order_diversity=order_diversity,
        prefix=prefix,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        max_length=max_length
    )

    # Regenerate if the output is too short
    output = dp.regenerate_if_too_short(
        input_text=text,
        output_text=output,
        min_ratio=min_ratio,
        lex_diversity=lex_diversity,
        order_diversity=order_diversity,
        prefix=prefix,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        max_length=max_length
    )
    
    return {"paraphrased_text": output, "input_length": len(text), "output_length": len(output)}
