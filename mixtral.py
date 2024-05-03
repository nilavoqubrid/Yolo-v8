
import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig

device = "cuda:1"

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device
        )
    return model, tokenizer


def generate_response(model, tokenizer, user_query, llm_new_tokens):

    prompt = user_query
    
    # Encode the input text
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate response using the model
    generation_config = GenerationConfig(
        penalty_alpha=0.6, 
        do_sample=True, 
        top_k=5, 
        temperature=0.1, 
        repetition_penalty=1.2,
        max_new_tokens=llm_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs = model.generate(**input_ids, generation_config=generation_config)

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = generated_text.split(f"{user_query} \nmodel")[-1].strip()
    
    return response



model, tokenizer = load_model(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1")

# user_input = "Who is the first PM of India?"
print()
user_input = input("Question:")
llm_new_tokens = 100
llm_response = generate_response(
                        model,
                        tokenizer,
                        user_input,
                        llm_new_tokens
                        )

print()
print("Response:", llm_response)
print()
