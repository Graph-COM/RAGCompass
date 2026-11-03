import time

from openai import OpenAI
from transformers import AutoTokenizer

model_map = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

def init_tokenizer_client(model, port):
    tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123")
    return tokenizer, client

def query_llm(prompt, tokenizer, client, model, max_new_tokens, base_timeout=60, temperature=0.0):
    input_ids = tokenizer.encode(str(prompt))
        
    if model in model_map:
        model = model_map[model]
    
    # Calculate dynamic timeout based on input length
    input_length = len(input_ids)
    # Use a more reasonable scaling function that doesn't grow linearly for very large inputs
    # This caps the maximum timeout at a reasonable value while still scaling with input size
    max_additional_timeout = 1800  # Maximum additional timeout of 30 minutes
    scaling_factor = min(input_length / 1000, 1000) * 3.6  # Cap at 1M tokens for timeout calculation
    dynamic_timeout = base_timeout + min(scaling_factor, max_additional_timeout)

    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout=dynamic_timeout,  # Dynamic timeout based on context length
            )
            
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        raise ValueError

PROMPT_JSON_TEMPLATE = {
    "ner": {
        "type": "object",
        "properties": {
            "named_entities": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 0
            }
        },
        "required": ["named_entities"]
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["triples"]
    },
    # R3AG retrieval-quality judge rubric (Zhao et al. 2026, Appendix A.4):
    # relevance / coverage / accuracy / ranking / redundancy, each 1-5. We ask
    # only for the five sub-scores and average them ourselves in post-processing
    # rather than also trusting the model's own "overall_score" arithmetic.
    "retrieval_quality": {
        "type": "object",
        "properties": {
            "relevance_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "coverage_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "accuracy_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "ranking_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "redundancy_score": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": [
            "relevance_score",
            "coverage_score",
            "accuracy_score",
            "ranking_score",
            "redundancy_score",
        ],
    },
}
