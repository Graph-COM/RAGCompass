def query_llm_hf(prompt, tokenizer, model, max_new_tokens, temperature=0):
    # Generate input_ids and attention_mask together
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # "<|eot_id|>" is Llama-3-specific. Other readers (e.g. Qwen2.5) do not have
    # it, and convert_tokens_to_ids then returns None/unk, which reaches
    # generate() as eos_token_id=[<id>, None] and raises
    # "TypeError: 'NoneType' object cannot be interpreted as an integer".
    # Keep it when the tokenizer really has it -- Llama's behaviour is
    # unchanged -- and drop it otherwise.
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in terminators:
        terminators.append(eot_id)
    terminators = [token_id for token_id in terminators if token_id is not None]

    # Build generation kwargs conditionally
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.pad_token_id
    }
    
    # Only add attention_mask if available
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    
    # Handle sampling parameters carefully
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        # Note: Some models don't support temperature parameter
        # For better compatibility, we'll omit temperature and rely on do_sample=True
    else:
        generation_kwargs["do_sample"] = False
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
    
    outputs = model.generate(**generation_kwargs)
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
