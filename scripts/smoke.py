"""Environment smoke test: MLX loads a model and generates. No tool calling."""
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Say hi in 3 words."}],
    add_generation_prompt=True, tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=32))
