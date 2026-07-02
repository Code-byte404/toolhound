# toolprobe report

## Environment
- chip: Apple M2 Pro
- macos: 26.5
- mlx: 0.31.2
- mlx_lm: 0.31.3
- toolprobe: 0.1.0

## Reliability

| model | quant | parse_ok | schema_valid | tool_correct | args_correct |
|---|---|---|---|---|---|
| qwen2.5-0.5b | bf16 | 0.88 [0.71, 1.00] | 0.88 [0.71, 1.00] | 0.88 [0.71, 1.00] | 0.59 [0.35, 0.82] |
| qwen2.5-0.5b | q4 | 0.59 [0.35, 0.82] | 0.59 [0.35, 0.82] | 0.59 [0.35, 0.82] | 0.47 [0.24, 0.71] |
