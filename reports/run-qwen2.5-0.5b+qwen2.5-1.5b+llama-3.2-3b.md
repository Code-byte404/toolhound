# toolprobe report

## Environment
- chip: Apple M2 Pro
- macos: 26.5
- mlx: 0.31.2
- mlx_lm: 0.31.3
- toolprobe: 0.1.0
- pa_seed: 0

## Reliability

| model | quant | method | parse_ok | schema_valid | tool_correct | args_correct |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | q4 | baseline | 0.45 [0.37, 0.53] | 0.45 [0.37, 0.53] | 0.45 [0.37, 0.53] | 0.29 [0.21, 0.37] |
| qwen2.5-0.5b | q4 | pa_tool | 0.50 [0.41, 0.59] | 0.50 [0.41, 0.59] | 0.50 [0.41, 0.59] | 0.28 [0.21, 0.37] |
| qwen2.5-1.5b | q4 | baseline | 0.96 [0.92, 0.99] | 0.96 [0.92, 0.99] | 0.96 [0.92, 0.99] | 0.71 [0.63, 0.79] |
| qwen2.5-1.5b | q4 | pa_tool | 0.96 [0.92, 0.99] | 0.96 [0.92, 0.99] | 0.96 [0.92, 0.99] | 0.43 [0.34, 0.51] |
| llama-3.2-3b | q4 | baseline | 1.00 [1.00, 1.00] | 0.68 [0.60, 0.76] | 0.98 [0.95, 1.00] | 0.61 [0.52, 0.69] |
| llama-3.2-3b | q4 | pa_tool | 0.99 [0.98, 1.00] | 0.68 [0.60, 0.76] | 0.96 [0.92, 0.99] | 0.56 [0.47, 0.65] |

## Method comparison

| model | quant | method | metric | baseline | method_val | delta | credible |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | q4 | pa_tool | parse_ok | 0.45 | 0.50 | +0.05 | no |
| qwen2.5-0.5b | q4 | pa_tool | schema_valid | 0.45 | 0.50 | +0.05 | no |
| qwen2.5-0.5b | q4 | pa_tool | tool_correct | 0.45 | 0.50 | +0.05 | no |
| qwen2.5-0.5b | q4 | pa_tool | args_correct | 0.29 | 0.28 | -0.01 | no |
| qwen2.5-1.5b | q4 | pa_tool | parse_ok | 0.96 | 0.96 | +0.00 | no |
| qwen2.5-1.5b | q4 | pa_tool | schema_valid | 0.96 | 0.96 | +0.00 | no |
| qwen2.5-1.5b | q4 | pa_tool | tool_correct | 0.96 | 0.96 | +0.00 | no |
| qwen2.5-1.5b | q4 | pa_tool | args_correct | 0.71 | 0.43 | -0.28 | no |
| llama-3.2-3b | q4 | pa_tool | parse_ok | 1.00 | 0.99 | -0.01 | no |
| llama-3.2-3b | q4 | pa_tool | schema_valid | 0.68 | 0.68 | +0.00 | no |
| llama-3.2-3b | q4 | pa_tool | tool_correct | 0.98 | 0.96 | -0.02 | no |
| llama-3.2-3b | q4 | pa_tool | args_correct | 0.61 | 0.56 | -0.05 | no |
