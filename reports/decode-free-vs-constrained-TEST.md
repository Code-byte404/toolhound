# toolprobe report

## Environment
- chip: Apple M2 Pro
- macos: 26.5
- mlx: 0.31.2
- mlx_lm: 0.31.3
- outlines_core: 0.2.14
- toolprobe: 0.1.0
- pa_seed: 0

## Reliability

| model | quant | method | decode | parse_ok | schema_valid | tool_correct | args_correct |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | q4 | baseline | free | 0.50 [0.41, 0.58] | 0.50 [0.41, 0.58] | 0.50 [0.41, 0.58] | 0.31 [0.23, 0.38] |
| qwen2.5-0.5b | q4 | baseline | constrained | 0.76 [0.69, 0.83] | 0.76 [0.69, 0.83] | 0.76 [0.69, 0.83] | 0.52 [0.44, 0.60] |
| qwen2.5-1.5b | q4 | baseline | free | 0.97 [0.94, 0.99] | 0.97 [0.94, 0.99] | 0.97 [0.94, 0.99] | 0.70 [0.63, 0.78] |
| qwen2.5-1.5b | q4 | baseline | constrained | 0.94 [0.90, 0.98] | 0.94 [0.90, 0.98] | 0.94 [0.90, 0.98] | 0.53 [0.44, 0.61] |
| llama-3.2-3b | q4 | baseline | free | 1.00 [1.00, 1.00] | 0.68 [0.60, 0.76] | 0.98 [0.96, 1.00] | 0.60 [0.52, 0.68] |
| llama-3.2-3b | q4 | baseline | constrained | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.98 [0.96, 1.00] | 0.71 [0.64, 0.79] |

## Decode comparison (constrained vs free)

| model | quant | method | metric | free | constrained | delta | credible |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | q4 | baseline | parse_ok | 0.50 | 0.76 | +0.27 | yes |
| qwen2.5-0.5b | q4 | baseline | schema_valid | 0.50 | 0.76 | +0.27 | yes |
| qwen2.5-0.5b | q4 | baseline | tool_correct | 0.50 | 0.76 | +0.27 | yes |
| qwen2.5-0.5b | q4 | baseline | args_correct | 0.31 | 0.52 | +0.21 | yes |
| qwen2.5-1.5b | q4 | baseline | parse_ok | 0.97 | 0.94 | -0.03 | no |
| qwen2.5-1.5b | q4 | baseline | schema_valid | 0.97 | 0.94 | -0.03 | no |
| qwen2.5-1.5b | q4 | baseline | tool_correct | 0.97 | 0.94 | -0.03 | no |
| qwen2.5-1.5b | q4 | baseline | args_correct | 0.70 | 0.53 | -0.18 | no |
| llama-3.2-3b | q4 | baseline | parse_ok | 1.00 | 1.00 | +0.00 | no |
| llama-3.2-3b | q4 | baseline | schema_valid | 0.68 | 1.00 | +0.32 | yes |
| llama-3.2-3b | q4 | baseline | tool_correct | 0.98 | 0.98 | +0.00 | no |
| llama-3.2-3b | q4 | baseline | args_correct | 0.60 | 0.71 | +0.11 | no |
