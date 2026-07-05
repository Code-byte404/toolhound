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
| qwen2.5-0.5b | q4 | baseline | free | 0.45 [0.37, 0.53] | 0.45 [0.37, 0.53] | 0.45 [0.37, 0.53] | 0.29 [0.21, 0.37] |
| qwen2.5-0.5b | q4 | baseline | constrained | 0.78 [0.70, 0.85] | 0.78 [0.70, 0.85] | 0.78 [0.70, 0.85] | 0.55 [0.47, 0.63] |
| qwen2.5-1.5b | q4 | baseline | free | 0.96 [0.92, 0.99] | 0.96 [0.92, 0.99] | 0.96 [0.92, 0.99] | 0.71 [0.63, 0.79] |
| qwen2.5-1.5b | q4 | baseline | constrained | 0.92 [0.86, 0.96] | 0.92 [0.86, 0.96] | 0.92 [0.86, 0.96] | 0.64 [0.56, 0.73] |
| llama-3.2-3b | q4 | baseline | free | 1.00 [1.00, 1.00] | 0.68 [0.60, 0.76] | 0.98 [0.95, 1.00] | 0.60 [0.52, 0.68] |
| llama-3.2-3b | q4 | baseline | constrained | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.98 [0.95, 1.00] | 0.70 [0.62, 0.78] |

## Decode comparison (constrained vs free)

| model | quant | method | metric | free | constrained | delta | credible |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | q4 | baseline | parse_ok | 0.45 | 0.78 | +0.33 | yes |
| qwen2.5-0.5b | q4 | baseline | schema_valid | 0.45 | 0.78 | +0.33 | yes |
| qwen2.5-0.5b | q4 | baseline | tool_correct | 0.45 | 0.78 | +0.33 | yes |
| qwen2.5-0.5b | q4 | baseline | args_correct | 0.29 | 0.55 | +0.26 | yes |
| qwen2.5-1.5b | q4 | baseline | parse_ok | 0.96 | 0.92 | -0.05 | no |
| qwen2.5-1.5b | q4 | baseline | schema_valid | 0.96 | 0.92 | -0.05 | no |
| qwen2.5-1.5b | q4 | baseline | tool_correct | 0.96 | 0.92 | -0.05 | no |
| qwen2.5-1.5b | q4 | baseline | args_correct | 0.71 | 0.64 | -0.07 | no |
| llama-3.2-3b | q4 | baseline | parse_ok | 1.00 | 1.00 | +0.00 | no |
| llama-3.2-3b | q4 | baseline | schema_valid | 0.68 | 1.00 | +0.32 | yes |
| llama-3.2-3b | q4 | baseline | tool_correct | 0.98 | 0.98 | +0.00 | no |
| llama-3.2-3b | q4 | baseline | args_correct | 0.60 | 0.70 | +0.10 | no |
