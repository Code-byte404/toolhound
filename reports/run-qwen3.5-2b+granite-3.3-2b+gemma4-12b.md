# toolprobe report

## Environment
- chip: Apple M2 Pro
- macos: 26.5
- mlx: 0.31.2
- mlx_lm: 0.31.3
- mlx_vlm: 0.6.3
- outlines_core: 0.2.14
- toolprobe: 0.1.0
- pa_seed: 0

## Reliability

| model | quant | method | decode | parse_ok | schema_valid | tool_correct | args_correct |
|---|---|---|---|---|---|---|---|
| qwen3.5-2b | q4 | baseline | free | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | 0.76 [0.69, 0.83] |
| granite-3.3-2b | q4 | baseline | free | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.75 [0.67, 0.82] |
| gemma4-12b | q4 | baseline | free | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.97 [0.94, 0.99] |
