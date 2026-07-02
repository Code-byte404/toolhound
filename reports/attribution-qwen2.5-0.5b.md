# toolprobe report

## Environment
- chip: Apple M2 Pro
- macos: 26.5
- mlx: 0.31.2
- mlx_lm: 0.31.3
- toolprobe: 0.1.0

## Attribution

| leniency | quant | pass | framework_template_bug | framework_parser_gap | model_format_failure | model_decision_failure |
|---|---|---|---|---|---|---|
| strict | bf16 | 4 | 16 | 0 | 0 | 1 |
| strict | q4 | 4 | 17 | 0 | 0 | 0 |
| lenient | bf16 | 3 | 16 | 0 | 0 | 2 |
| lenient | q4 | 4 | 17 | 0 | 0 | 0 |
