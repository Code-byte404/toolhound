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
| strict | bf16 | 16 | 1 | 0 | 0 | 4 |
| strict | q4 | 16 | 1 | 0 | 0 | 4 |
| lenient | bf16 | 15 | 1 | 0 | 0 | 5 |
| lenient | q4 | 16 | 1 | 0 | 0 | 4 |
