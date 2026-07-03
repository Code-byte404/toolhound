# Toolhound demo — renders assets/demo.gif
#
# Requires charmbracelet/vhs (https://github.com/charmbracelet/vhs):
#     brew install vhs
# and the models already downloaded/cached (so the GIF isn't dominated by the
# first-run HF download). Warm the cache once before recording:
#     toolprobe run --model qwen2.5-0.5b --quant q4 --cases cases/smoke.jsonl
# Then record:
#     vhs demo.tape
#
# Tune the Sleep/Wait values to your machine — MLX load time varies.

Output assets/demo.gif

Set Shell zsh
Set FontSize 18
Set Width 1200
Set Height 660
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set PlaybackSpeed 1.0

# --- warm up off-camera so the GIF opens on a clean prompt ---
Hide
Type "conda activate toolhound && clear" Enter
Sleep 1s
Show

Sleep 800ms
Type "# Toolhound: which layer broke the tool call?"
Enter
Sleep 1s

# --- reliability run on the 3-case smoke set (fast, real) ---
Type "toolprobe run --model qwen2.5-0.5b --quant q4 --cases cases/smoke.jsonl"
Enter
Wait@90s      # waits for the command to finish (model load + generation)
Sleep 2.5s    # let the reliability table sit on screen

# --- attribution: name the culprit ---
Type "toolprobe attribute --model qwen2.5-0.5b"
Enter
Wait@90s
Sleep 3.5s    # hold on the four-cause table — the payoff frame

Type "# ↑ strict + lenient tiers, so the verdict doesn't flip with parser leniency"
Enter
Sleep 3s
