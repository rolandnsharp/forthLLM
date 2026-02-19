# forthLLM

Pure Forth GPT experiments.

## Requirements
- `gforth` (tested with `0.7.3`)

## Layout
- `gpt.fth` - main Forth GPT implementation
- `gpt.4th` - alternate Forth implementation
- `training_data/` - local corpus files (ignored by git)
- `input.txt` - local generated/training input (ignored by git)

## Run
```bash
make run
```

Directly:
```bash
gforth gpt.fth
```

## Notes
- The program executes `main` at file end and starts training immediately.
- Keep `training_data/` and `input.txt` local; they are excluded from repository tracking.
