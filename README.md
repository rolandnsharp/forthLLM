# forthLLM

Pure Forth GPT experiments with a TypeScript CLI runner.

## Requirements
- `gforth` (tested with `0.7.3`)

## Layout
- `models/forth/gpt.fth` - main Forth GPT implementation
- `models/forth/gpt.4th` - alternate Forth implementation
- `models/forth/hello.fth` - tiny Forth example
- `src/cli.ts` - TypeScript CLI (`forth-infer`) for running inference on Forth models
- `training_data/` - local corpus files (ignored by git)
- `input.txt` - local generated/training input (ignored by git)

## Run
```bash
make run
```

Directly:
```bash
gforth models/forth/gpt.fth
```

## CLI Inference Tool (TypeScript)
Build:
```bash
npm install
npm run build
```

Run inference through a selected Forth model:
```bash
node dist/cli.js --model models/forth/gpt.fth
node dist/cli.js --model models/forth/gpt.4th --timeoutSec 60
```

JSON output:
```bash
node dist/cli.js --model models/forth/gpt.fth --json
```

## Notes
- The program executes `main` at file end and starts training immediately.
- Keep `training_data/` and `input.txt` local; they are excluded from repository tracking.
