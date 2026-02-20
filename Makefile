.PHONY: run run-alt build-cli infer

run:
	gforth models/forth/gpt.fth

run-alt:
	gforth models/forth/gpt.4th

build-cli:
	npm run build

infer:
	node dist/cli.js --model models/forth/gpt.fth
