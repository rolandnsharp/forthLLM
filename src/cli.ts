#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';

interface CliOptions {
  model?: string;
  engine: string;
  timeoutSec: number;
  json: boolean;
  raw: boolean;
}

interface RunResult {
  exitCode: number | null;
  timedOut: boolean;
  stdout: string;
  stderr: string;
}

interface ParsedOutput {
  model: string;
  docs?: number;
  vocabSize?: number;
  numParams?: number;
  lastStep?: number;
  totalSteps?: number;
  lastLoss?: number;
  generations: string[];
}

function listModels(): string[] {
  const roots = [join(process.cwd(), 'models', 'forth'), process.cwd()];
  const out: string[] = [];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    const relRoot = root === process.cwd() ? '' : 'models/forth/';
    for (const f of readdirSync(root)) {
      if (f.endsWith('.fth') || f.endsWith('.4th')) out.push(`${relRoot}${f}`);
    }
  }
  return [...new Set(out)].sort();
}

function usage(): never {
  const models = listModels();
  const modelHint = models.length ? models.join(', ') : 'gpt.fth';
  console.error(
    [
      'Usage:',
      '  forth-infer --model <file> [--timeoutSec 180] [--engine gforth] [--json] [--raw]',
      '',
      `Available models: ${modelHint}`,
    ].join('\n')
  );
  process.exit(1);
}

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    engine: 'gforth',
    timeoutSec: 180,
    json: false,
    raw: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--model') {
      opts.model = argv[++i];
    } else if (a === '--engine') {
      opts.engine = argv[++i] ?? opts.engine;
    } else if (a === '--timeoutSec') {
      const n = Number(argv[++i]);
      if (!Number.isFinite(n) || n <= 0) usage();
      opts.timeoutSec = n;
    } else if (a === '--json') {
      opts.json = true;
    } else if (a === '--raw') {
      opts.raw = true;
    } else if (a === '--help' || a === '-h') {
      usage();
    } else {
      usage();
    }
  }

  if (!opts.model) {
    const defaults = ['models/forth/gpt.fth', 'models/forth/gpt.4th', 'gpt.fth', 'gpt.4th'];
    opts.model = defaults.find((f) => existsSync(f));
  }

  if (!opts.model) usage();
  return opts;
}

function runModel(engine: string, model: string, timeoutSec: number): Promise<RunResult> {
  return new Promise((resolve) => {
    const child = spawn(engine, [model], {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    child.stdout.on('data', (d: Buffer) => {
      stdout += d.toString('utf8');
    });
    child.stderr.on('data', (d: Buffer) => {
      stderr += d.toString('utf8');
    });

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
      setTimeout(() => child.kill('SIGKILL'), 2000).unref();
    }, timeoutSec * 1000);

    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ exitCode: code, timedOut, stdout, stderr });
    });
  });
}

function parseOutput(model: string, stdout: string): ParsedOutput {
  const lines = stdout
    .split(/\r?\n/)
    .map((l) => l.trimEnd())
    .filter((l) => l.length > 0);

  const out: ParsedOutput = {
    model: basename(model),
    generations: [],
  };

  for (const l of lines) {
    const docs = l.match(/^num docs:\s+(\d+)/i);
    if (docs) out.docs = Number(docs[1]);
    const vocab = l.match(/^vocab size:\s+(\d+)/i);
    if (vocab) out.vocabSize = Number(vocab[1]);
    const params = l.match(/^num params:\s+(\d+)/i);
    if (params) out.numParams = Number(params[1]);

    const step = l.match(/^step\s+(\d+)\s*\/\s*(\d+)\s*\|\s*loss\s+([\d.+\-eE]+)/i);
    if (step) {
      out.lastStep = Number(step[1]);
      out.totalSteps = Number(step[2]);
      out.lastLoss = Number(step[3]);
    }

    const reply = l.match(/^reply\s+\d+\s*:\s*(.*)$/i);
    if (reply) out.generations.push(reply[1]);

    const sample = l.match(/^sample\s+\d+\s*:\s*(.*)$/i);
    if (sample) out.generations.push(sample[1]);

    const bracket = l.match(/^\[\s*\d+\s*\]\s*(.*)$/);
    if (bracket) out.generations.push(bracket[1]);
  }

  return out;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const model = opts.model as string;

  if (!existsSync(model)) {
    console.error(`Model not found: ${model}`);
    process.exit(2);
  }

  const run = await runModel(opts.engine, model, opts.timeoutSec);

  if (opts.raw) {
    process.stdout.write(run.stdout);
    if (run.stderr) process.stderr.write(run.stderr);
    process.exit(run.timedOut ? 124 : (run.exitCode ?? 1));
  }

  const parsed = parseOutput(model, run.stdout);
  if (opts.json) {
    console.log(
      JSON.stringify(
        {
          ...parsed,
          timedOut: run.timedOut,
          exitCode: run.exitCode,
        },
        null,
        2
      )
    );
  } else {
    console.log(`model: ${parsed.model}`);
    if (parsed.docs !== undefined) console.log(`num docs: ${parsed.docs}`);
    if (parsed.vocabSize !== undefined) console.log(`vocab size: ${parsed.vocabSize}`);
    if (parsed.numParams !== undefined) console.log(`num params: ${parsed.numParams}`);
    if (parsed.lastStep !== undefined && parsed.totalSteps !== undefined && parsed.lastLoss !== undefined) {
      console.log(`last step: ${parsed.lastStep}/${parsed.totalSteps}`);
      console.log(`last loss: ${parsed.lastLoss}`);
    }
    console.log(`timed out: ${run.timedOut}`);
    if (parsed.generations.length) {
      console.log('inference:');
      for (const [i, g] of parsed.generations.entries()) {
        console.log(`  ${i + 1}. ${g}`);
      }
    } else {
      console.log('inference: (none captured)');
    }
  }

  if (run.exitCode && !run.timedOut) process.exit(run.exitCode);
  if (run.timedOut) process.exit(124);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
