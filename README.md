# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 course project. We train a **CNN** and an **RNN (BiLSTM)** to predict which
LLM family (GPT, Claude, Llama, Qwen, Mistral, Google) wrote a response, and study
whether the user prompt helps (RQ1, RQ2), whether fingerprints transfer across task
types (RQ3), and which signals distinguish the families (RQ4).

The report is in [`report/report.pdf`](report/report.pdf).

## Data

All data come from the public **AlpacaEval** repository
(<https://github.com/tatsu-lab/alpaca_eval>, Apache-2.0), which stores the raw outputs of
many LLMs on the same 805 instructions (`results/<model>/model_outputs.json`).
We use 12 official vendor models, 2 per family:

| Family | Models |
|---|---|
| GPT | gpt-4-1106-preview, gpt-4o-2024-05-13 |
| Claude | claude-3-opus-20240229, claude-3.5-sonnet-20240620 |
| Llama | Meta-Llama-3-70B-Instruct, Llama-2-70b-chat |
| Qwen | Qwen1.5-72B-Chat, Qwen2-72B-Instruct |
| Mistral | mistral-large-2402, Mixtral-8x7B-Instruct-v0.1 |
| Google | gemini-pro, gemma-7b-it |

`data/build_dataset.py` turns these into `data/dataset.csv` with columns
`prompt_id, split, task_type, source_subset, llm_family, llm_model, llm_input, llm_output`
(9,636 rows = 803 prompts x 12 models, perfectly balanced: 1,606 per family).
Train/val/test = 70/15/15 **split by prompt**, so test prompts are never seen in training.

## Code

| File | What it does |
|---|---|
| `data/build_dataset.py` | download-format -> `dataset.csv`, prompt-level split, task-type tags |
| `src/text_utils.py` | tokenizer (keeps punctuation, markdown and newlines), vocab, input modes, text ablations |
| `src/models.py` | `TextCNN` (kernels 3/4/5 x 128 filters) and `BiLSTM` (128 hidden/direction, max+mean pooling) |
| `src/train.py` | training loop (Adam, early stopping on val macro-F1), RQ1/RQ2 runs |
| `src/baseline_tfidf.py` | TF-IDF + logistic regression reference model |
| `src/rq2_bias.py` | RQ2(iii): simulated collection bias, input-only CNN |
| `src/rq3_cross_task.py` | RQ3: train without a task type, test on it |
| `src/rq4_features.py` | RQ4: stylometric features, feature-group classifiers, distinctive words |
| `src/rq4_ablation.py` | RQ4: retrain CNN after removing formatting / punctuation / word order / length |
| `src/make_figures.py` | aggregates `results/` into `results/summary.json` and `figures/` |

## Reproduce

```bash
pip install torch pandas scikit-learn matplotlib reportlab
bash run_all.sh            # CPU only, a few hours on 2 cores
python report/build_report.py
```

All raw per-run results (metrics, confusion matrices, training curves, predictions)
are in `results/`.
