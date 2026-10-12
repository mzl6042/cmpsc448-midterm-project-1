# Who Wrote It? - CMPSC 448 Project

For this project, we have attempted to identify which LLM wrote a response based on their writing style. We have trained a CNN and an RNN (BiLSTM) to classify responses to 6 different LLM families: GPT, Claude, Llama, Qwen, Mistral, and Google.

We have also tried to answer a few other questions. RQ1 and RQ2 are about whether including the original user prompt is helpful for prediction. RQ3 asks whether the models can generalize to different task types. RQ4 classifies the writing patterns of each LLM to understand why the models can distinguish them.

The full report is in `report/report.pdf`.

## Dataset

We have used the public [AlpacaEval repository](https://github.com/tatsu-lab/alpaca_eval) under the Apache-2.0 license. It contains responses of different LLMs to the same instructions.

We have used 12 models in total, 2 for each family:

- GPT: gpt-4-1106-preview, gpt-4o-2024-05-13

- Claude: claude-3-opus-20240229, claude-3.5-sonnet-20240620

- Llama: Meta-Llama-3-70B-Instruct, Llama-2-70b-chat

- Qwen: Qwen1.5-72B-Chat, Qwen2-72B-Instruct

- Mistral: mistral-large-2402, Mixtral-8x7B-Instruct-v0.1

- Google: gemini-pro, gemma-7b-it

We have used `data/build_dataset.py` to build the dataset. It has 9,636 responses from 803 prompts, with 1,606 responses for each family.

We have split it to 70% training, 15% validation, and 15% testing sets. The split is done on the prompt level, so that the same prompt does not appear in both training and testing sets.

The dataset is stored in `data/dataset.csv`. It also contains information about the prompt, the model name, the LLM family, the response, and the task type.

## Code

The code is mostly in the `src` folder.

`models.py` defines our CNN and BiLSTM models. `text_utils.py` contains tokenization and other preprocessing utilities, and `train.py` is used to train the models and run the RQ1 and RQ2 experiments.

We have also used `baseline_tfidf.py` for a baseline TF-IDF and logistic regression model, to compare with our neural networks.

For other experiments:

- `rq2_bias.py` checks how much collection bias can impact the results.

- `rq3_cross_task.py` evaluates the models on different task types.

- `rq4_features.py` classifies the writing patterns and difference between LLM families.

- `rq4_ablation.py` tries removing punctuation, formatting, and word order to see how much they impact the results.

- `make_figures.py` makes the figures and summarizes the results.

## How to Run

First, install the required packages:

```bash

pip install torch pandas scikit-learn matplotlib reportlab

```

Then, run the experiments:

```bash

bash run_all.sh

```

It will run on CPU and may take several hours, depending on the machine.

To generate the report:

```bash

python report/build_report.py

```

## Results

The results from the experiments are in the `results` folder. It contains the model accuracy, F1 scores, confusion matrices, predictions, and training curves.

The summary is in `results/summary.json`, and the figures are in the `figures` folder.

The main purpose of the project is to see if different LLMs have distinguishable writing patterns, and how accurately CNN and RNN models can identify them.