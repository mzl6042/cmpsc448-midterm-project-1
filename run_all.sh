#!/bin/bash
# Reproduces every number in the report (CPU only, ~4-5 hours on 2 cores).
# We actually ran it as two parallel queues (queue A = LSTM, queue B = everything else).
set -e
cd "$(dirname "$0")"

# 0) data
if [ ! -d alpaca_eval ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/tatsu-lab/alpaca_eval.git
  (cd alpaca_eval && git sparse-checkout set results --no-cone)
fi
python data/build_dataset.py --alpaca_dir alpaca_eval/results

queue_a() {   # RNN (BiLSTM) for RQ1 / RQ2
  for mode in output both input; do
    for seed in 0 1; do
      OMP_NUM_THREADS=1 python src/train.py --model lstm --mode $mode --seed $seed
    done
  done
}

queue_b() {   # CNN for RQ1 / RQ2, then extra analyses
  for mode in output both input; do
    for seed in 0 1 2; do
      OMP_NUM_THREADS=1 python src/train.py --model cnn --mode $mode --seed $seed
    done
  done
  python src/baseline_tfidf.py
  OMP_NUM_THREADS=1 python src/rq2_bias.py
  OMP_NUM_THREADS=1 python src/rq3_cross_task.py --seeds 1
  python src/rq4_features.py
  OMP_NUM_THREADS=1 python src/rq4_ablation.py --seeds 1
}

mkdir -p logs
queue_a > logs/queue_a.txt 2>&1 &
queue_b > logs/queue_b.txt 2>&1
wait
python src/make_figures.py
