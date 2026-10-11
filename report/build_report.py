"""Builds report/report.pdf from results/summary.json + figures/ (run from repo root)."""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

S = json.load(open("results/summary.json"))
# FULL=1 adds the extra-credit RQ3/RQ4 section once those runs are finished
FULL = os.environ.get("FULL", "0") == "1"
D = json.load(open("results/dataset_stats.json"))
FAM = ["Claude", "GPT", "Google", "Llama", "Mistral", "Qwen"]

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=10.5, leading=13.6,
                      alignment=TA_JUSTIFY, spaceAfter=5)
small = ParagraphStyle("s", parent=body, fontSize=8.8, leading=10.8, alignment=TA_CENTER,
                       textColor=colors.HexColor("#333333"), spaceAfter=8)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13.5,
                    spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1a1a1a"), keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                    spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1a1a1a"), keepWithNext=1)
title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21)
cell = ParagraphStyle("c", parent=body, fontSize=8.6, leading=10.4, alignment=TA_CENTER, spaceAfter=0)
cell_l = ParagraphStyle("cl", parent=cell, alignment=0)
mono = ParagraphStyle("m", parent=cell_l, fontName="Courier", fontSize=7.6, leading=9.2)

story = []
P = lambda t, st=body: story.append(Paragraph(t, st))
fig_no = [0]
tab_no = [0]


def fig(path, caption, width=6.3):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    fig_no[0] += 1
    story.append(KeepTogether([Image(path, width=width * inch, height=width * inch * ih / iw),
                               Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", small)]))
    return fig_no[0]


def table(rows, caption, col_widths=None, header_rows=1, left_cols=0, font=None):
    tab_no[0] += 1
    data = []
    for r_i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else
                     Paragraph(("<b>%s</b>" % c) if r_i < header_rows else c,
                               font or (cell_l if c_i < left_cols else cell))
                     for c_i, c in enumerate(r)])
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", small), t,
                               Spacer(1, 8)]))
    return tab_no[0]


pct = lambda x: f"{100 * x:.1f}"
pm = lambda d: f"{100 * d['acc']:.1f} ± {100 * d['acc_std']:.1f}"
R = S["rq12"]
TF = S["tfidf"]

# =====================================================================================
P("Who Wrote It? Identifying LLMs from Their Responses", title)
P("CMPSC 448 Course Project &nbsp;·&nbsp; Fall 2026", small)
P("<b>Team leader:</b> Max Lin &nbsp;&nbsp; <b>Members:</b> Max Lin (individual project)", small)
P("<b>Code:</b> GitHub repository (link submitted on Canvas); every number in this report is produced "
  "by <font face='Courier'>run_all.sh</font> and stored in <font face='Courier'>results/</font>.", small)
story.append(Spacer(1, 4))

out_c, out_l = R["cnn_output"], R["lstm_output"]
SUMMARY_EXTRA = ""  # set in sec4_extra.py when FULL (written after its results are in)
P("<b>Summary.</b> We collected 9,636 responses written by 12 LLMs from 6 families (GPT, Claude, Llama, "
  "Qwen, Mistral, Google) to the <i>same</i> 803 user prompts, and trained a convolutional network (CNN) and "
  f"a bidirectional LSTM (RNN) to predict the family. From the response alone the CNN reaches "
  f"<b>{pct(out_c['acc'])}%</b> accuracy and the BiLSTM <b>{pct(out_l['acc'])}%</b> on unseen prompts "
  "(chance = 16.7%). Because every prompt was answered by every family, the prompt alone carries no "
  "information (input-only = chance), and adding it to the response does not help. A simulated "
  "data-collection bias shows how input-only classifiers can nevertheless look accurate on real-world "
  "datasets (RQ2)." + (" " + SUMMARY_EXTRA if FULL else ""))

# ------------------------------------------------------------------------------- 1
P("1. Problem Definition and Dataset Curation", h1)
P("1.1 Task", h2)
P("Given a text produced by a chat LLM (and optionally the user prompt that produced it), predict which "
  "<b>LLM family</b> wrote it. Each example is a triple <i>(LLM_name, LLM_input, LLM_output)</i>. We "
  "classify at the family level (6 classes) rather than the exact model because the assignment asks "
  "about LLM families, and because two versions from one vendor (e.g. GPT-4-Turbo and GPT-4o) are "
  "trained with similar data and post-training recipes, so the family is the natural unit of a "
  "\"fingerprint\". We also keep the exact model name, which we use for a per-model analysis.")

P("1.2 Data source and collection", h2)
P("Generating thousands of responses ourselves through paid APIs was not feasible, so we obtained "
  "publicly released outputs from <b>AlpacaEval</b> (Li et al., 2023; github.com/tatsu-lab/alpaca_eval, "
  "Apache-2.0 license). AlpacaEval is a leaderboard that stores, for every evaluated model, its raw answer "
  "to the same 805 instructions (<font face='Courier'>results/&lt;model&gt;/model_outputs.json</font>). "
  "The instructions come from five public instruction sets (Self-Instruct, OpenAssistant, Koala, "
  "Anthropic helpful-base, Vicuna) and cover advice, factual questions, creative writing, coding and math. "
  "The data contains no private or personal information; prompts were written by researchers or "
  "crowd-workers for public benchmarks.")
P("From the ~230 entries on the leaderboard we kept only <b>official vendor models</b> (no community "
  "fine-tunes such as Vicuna or Zephyr, whose \"family\" would be ambiguous) and chose two models per family "
  "so that every family is represented by more than one model version (Table 1). This matters: if each "
  "family were a single model, the classifier could memorize one model's quirks instead of a family "
  "fingerprint.")

rows = [["Family", "Models (AlpacaEval name)", "Responses", "Mean words", "Median words"]]
per_model = {}
for r in D["per_model"]:
    per_model.setdefault(r["llm_family"], []).append(r["llm_model"])
for r in D["per_family"]:
    rows.append([r["llm_family"], ", ".join(per_model[r["llm_family"]]), f"{r['n']:,}",
                 f"{r['mean_words']:.0f}", f"{r['median_words']:.0f}"])
rows.append(["Total", "12 models", "9,636", "", ""])
table(rows, "Dataset composition. Every prompt is answered by all 12 models, so classes are perfectly "
      "balanced.", col_widths=[0.75 * inch, 3.2 * inch, 0.8 * inch, 0.8 * inch, 0.85 * inch], left_cols=2)

P("1.3 Cleaning, splitting and annotation", h2)
P(f"(1) We removed empty responses (1 from gemini-pro) and then kept only prompts that <i>all 12</i> models "
  f"answered, leaving <b>{D['prompts']} prompts × 12 = 9,636 examples</b> (1,606 per family). "
  f"Prompts are short (mean {D['prompt_words_mean']:.0f} words, median {D['prompt_words_median']:.0f}); "
  "responses average 200–300 words. "
  "(2) We split <b>by prompt</b>, not by row: 70/15/15% of the prompts "
  f"({D['splits_prompts']['train']}/{D['splits_prompts']['val']}/{D['splits_prompts']['test']}) go to "
  f"train/validation/test together with all 12 of their answers "
  f"({D['splits_rows']['train']:,}/{D['splits_rows']['val']:,}/{D['splits_rows']['test']:,} rows). A random "
  "row split would put GPT's answer to a prompt in training and Claude's answer to the same prompt in "
  "test, letting the model use topic words; the prompt split guarantees that every test prompt is new. "
  "(3) We also tag every prompt with a task type using keyword rules "
  "(<font face='Courier'>data/build_dataset.py</font>): <b>technical</b> (code, math, data formats; "
  f"{D['task_types']['technical']} prompts), <b>writing</b> (stories, poems, emails, rewriting; "
  f"{D['task_types']['writing']}) and <b>general QA</b> (everything else: explanations, advice, "
  f"facts; {D['task_types']['qa']}). We manually checked a sample of the tags; they are imperfect "
  "(a few borderline prompts land in the wrong bucket) but adequate for a coarse cross-task test.")
P("<b>Why this design.</b> Because all families answer the same prompts, any difference the classifier "
  "picks up must come from <i>how</i> the model writes, not from <i>what</i> it was asked. This "
  "controlled design is what makes RQ2 interpretable (Section 3.2). The trade-off is that responses "
  "come from one benchmark and one time period (2023–2024 model versions), with each vendor's default "
  "API settings.")

# ------------------------------------------------------------------------------- 2
P("2. CNN / RNN Implementation and Training", h1)
P("2.1 Text representation", h2)
P("We wrote our own tokenizer instead of a standard word tokenizer because stylistic signals live in "
  "characters that most tokenizers throw away. The regular expression keeps words (case preserved, with "
  "contractions like <i>I'd</i> as one token), numbers, and <i>every</i> punctuation or markdown symbol as "
  "a separate token (<font face='Courier'>*</font>, <font face='Courier'>#</font>, "
  "<font face='Courier'>-</font>, <font face='Courier'>:</font>, ...). Line breaks become a special "
  "<font face='Courier'>&lt;nl&gt;</font> token, so a bulleted list looks like "
  "<font face='Courier'>&lt;nl&gt; - **</font> to the network. The vocabulary is built on the "
  f"training split only (tokens with frequency ≥ 2, ~{R['cnn_output']['vocab']:,} types); unseen tokens map "
  "to <font face='Courier'>&lt;unk&gt;</font>. Responses are truncated to 512 tokens "
  f"({100 * D['truncated_frac']:.0f}% of responses are longer) and prompts to 128 tokens. "
  "The three input settings of RQ2 are: <b>output only</b>; <b>input only</b>; and <b>input + output</b> = "
  "prompt tokens, a <font face='Courier'>&lt;sep&gt;</font> token, then response tokens.")

P("2.2 Models", h2)
P("<b>CNN</b> (Kim, 2014). A 128-d embedding layer (trained from scratch), then three parallel 1-D "
  "convolutions with kernel widths 3, 4 and 5 and 128 filters each (ReLU). Each feature map is max-pooled "
  "over time (padding positions are masked out), giving a 384-d vector that goes through dropout (0.5) and "
  "a linear layer to 6 logits. Each filter acts like a learned n-gram detector, e.g. "
  "<font face='Courier'>&lt;nl&gt; 1 . **</font> or <i>What a great question</i>, and max-pooling asks "
  "\"does this pattern appear anywhere in the response?\"")
P("<b>RNN: bidirectional LSTM.</b> The same 128-d embedding (dropout 0.3), a single-layer bidirectional "
  "LSTM with 128 hidden units per direction, and a sentence vector made of the masked max-pool and the "
  "masked mean-pool of all hidden states (512-d), followed by dropout and a linear layer. We first used the "
  "final hidden states with <font face='Courier'>pack_padded_sequence</font>, but on our CPU-only setup a "
  "batch with mixed lengths took ~30 s versus ~1 s unpacked, so we run the LSTM on padded batches and "
  "rely on pooling with masks instead of the last state. Batches are bucketed by length so there is very "
  "little padding.")

P("2.3 Training details", h2)
rows = [["Setting", "CNN", "BiLSTM"],
        ["Parameters", f"{R['cnn_output']['params']/1e6:.2f} M", f"{R['lstm_output']['params']/1e6:.2f} M"],
        ["Embedding", "128-d, random init, trained", "128-d, random init, trained"],
        ["Core layer", "conv k=3,4,5 × 128 filters", "BiLSTM 2 × 128 hidden"],
        ["Pooling", "masked max over time", "masked max + mean over time"],
        ["Dropout", "0.5 (before output)", "0.3 (embeddings and before output)"],
        ["Optimizer", "Adam, lr 1e-3, batch 64", "Adam, lr 1e-3, batch 64"],
        ["Gradient clipping", "norm 5", "norm 5"],
        ["Epochs / early stop", "max 15, patience 4 on val macro-F1", "max 15, patience 4 on val macro-F1"],
        ["Seeds (RQ1/RQ2)", "3", "2"],
        ["Avg. time per output-only run", f"{R['cnn_output']['time_min']:.0f} min (1 CPU thread)",
         f"{R['lstm_output']['time_min']:.0f} min (1 CPU thread)"]]
table(rows, "Model and training hyper-parameters.", col_widths=[1.7 * inch, 2.3 * inch, 2.4 * inch],
      left_cols=1)
P("Loss is cross-entropy. After every epoch we evaluate macro-F1 on the validation prompts and keep the "
  "best checkpoint; training stops when validation F1 has not improved for 4 epochs. Test prompts are "
  "used only once, for the final numbers. We report the mean ± standard deviation over random seeds "
  "(3 for the CNN, 2 for the slower LSTM). Everything ran on a 2-core CPU without a GPU, which is why we "
  "kept the models small and the number of LSTM seeds low. Hyper-parameters were set once from common "
  "defaults and a single trial run, not tuned on the test set. As a non-neural reference point (not "
  "required) we also trained a TF-IDF (word 1–2-grams + character 2–5-grams) logistic-regression model "
  "on the same splits.")
fig("figures/training_curves.png", "Training loss (grey) and validation macro-F1 per epoch, output-only "
    "setting, all seeds. Validation F1 is noisy because the validation set has only 120 prompts; early "
    "stopping keeps the best epoch.")

# ------------------------------------------------------------------------------- 3
story.append(CondPageBreak(3.5 * inch))
P("3. Results (RQ1 and RQ2)", h1)
rows = [["Input used", "CNN acc (%)", "CNN macro-F1", "BiLSTM acc (%)", "BiLSTM macro-F1",
         "TF-IDF+LR acc (reference)"]]
for mode, lbl in [("output", "Output only (RQ1)"), ("input", "Input only"), ("both", "Input + Output")]:
    c, l = R[f"cnn_{mode}"], R[f"lstm_{mode}"]
    rows.append([lbl, pm(c), f"{c['f1']:.3f}", pm(l), f"{l['f1']:.3f}", pct(TF[mode]["acc"])])
table(rows, "Test results on 1,452 responses to 121 unseen prompts (6 balanced classes, chance = 16.7%). "
      "Mean ± std over seeds (CNN: 3, BiLSTM: 2).",
      col_widths=[1.35 * inch, 0.95 * inch, 0.85 * inch, 1.0 * inch, 0.95 * inch, 1.15 * inch], left_cols=1)
fig("figures/rq12_accuracy.png", "Test accuracy by input setting. Error bars: std over seeds. The dashed "
    "line is chance (1/6).", width=5.0)

P("3.1 RQ1: Can we identify which LLM generated a response?", h2)
best = max(["cnn", "lstm"], key=lambda m: R[f"{m}_output"]["acc"])
P(f"<b>Yes, well above chance.</b> From the response alone, the CNN identifies the family of an answer to "
  f"a never-seen prompt {pct(out_c['acc'])}% of the time (macro-F1 {out_c['f1']:.3f}), and the BiLSTM "
  f"{pct(out_l['acc'])}% (macro-F1 {out_l['f1']:.3f}), versus 16.7% for guessing. The CNN is "
  f"{100*(out_c['acc']-out_l['acc']):.1f} points better and much more stable across seeds (std "
  f"{100*out_c['acc_std']:.1f} vs. {100*out_l['acc_std']:.1f}). We think the CNN fits this task better "
  "because the evidence is local and position-independent (a bold heading, an opening phrase, a list "
  "marker), which is exactly what max-pooled n-gram filters detect, while the LSTM has to carry such "
  "evidence through hundreds of steps and is harder to optimize from scratch on ~6.7k examples. The "
  f"TF-IDF reference reaches {pct(TF['output']['acc'])}%, so a good part of the signal is available as "
  "surface n-grams; the neural models do not find much beyond it with this amount of data.")
fig("figures/rq1_confusion.png", "Row-normalized confusion matrices (output only, summed over seeds). "
    "Diagonal = recall per family.", width=5.8)
cm = out_c["confusion_norm"]
rec = {f: cm[i][i] for i, f in enumerate(FAM)}
order = sorted(FAM, key=lambda f: -rec[f])
pmc = S["per_model_acc"]["cnn"]
P(f"<b>Not all families are equally recognizable.</b> CNN recall ranges from {pct(rec[order[0]])}% "
  f"({order[0]}) and {pct(rec[order[1]])}% ({order[1]}) down to {pct(rec[order[-2]])}% ({order[-2]}) and "
  f"{pct(rec[order[-1]])}% ({order[-1]}) (Figure {fig_no[0]}). Google's models are distinctive because they "
  "often skip any introduction and start directly with a bold list item or give a one-line answer; "
  "Claude 3 almost never uses <font face='Courier'>**bold**</font> markdown in this data. Qwen and "
  "Mistral are the hardest: their answers look like a generic \"average\" assistant answer (an intro "
  "sentence, a numbered list, a short conclusion). They are most often confused with each other and "
  "with Claude, Mistral also with Llama and Qwen also with GPT. One plausible reason is that open models are often fine-tuned on "
  "conversations generated by, or modeled after, GPT-style assistants, which would blur their style.")
rows = [["Family", "Model", "CNN acc on its responses (%)"]]
for f in FAM:
    for m in per_model[f]:
        rows.append([f, m, pct(pmc[m])])
table(rows, "Per-model accuracy (CNN, output only, mean over seeds). The two members of a family are "
      "often not equally recognizable.", col_widths=[0.9 * inch, 2.4 * inch, 1.8 * inch], left_cols=2)
P(f"Within families the gap can be large: e.g. claude-3.5-sonnet ({pct(pmc['claude-3-5-sonnet-20240620'])}%) "
  f"vs. claude-3-opus ({pct(pmc['claude-3-opus-20240229'])}%), gpt-4o ({pct(pmc['gpt-4o-2024-05-13'])}%) vs. "
  f"gpt-4-turbo ({pct(pmc['gpt4_1106_preview'])}%), and Llama-3 ({pct(pmc['Meta-Llama-3-70B-Instruct'])}%) "
  f"vs. Llama-2 ({pct(pmc['llama-2-70b-chat-hf'])}%). So a \"family fingerprint\" is partly the fingerprint "
  "of its most distinctive member, and successive versions of one vendor's model can drift apart in style.")

P("3.2 RQ2: Does the user's prompt help identify the LLM?", h2)
inp_c, inp_l, both_c, both_l = R["cnn_input"], R["lstm_input"], R["cnn_both"], R["lstm_both"]
P(f"<b>(i) Can the LLM be predicted from the prompt alone? No, not in our data.</b> The input-only CNN "
  f"scores {pct(inp_c['acc'])}% and the BiLSTM {pct(inp_l['acc'])}%, i.e. chance. This is expected by "
  "construction: every prompt appears 12 times in the dataset, twice with each label, so the prompt "
  "carries zero information about the label. The networks correctly learn to output roughly the same "
  "distribution for every prompt. We consider this the most important sanity check of the dataset: if "
  "input-only had been above chance, our split or our data would have been leaking.")
P(f"<b>(ii) Does adding the input improve classification? Not meaningfully.</b> Input + output gives "
  f"{pct(both_c['acc'])}% (CNN) and {pct(both_l['acc'])}% (BiLSTM), compared with {pct(out_c['acc'])}% and "
  f"{pct(out_l['acc'])}% for output only; TF-IDF also does not improve ({pct(TF['both']['acc'])}% vs. "
  f"{pct(TF['output']['acc'])}%). The changes are small and go in both directions, i.e. within run-to-run noise. In principle the prompt could "
  "help as <i>context</i> (e.g. knowing a one-line answer was given to a request for a long essay), but "
  "with our data size the models cannot learn such interactions, and the extra prompt tokens mostly add "
  "noise.")
B = {round(r["p_bias"], 3): r for r in S["rq2_bias"]}
b0, b8 = B[round(1 / 6, 3)], B[0.8]
P("<b>(iii) When can input-only classification work surprisingly well?</b> Only when the dataset itself "
  "links prompts to models. That happens easily in practice: if one person collects GPT answers to coding "
  "questions and another collects Llama answers to chit-chat, a classifier can learn \"coding prompt → "
  "GPT\" without ever looking at the writing. To show this we built biased versions of our data: each "
  "prompt keeps a <i>single</i> response, and with probability <i>p</i> that response comes from the family "
  "we linked to the prompt's source collection (Self-Instruct→GPT, OpenAssistant→Claude, Koala→Llama, "
  "helpful-base→Qwen, Vicuna→Mistral), otherwise from a random other family. We trained the input-only "
  "CNN on each version (3 seeds).")
fig("figures/rq2_bias.png", "Input-only CNN accuracy when the data collection links prompt sources to "
    "families with probability p (p = 1/6: no link). Dashed: always predicting the most frequent class.",
    width=3.8)
P(f"With no link (<i>p</i> = 1/6) the prompt-only model is at {pct(b0['acc'])}%, near chance. As the "
  f"collection bias grows, the same model rises to {pct(b8['acc'])}% at <i>p</i> = 0.8 (majority-class "
  f"baseline {pct(b8['maj'])}%) — without seeing a single word written by an LLM. The prompt sources differ "
  "in style (Self-Instruct prompts are templated tasks, Koala prompts are casual chat, etc.), so the "
  "network learns the source, and through the source the label. <b>So a high input-only score is a "
  "warning sign of dataset bias, not evidence of an LLM fingerprint.</b> This is why we used a dataset "
  "where every family answers every prompt.")


if FULL:
    exec(open("report/sec4_extra.py").read())
exec(open("report/sec5_lessons.py").read())


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, f"CMPSC 448 · Who Wrote It? · page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate("report/report.pdf", pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                  topMargin=0.8 * inch, bottomMargin=0.8 * inch, title="Who Wrote It? Identifying LLMs "
                  "from Their Responses", author="Max Lin").build(story, onFirstPage=footer,
                                                                   onLaterPages=footer)
print("pages built; figures:", fig_no[0], "tables:", tab_no[0])
