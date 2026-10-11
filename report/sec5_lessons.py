# appended into build_report.py's namespace via exec (keeps the long text out of the main file)
P("5. Lessons and Experience", h1)
P("<b>The split matters more than the model.</b> A random row split is the default in most tutorials. "
  "With 12 answers per prompt, it would have put most test prompts into training under other labels "
  "and let topic words leak. Splitting by prompt was a small code change but it decides what the reported number "
  "means. The input-only result (exactly chance) is the sanity check that the split works.")
P("<b>Dataset design answers research questions before any training.</b> Choosing a dataset where every "
  "family answers the same prompts made RQ2 almost trivially answerable for our data, and the "
  "bias simulation showed us how easily a less controlled dataset produces an impressive but meaningless "
  "input-only score. When we read about \"AI text detectors\" with very high accuracy, the first question "
  "we now ask is how the data were collected.")
P("<b>Tokenization is a modeling decision.</b> A standard tokenizer would have deleted most of the "
  "signal: markdown symbols, list markers and line breaks are exactly where the families differ "
  "most visibly (e.g. Claude 3 almost never writes **bold**, Gemini often starts with a bulleted list)." +
  (" The ablations in Section 4.2 confirm this." if FULL else ""))
P("<b>A simple baseline keeps you honest.</b> The TF-IDF + logistic regression model trained in about a "
  "minute and was at least as accurate as both neural networks. With only ~6,700 training examples and "
  "no pre-trained embeddings, the CNN and LSTM must learn word meanings from scratch, while n-gram "
  "counts capture the surface habits directly. A pre-trained transformer would likely do better; we "
  "stayed with the required architectures and a CPU budget.")
P("<b>CNN vs. RNN in practice.</b> The CNN was faster (≈57 s vs. ≈84 s per epoch on one CPU thread), more "
  "stable across seeds and more accurate. Our LSTM was also slowed by an engineering problem: packed sequences were "
  "~30× slower on our CPU build, and we had to switch to padded, length-bucketed batches with masked "
  "pooling. Profiling one batch before launching hours of training would have saved us a failed run.")
P("<b>Limitations.</b> (i) All responses come from one benchmark with default settings and no system "
  "prompts; real users change temperature, system prompts and formatting instructions, which would "
  "weaken formatting cues. (ii) Two models per family is the minimum to talk about a \"family\"; newer "
  "versions may write differently (e.g. Claude 3.5 Sonnet already differs from Claude 3 Opus in our "
  "per-model results). (iii) Because of the CPU budget we ran 3 seeds for the CNN and 2 for the "
  "BiLSTM, and did no hyper-parameter search." + (" The task-type tags for RQ3 come from keyword rules, "
  "the technical test set is small (17 prompts), and the RQ3/RQ4 ablation runs use one seed, so those "
  "numbers have wider uncertainty." if FULL else ""))

P("References", h1)
for ref in [
    "Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.",
    "Hochreiter, S. &amp; Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.",
    "Li, X., Zhang, T., Dubois, Y., et al. (2023). AlpacaEval: An Automatic Evaluator of "
    "Instruction-following Models. github.com/tatsu-lab/alpaca_eval.",
    "Monroe, B., Colaresi, M., &amp; Quinn, K. (2008). Fightin' Words: Lexical Feature Selection and "
    "Evaluation for Identifying the Content of Political Conflict. Political Analysis.",
    "Covington, M. &amp; McFall, J. (2010). Cutting the Gordian Knot: The Moving-Average Type-Token Ratio "
    "(MATTR). Journal of Quantitative Linguistics.",
]:
    P(ref, ParagraphStyle("ref", parent=body, fontSize=9, leading=11, leftIndent=12, firstLineIndent=-12))
