# RQ3 + RQ4 (extra credit); exec'd by build_report.py when FULL
# ------------------------------------------------------------------------------- 4
P("4. In-depth Analyses (extra credit: RQ3 and RQ4)", h1)
P("4.1 RQ3: Do LLM fingerprints generalize across task types?", h2)
P("<b>Protocol.</b> For each task type T (technical, writing, QA) we compare two output-only CNNs evaluated "
  "on <i>the same</i> test responses of type T: a <b>cross-task</b> model trained only on prompts of the "
  "other two types, and an <b>in-domain</b> model trained on a random sample of training prompts of the "
  "<i>same size</i> drawn from all types (so it sees some prompts of type T). Using the same test set and "
  "the same amount of training data isolates the effect of never having seen the task type. Because of "
  "our compute budget this experiment uses one seed per condition.")
R3 = {(r["held_out"], r["cond"]): r for r in S["rq3"]}
rows = [["Held-out task type", "Test responses", "Train responses", "In-domain acc (%)",
         "Cross-task acc (%)", "Change (points)"]]
for T, lbl in [("technical", "Technical (code/math)"), ("writing", "Writing"), ("qa", "General QA")]:
    i, c = R3[(T, "in_domain")], R3[(T, "cross")]
    rows.append([lbl, f"{int(c['n_test'])}", f"{int(c['n_train']):,}", pct(i["acc"]), pct(c["acc"]),
                 f"{100 * (c['acc'] - i['acc']):+.1f}"])
table(rows, "RQ3: output-only CNN, same test responses per row; the in-domain model gets a random "
      "training sample of the same size as the cross-task model.",
      col_widths=[1.5 * inch, 0.85 * inch, 0.85 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch], left_cols=1)
fig("figures/rq3_cross_task.png", "RQ3: accuracy on each task type when the training data does / does "
    "not contain that task type (one seed).", width=4.2)
RQ3_TEXT = None  # filled below after looking at the numbers

P("4.2 RQ4: What characteristics distinguish different LLMs?", h2)
P("We used two complementary approaches: (a) measure interpretable features of each response and see "
  "how well they alone predict the family, and (b) remove a type of signal from the text and retrain "
  "the CNN to see how much accuracy it loses.")
P("<b>(a) Interpretable features.</b> For every response we computed 27 features in four groups "
  "(<font face='Courier'>src/rq4_features.py</font>): <i>length</i> (characters, words, lines, paragraphs); "
  "<i>lexical</i> (average word and sentence length, lexical diversity measured with the moving-average "
  "type-token ratio MATTR, which unlike plain TTR does not depend on text length, use of \"I\", \"you\" and "
  "contractions); <i>formatting</i> (markdown headings, bullet and numbered-list lines, bold spans, code "
  "blocks, and the rates of colons, exclamation and question marks, dashes, parentheses, emoji); and "
  "<i>discourse habits</i> (opens with \"Sure/Certainly\", opens with \"Here\", closes with a summary or an "
  "offer of more help, \"it's important to note\"). Figure "
  f"{fig_no[0] + 1} shows the clearest differences.")
fig("figures/rq4_feature_means.png", "Mean feature values per family (all 9,636 responses).", width=6.4)
FR = S["rq4_features"]
G = FR["group_results"]
rows = [["Feature group", "# features", "Random-forest test acc (%)"]]
for g, lbl in [("length", "Length"), ("lexical", "Lexical"), ("formatting", "Formatting / structure"),
               ("discourse", "Discourse habits"), ("all", "All features")]:
    rows.append([lbl, str(G[g]["n_features"]), f"{pct(G[g]['acc'])} ± {pct(G[g]['acc_std'])}"])
rows.append(["(CNN on full text, for comparison)", "–", pm(out_c)])
table(rows, "Predicting the family from hand-crafted features only (random forest, 400 trees, trained on "
      "train+val, 3 seeds).", col_widths=[2.4 * inch, 0.8 * inch, 1.8 * inch], left_cols=1)
imp = FR["perm_importance"][:8]
P(f"Formatting is by far the most informative group ({pct(G['formatting']['acc'])}% with 11 features), then "
  f"length ({pct(G['length']['acc'])}%); lexical statistics and discourse habits are weak on their own. "
  f"All 27 features together reach {pct(G['all']['acc'])}%, close to the CNN's {pct(out_c['acc'])}% — so a "
  "handful of countable properties explains most of what the networks can do. The most important "
  "features by permutation importance are " +
  ", ".join(f"<i>{n.replace('_', ' ')}</i>" for n, _ in imp[:6]) + ". Concretely: GPT-4 writes the "
  "longest answers and is the only family that regularly uses markdown headings; Gemini/Gemma use many "
  "bullet points and bold text and often start immediately with a list; Claude 3 uses numbered lists "
  "but essentially never bold; Llama-3 opens enthusiastically (\"What a great question!\") and uses "
  "the most exclamation marks; Mistral produces the fewest line breaks (more paragraph-style answers).")
W = FR["log_odds_words"]
rows = [["Family", "Most over-used words (log-odds vs. the other 5 families)"]]
for f in FAM:
    rows.append([f, ", ".join(W[f][:10])])
table(rows, "Distinctive words (weighted log-odds with an informative Dirichlet prior, Monroe et al. 2008, "
      "counted once per response and restricted to words used in ≥ 25 different prompts so that topic "
      "words from a single prompt do not dominate). Fragments like \"s\", \"don\", \"re\" for GPT come from "
      "curly apostrophes (don’t) that our tokenizer splits.",
      col_widths=[0.8 * inch, 5.3 * inch], left_cols=2)
import csv
hab = {r["llm_model"]: r for r in csv.DictReader(open("results/rq4/habits_per_model.csv"))}
P("The word lists show recognizable \"personalities\": Llama is enthusiastic (<i>fascinating, hope, "
  "delicious, worth noting</i>), Mistral is eager to help (<i>I'd be happy to, sure, absolutely</i>), "
  "GPT hedges (<i>might, sometimes, possibly, often</i>), Claude is organized and precise (<i>key, several, "
  "specific, e.g.</i>), Qwen likes connectives (<i>moreover, although, due to, crucial</i>), and Google's "
  "models use answer-sheet vocabulary (<i>answer, additional tips, notes, therefore, conclusion</i>). We "
  "also found habits that are almost unique to one model (percent of responses): gpt-4o writes "
  f"typographic apostrophes (’) in {hab['gpt-4o-2024-05-13']['curly_apostrophe']}% of answers vs. ≤ 0.5% "
  "for every non-OpenAI model except Qwen2 "
  f"({hab['Qwen2-72B-Instruct']['curly_apostrophe']}%); gemini-pro starts "
  f"{hab['gemini-pro']['starts_with_list_or_bold']}% of its answers directly with a list item or bold text "
  f"(Claude 3.5 Sonnet: {hab['claude-3-5-sonnet-20240620']['starts_with_list_or_bold']}%); and "
  f"gemini-pro gives answers shorter than 20 words {hab['gemini-pro']['very_short_<20_words']}% of the "
  "time (e.g. answering \"When were smart phones first made?\" with just \"1992\").")

P("<b>(b) Ablations: does removing a signal hurt the CNN?</b> We retrained the output-only CNN on "
  "modified versions of every response (train, validation and test modified the same way): "
  "<i>markdown & lines removed</i> (all <font face='Courier'>#*_`&gt;|</font> symbols, list markers and "
  "line breaks deleted, leaving one plain paragraph); <i>lowercase words only</i> (additionally all "
  "punctuation and capitalization removed); <i>words shuffled</i> (the response's tokens in random order, "
  "keeping the bag of words but destroying all order and phrasing); and <i>first 50 tokens only</i> (the "
  "opening of the answer, which also removes length). One seed per ablation.")
A = {r["ablation"]: r for r in S["rq4_ablation"]}
rows = [["Text given to the CNN", "Test acc (%)", "Change vs. original (points)"]]
for ab, lbl in [("none", "Original response (RQ1)"), ("no_format", "Markdown & line breaks removed"),
                ("lower_nopunct", "Lowercase words only (no punctuation/format)"),
                ("shuffle", "Words shuffled (bag of words)"), ("first50", "First 50 tokens only")]:
    if ab in A:
        rows.append([lbl, pct(A[ab]["acc"]), "–" if ab == "none" else f"{100 * (A[ab]['acc'] - A['none']['acc']):+.1f}"])
table(rows, "RQ4 ablations (output-only CNN).", col_widths=[3.0 * inch, 1.0 * inch, 1.8 * inch], left_cols=1)
fig("figures/rq4_ablation.png", "CNN accuracy after removing different kinds of signal.", width=4.6)
