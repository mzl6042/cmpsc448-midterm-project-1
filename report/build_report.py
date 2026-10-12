"""Make the report PDF. Run this from the main project folder."""

import json
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Image, Table, TableStyle, Spacer, KeepTogether, CondPageBreak, SimpleDocTemplate

# Load the results made by the other scripts
S = json.load(open("results/summary.json"))
D = json.load(open("results/dataset_stats.json"))
R = S["rq12"]
TF = S["tfidf"]
FULL = os.environ.get("FULL") == "1"
FAM = ["Claude", "GPT", "Google", "Llama", "Mistral", "Qwen"]

styles = getSampleStyleSheet()
body = ParagraphStyle("regular", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5, leading=13.6, spaceAfter=5)
small = ParagraphStyle("smalltext", parent=body, fontSize=8.8, alignment=TA_CENTER, spaceAfter=8)
h1 = ParagraphStyle("mainheading", parent=styles["Heading1"], fontSize=13.5, spaceBefore=10, spaceAfter=5)
h2 = ParagraphStyle("subheading", parent=styles["Heading2"], fontSize=11, spaceBefore=7, spaceAfter=3)
title = ParagraphStyle("reporttitle", parent=styles["Title"], fontSize=17)
cell = ParagraphStyle("tabletext", parent=body, fontSize=8.6, leading=10.4, alignment=TA_CENTER, spaceAfter=0)
cell_l = ParagraphStyle("lefttabletext", parent=cell, alignment=0)

story = []
fig_no = [0]
tab_no = [0]

def P(text, style=None):
    if style is None:
        style = body
    story.append(Paragraph(text, style))

# these two are just for adding the pictures and tables
# to the report

def fig(path, caption, width=6.3):
    fig_no[0] += 1
    w, h = ImageReader(path).getSize()
    pic = Image(path, width=width * inch, height=width * inch * h / w)
    story.append(KeepTogether([pic, Paragraph("Figure " + str(fig_no[0]) + ". " + caption, small)]))
    return fig_no[0]

def table(rows, caption, col_widths=None, header_rows=1, left_cols=0, font=None):
    tab_no[0] += 1
    new_rows = []
    for i in range(len(rows)):
        new_row = []
        for j in range(len(rows[i])):
            value = rows[i][j]
            if isinstance(value, str):
                if i < header_rows:
                    value = "<b>" + value + "</b>"
                value = Paragraph(value, font or (cell_l if j < left_cols else cell))
            new_row.append(value)
        new_rows.append(new_row)
    t = Table(new_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)
    ]))
    story.append(Paragraph("Table " + str(tab_no[0]) + ". " + caption, small))
    story.append(t)
    story.append(Spacer(1, 8))
    return tab_no[0]

def pct(value):
    return f"{100 * value:.1f}"

def pm(d):
    return f"{100 * d['acc']:.1f} ± {100 * d['acc_std']:.1f}"

P("Who Wrote It? Identifying LLMs from Their Responses", title)
P("CMPSC 448 Course Project &nbsp;·&nbsp; Fall 2026", small)
P("<b>Team leader:</b> Max Lin &nbsp;&nbsp; <b>Members:</b> Max Lin (individual project)", small)
P("<b>Code:</b> GitHub repository (submitted on Canvas). Numbers come from "
  "the scripts and result files in the repo.", small)
story.append(Spacer(1, 4))
out_c, out_l = R["cnn_output"], R["lstm_output"]
SUMMARY_EXTRA = ""  # gets filled in if extra sections are enabled
P("<b>Summary.</b> This project uses 9,636 answers from 12 language models across six "
  "families. They all answered the same 803 prompts. We trained a CNN and a bidirectional LSTM "
  "to see if their responses have recognizable patterns. The CNN got "
  f"{pct(out_c['acc'])}% accuracy and the LSTM got {pct(out_l['acc'])}% on test prompts "
  "(random guessing would be about 16.7%). We also tested using only the prompt and using "
  "the prompt together with the answer. The prompt did not really help. Finally, we tried a "
  "biased version of the dataset to see how that could change the scores." 
  + (" " + SUMMARY_EXTRA if FULL else ""))