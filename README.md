# FinSignal-LLM

If an LLM reads a news headline about a company, can it hint at whether the stock will beat the market the next day?

Example: the headline says *"Apple beats earnings estimates and raises guidance"*. The LLM calls it positive. Did Apple actually do better than the market on the next trading day? I do this for thousands of headlines and compare the LLM against older tools (a word list, a small classifier and FinBERT).


---

## The question 

> Do LLM-extracted signals from financial headlines predict **next-day abnormal returns** better than a lexicon or FinBERT?

"Abnormal return" means the stock's return minus SPY's (the market's) return. If Apple is up 1.2% and the market is up 0.3%, the abnormal return is +0.9%. This removes the "whole market went up" effect.

Two parts:

- **Part A: does the LLM read headlines correctly?** Checked on Financial PhraseBank, which has human-labelled sentiment.
- **Part B: do its outputs predict returns?** Checked with a time-based split and a simple long-short backtest.

---

## How it works

```
news ──► data_prep ──► dataset.csv ──┬─► baselines (lexicon, TF-IDF+LR, FinBERT) ─┐
(date, ticker, headline)  align + label │                                          ├─► evaluate ──► IC, accuracy,
                                       └─► llm_extract (JSON, cached) ────────────┘   backtest, bootstrap CIs
```

| Step | What it does | File |
|---|---|---|
| Alignment and labels | Maps each headline to the first trading day it could really be traded, then attaches the next-day abnormal return vs SPY | `src/data_prep.py` |
| Baselines | Loughran-McDonald word list, TF-IDF + logistic regression, FinBERT (`P(pos) - P(neg)`) | `src/baselines.py` |
| LLM extraction | Headline + ticker in, JSON out: `sentiment`, `score` (-1 to 1), `event_type`, `confidence`. Answers are validated and cached on disk | `src/llm_extract.py` |
| Evaluation | IC, direction accuracy, long-short quintile backtest with costs, day-level bootstrap intervals | `src/evaluate.py` |
| Secrets | Reads the API key from a git-ignored `.env` file | `src/config.py` |

### One headline, start to finish

The numbers in this table are made up to show the flow. They are not results.

| Step | Value |
|---|---|
| Headline | "Apple beats earnings estimates and raises guidance" |
| Published | Tuesday 6:30 pm Eastern (after the market closed) |
| Entry date | **Wednesday** close (first close you could trade at) |
| Next close | Thursday close |
| AAPL return Wed to Thu | +1.2% |
| SPY return Wed to Thu | +0.3% |
| Abnormal return | **+0.9%**, so `label_up = 1` |
| LLM output | `{"sentiment": "positive", "score": 0.6, "event_type": "earnings", "confidence": 0.9}` |

The question for the whole project: across thousands of rows like this, does `score` line up with the abnormal return better for the LLM than for the baselines?

### Timestamps (the part that is easy to get wrong)

Results that look too good usually come from mixing up *when the news was known* with *when you could trade*. So I'm conservative:

- News during market hours is traded at **that day's close**. News after 4pm Eastern, or on a weekend, moves to the next business day.
- The target is the return from that close to the **next** close, minus SPY's.
- **96% of my headlines only have a date, no time.** I can't tell if they came out before or after the close, so they all enter at the *next* business day's close. That avoids leaking the answer, but it also means I'm testing the **drift after the news**, not the immediate reaction. A null result here does not prove the LLM misreads news. Most of the reaction is already in the price.

### The LLM step

The model is `openai/gpt-oss-20b` on Groq's free tier, zero-shot, temperature 0, JSON output. I send 10 headlines per request because the free tier allows only 1,000 requests and 200,000 tokens a day (I measured these limits). Even so, scoring 3,000 headlines took about a day and a half of waiting out the daily cap. Every answer is checked (allowed labels, score in range) before it is cached, and failures are retried instead of being turned into a fake "neutral". In the final run, 0 of 3,000 answers failed.

I checked that batching doesn't change the answers much: on 200 PhraseBank sentences, single-item and batched calls gave the same label 90% of the time, and the accuracy difference (90.5% vs 93.5%) is within noise for 200 sentences. The model sees no date, only the ticker and the headline.

---

## Data

| What | Where to get it | Put it here |
|---|---|---|
| Headlines (Benzinga, about 1.4M rows, 2009-2020) | [Kaggle: Massive Stock News Analysis DB for NLP Backtests](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) (original), or the [Hugging Face mirror](https://huggingface.co/datasets/khaihernlow/massive-stock-news-analysis-db-for-nlpbacktests/tree/main/data), file `raw_analyst_ratings.jsonl` (about 440 MB) | `data/raw/raw_analyst_ratings.jsonl` |
| Sentiment labels for Part A | [Financial PhraseBank on Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank) (`data/FinancialPhraseBank-v1.0.zip`, Malo et al.). I only use `Sentences_AllAgree.txt` | `data/raw/financial_phrasebank/` |
| Word list for the lexicon baseline | Loughran-McDonald, [official page](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) | `data/raw/lm_pos_neg.csv` (see the note below) |
| Prices | `yfinance` (daily adjusted closes), SPY as the benchmark | downloaded inside notebook 01 |

About the word list: the code expects a small CSV with two columns, `word` (upper case) and `polarity` (+1 positive, -1 negative). I built mine from the Loughran-McDonald list that ships inside the [`pysentiment2`](https://pypi.org/project/pysentiment2/) package (I took only its data file, no code), and notebook 02 checks that it has the 354 positive and 2,355 negative words I expect. You can also build the same file from the official master dictionary.

Each source has its own license, so check it before reusing the data. The Hugging Face file name and size are the ones I used (440 MB, 439,674,269 bytes), and all the links above were checked and working when I added them.

What I actually used: 1.41M raw rows, cut to the window 2018-01-01 to 2020-06-10, minus duplicates and generic roundups ("Stocks That Hit 52-Week Highs On Friday" and similar), leaves 252,529. From that I sampled **30,000 headlines on 100 tickers over 636 trading days**. The tickers are the ones with the most headlines that still have Yahoo prices today: mostly large-cap stocks, but 15 of the 100 are ETFs (QQQ, IWM, sector funds and so on) and both GOOG and GOOGL are in.

Split in time, with a 2-business-day gap so a return window can't overlap: **22,374 rows to train, 7,542 to test** (test runs 2020-01-14 to 2020-06-10, 107 days). The test period includes the March 2020 crash.

The raw data is not in the repo (it's large). `data/raw/`, `data/processed/`, `data/llm_cache/` and `.env` are git-ignored.

---

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

Only the LLM step needs an API key (Groq). Secrets live in one git-ignored file, `.env`, in the project root:

```powershell
copy .env.example .env
```

Then open `.env` and put your key after the `=`:

```
GROQ_API_KEY=your-key-here
```

Never commit `.env`. To use another Groq model, set `GROQ_MODEL`. Groq retires models over time, so notebook 03 checks the model is still served before it runs.

Then:

```powershell
jupyter lab
```

### Notebooks

Run them in order. Each one fixes the random seed (42) and says what it checks before it does it.

| Notebook | What it does |
|---|---|
| `01_data_eda` | Builds `dataset.csv`; checks timestamps, duplicates, class balance, alignment (with hand-checked examples) |
| `02_baselines` | Lexicon, TF-IDF + LR and FinBERT, scored and compared on the same split |
| `03_llm_extraction` | Prompt, validation tests, Part A on PhraseBank, batch vs single, scoring the test headlines |
| `04_signal_backtest` | Checks the backtest on fake signals first, then the long-short results with costs |
| `05_analysis_leakage` | Event types, ablations, sub-periods, hiding company names, error review, robustness |

---

## How I judge results

Decided before looking at any results, so I can't tune my way to a good number.

- **Time-based split only.** Train on the past, test on the future. No shuffling.
- **Rank IC:** rank correlation between the score and the next-day abnormal return, computed within each day, then averaged. Within-day matters because stocks move together; comparing stocks on the same day removes the market-wide part. In this field even 0.02 to 0.05 would count as meaningful.
- **Direction accuracy** against the majority-class rate, not against 50%: in this dataset "stock beats SPY" happens 49.16% of the time, so always guessing "down" scores 50.84%.
- **Long-short backtest:** each day, buy the top 20% of stocks by score and short the bottom 20%, equal weights. A stock's several headlines are averaged into one score per day first. Costs: 10 bps per trade with the whole book turning over daily, which is 40 bps per day. That is a worst case with no borrow fees or market impact.
- **Confidence intervals** come from resampling whole days (a cluster bootstrap), because headlines from the same day are not independent. If an interval includes 0, I don't call it a result.
- **Check the checker.** Notebook 04 first runs the backtest on fake signals where I know the answer (pure noise, a planted weak signal, a perfect oracle) to make sure it finds a signal when one exists and nothing when it doesn't.

---

## Results

All of this is on the test period, with all four signals compared on the **same 3,000 test headlines** (a random 40% sample, because of the free-tier limits).

### Part A: can the LLM read a headline?

200 PhraseBank sentences (all annotators agree):

| | Accuracy | Macro-F1 |
|---|---|---|
| LLM (`gpt-oss-20b`, zero-shot) | 93.5% | 0.936 |
| FinBERT | 97.5% | 0.960 |
| Always guess the most common class | 61.5% | 0.254 |

The LLM reads sentiment well, and a bit worse than FinBERT. FinBERT has an edge here because it was fine-tuned on this kind of text, so this is not a clean comparison. 200 sentences is also a small sample.

### Part B: does it predict returns?

Long-short, gross of costs, mean per day with a 95% day-bootstrap interval:

| Signal | Days | Gross, bps/day | 95% interval | Net of 10 bps/trade |
|---|---|---|---|---|
| Lexicon | 77 | +16.1 | [-61, +92] | -24.0 |
| TF-IDF + LR | 102 | -28.6 | [-88, +30] | -68.6 |
| FinBERT | 102 | +4.7 | [-51, +56] | -35.3 |
| LLM | 96 | +43.5 | [-31, +116] | +3.5 |

Rank view (mean daily IC): LLM +0.023 [-0.026, +0.076], FinBERT +0.014. LLM direction accuracy is 48.95% [46.2, 51.9], below the 50.84% you get by always guessing "down".

**How I read it:**

- **Every interval contains 0.** The LLM has the largest point estimate, but its interval runs from -31 to +116 bps per day. After costs it is at break-even, not profitable.
- **The numbers move a lot with the sample.** The same lexicon scored +56 bps on all 7,542 test rows and +16 bps on the 3,000-row subset. That swing is about as big as the LLM's whole advantage.
- **This is a statement about power as much as about the model.** With about 100 test days, I can't tell an edge smaller than roughly 75 bps per day from zero. A real but modest edge would not show up here.

### What I looked at to understand it (notebook 05)

- **By event type:** earnings (+0.11), guidance (+0.12) and M&A (+0.12) look best, which is the pattern you'd hope for, but those groups are small and every interval contains 0. With 8 groups, one of them looking good by chance is normal. A lead, not a finding.
- **Which part of the output matters:** sign only, score, score x confidence, confident rows only all give about the same IC (0.02 to 0.04). The model's confidence doesn't help.
- **Sub-periods:** March 2020 is the best period for both the LLM and FinBERT (about +0.09 each, intervals still include 0), which looks more like a crash-regime effect than something about LLMs.
- **Leakage check (company names hidden):** the model may remember how 2020 turned out, so I replaced company names and tickers with "the company" on 300 headlines and scored them again. 87% kept the same sentiment label and the IC change was -0.042 [-0.093, +0.005] (the masked version was, if anything, slightly better, which is not what name leakage would look like). So I found no evidence the name drives the signal, but 300 rows can only detect a change of about 0.12, and hiding names does not hide events (a Fed announcement is still the Fed announcement in March 2020).
- **Robustness:** about 45% of rows have a headline the source tags to two or more tickers. I checked whether using only single-ticker headlines gives a cleaner signal. It didn't (LLM IC +0.015 vs +0.051 on the rest, both within noise), so the main analysis keeps all rows.
- **Errors:** 51% of the model's non-neutral calls had the wrong direction. Some of the most confident misses are data problems, for example a headline about another company tagged to the wrong ticker, or news whose price move had already happened before my next-day entry.

---

## Limitations

- **Small test.** 106 test days, 3,000 headlines, one model, one prompt, one market regime that includes a crash. Few-shot prompting is implemented but I did not run it on the test set, so there is no few-shot result.
- **Date-only news.** I'm testing drift after the news, not the instant reaction. With exact timestamps the answer could differ.
- **The model may remember 2020.** Hiding names reduces this but cannot remove it. A clean test needs news from after the model's training cutoff, which I don't have.
- **Survivorship bias.** 23 of my 150 candidate tickers had no price history on Yahoo (acquired, delisted or renamed), so the universe only has companies that survived. It is mostly large caps, but 15 of the 100 tickers are ETFs (their headlines are about markets or sectors, not one company) and GOOG and GOOGL are the same company counted twice. I did not test whether dropping them changes anything.
- **Prices are split-adjusted** and I take closing prices as fills, so real execution would be worse.
- **Costs are a flat worst case.** No borrow fees, no market impact, no bid-ask model beyond 10 bps.
- **Few names per day.** About 4 to 5 stocks per leg, so each daily spread is noisy.
- **Not exactly reproducible from scratch.** Notebook 01 downloads prices from Yahoo, and adjusted prices change over time. The saved `dataset.csv` fixes the data, and notebooks 02 to 05 refer to rows by their position in it (`row_id`). If you rebuild the dataset, re-run notebooks 02 to 05 too. The LLM cache is keyed by headline text, so those answers still match.
- **Not done yet:** `requirements.txt` has no pinned versions.

If I continue, the next things I'd try are more test days, a second model to see if the null is specific to this one, and news from after the model's cutoff.

---

## Project layout

```
FinSignal-LLM/
├── data/
│   ├── raw/          # input files (git-ignored)
│   ├── processed/    # dataset and score files (git-ignored)
│   └── llm_cache/    # cached LLM answers (git-ignored)
├── notebooks/        # 01 to 05, run in order
├── src/              # config, data_prep, baselines, llm_extract, evaluate
├── docs/             # empty for now (no separate write-up; the notebooks hold the results)
├── results/          # empty for now
├── tests/            # pytest tests (`python -m pytest tests`)
├── requirements.txt
└── LICENSE
```

---

## AI assistance

I built this project with help from an AI assistant (Claude, ChatGpt, Deepseek). It helped me plan the work, writing and debug the code and notebooks, and help me for rephrasing the draft of this README.

This is separate from the LLM the project *tests*. That one is `gpt-oss-20b` on Groq, and it only turns headlines into scores.

Because of this, I don't treat any claim here as true just because the code ran. The numbers in the results section are copied from notebook outputs, which can be re-run, and the notebooks include checks on fake data where the answer is known. But the code and the reasoning deserve a critical read, not trust, so if you see a mistake, please tell me. 

---

## Disclaimer

This is a research and learning project, not financial advice. Nothing here has been tested with real money.

## References

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* (`ProsusAI/finbert` on Hugging Face)
- Loughran, T. and McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance. (word lists)
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* (Financial PhraseBank)

## License

MIT, see [LICENSE](LICENSE).
