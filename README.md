# Tokenizer Vocabulary Allocation across Writing Systems

Working repository for a bachelor's thesis on adapting the tokenizer of a pretrained LLM to several writing systems under a fixed vocabulary size, with one-to-many streaming translation as the target application. The repository holds the code, intermediate results and project notes. Each step has its own section below, so that the choices made earlier remain traceable when the next experiments are planned.

Working thesis title: *Optimizing Tokenizer Vocabulary Allocation across Writing Systems under a Fixed Vocabulary Size for One-to-Many Streaming Translation.*

## Status

| Step | Content | State |
| --- | --- | --- |
| 1 | Results of the research internship: baseline, pruning, vocabulary substitution, FVT initialization | done (intrinsic evaluation only) |
| 2 | Extrinsic evaluation and a second initialization method | planned |
| 3 | Vocabulary allocation across scripts as an optimization problem | planned |

## Step 1. Results of the research internship

### Task

Adapt the tokenizer of an already trained LLM, `ai-sage/GigaChat3.1-10B-A1.8B`, to a multilingual scenario with different writing systems. The vocabulary size stays fixed: new tokens are written into slots freed by removing unused ones, so the embedding matrices keep their shape.

Model properties relevant for the method: a mixture-of-experts model (`DeepseekV3ForCausalLM`), 128,256 embedding rows, untied input and output embeddings, byte-level BPE.

### Target languages

| Language | Writing system | Role |
| --- | --- | --- |
| English | Latin | reference, not modified |
| Russian | Cyrillic | reference for premium, not modified |
| Chinese | Han characters, no whitespace between words | adapted |
| Arabic | Arabic script (cursive) | adapted |
| Hindi | Devanagari | adapted |

### Metrics

Intrinsic metrics can be computed without training or fine-tuning the model. Extrinsic metrics require a model with the new vocabulary and are not yet computed (see Step 2).

| Type | Metric | What it measures | Source |
| --- | --- | --- | --- |
| Intrinsic | Fertility | tokens per word (for Chinese, words come from `jieba`) | standard |
| Intrinsic | Bytes / characters per token | compression | standard |
| Intrinsic | Rényi efficiency (α = 2.5) | balance of token frequencies, normalized by vocabulary size | Zouhar et al., 2023 |
| Intrinsic | STRR | share of frequent words encoded as one token | Nayeem et al., 2025 |
| Intrinsic | Tokenization premium | length of a text relative to its translation in a reference language | Petrov et al., 2023 |
| Extrinsic | chrF / COMET, win rate, MMLU, latency | translation quality, forgetting, inference cost | planned |

MorphScore (Arnett et al., 2025) was reviewed but is not part of the reported results.

### Data

* FLORES+ (`openlanguagedata/flores_plus`), parallel text, used for premium.
* FineWeb-2 (FineWeb-Edu for English), used for the remaining intrinsic metrics and for training the language-specific tokenizers.
* `hermitdave/FrequencyWords`, frequency lists used as the word list for STRR.

### Baseline

Metrics of the original tokenizer on a FineWeb sample.

| Language | Fertility | Bytes / token | Chars / token | STRR, % | Rényi eff. (α = 2.5) |
| --- | --- | --- | --- | --- | --- |
| Russian | 1.7 | 7.4 | 4.6 | 18.8 | 0.38 |
| English | 1.4 | 4.4 | 4.6 | 71.9 | 0.37 |
| Chinese | 1.7 | 2.7 | 1.1 | 49.0 | 0.43 |
| Arabic | 1.8 | 5.7 | 3.3 | 19.3 | 0.44 |
| Hindi | 4.9 | 2.6 | 1.0 | 0.2 | 0.28 |

On FLORES+, Hindi is encoded 3.2 to 4.7 times longer than any of the other four languages, while Russian, English, Arabic and Chinese lie close to each other (0.68 to 1.48 times). Hindi is therefore the main target of the adaptation, Chinese and Arabic are secondary, and Russian and English are left untouched.

### Pipeline

1. **Vocabulary analysis.** For every token id, `decode([id])` is re-encoded and compared with the id. Result: 123,863 normal, 2 special, 770 broken (fragments of multi-byte UTF-8 characters, decoded with U+FFFD), 3,378 unreachable (the string is in the vocabulary but the tokenizer never produces it as one token) and 243 unallocated ids.
2. **Detection of under-trained tokens** following Land and Bartolo (2024). The input embeddings of this model are untied, and the official repository states that weight decay is not applied to embeddings for the `deepseek_v3` architecture, so the L2 norm of the input embedding is not a reliable indicator here. The main indicator is the cosine distance between a token's output embedding and the mean output embedding of unused tokens; the L2 norm is kept as a secondary diagnostic. Spearman correlation between the two indicators is 0.531. The 2 % of normal tokens with the lowest score (2,477 candidates) are verified.
3. **Verification by prompting.** The candidate id is substituted for a placeholder token directly in the tensor of input ids, and the model is asked to repeat it, to explain it, and to continue repeating it. The maximum probability of reproducing the token decides the label: above 0.9 strong rejected, above 0.1 weak rejected, 0.001 to 0.1 weak verified, below 0.001 strong verified. 2,289 of the 2,477 candidates are confirmed as under-trained (strong and weak verified).
4. **Pruning** with respect to the BPE merge structure. A token is removed together with the merge rule that creates it, and a token that is a component of another token remaining in the vocabulary is kept. 5,481 tokens were removed and 177 were kept as components. Together with the 243 unallocated ids this gives a pool of 5,724 free slots.
5. **Allocation of slots and training of new tokenizers.** Slots are divided in proportion to (premium − 1) relative to Russian. A separate byte-level BPE tokenizer is trained for each language on FineWeb-2 (about 5.0 M words for Hindi, about 2.0 M for Arabic and Chinese). Vocabulary size is (number of the language's tokens in GigaChat + number of new tokens) × 2 + 256, which leaves a margin for the filtering below. A candidate token is accepted if it contains no U+FFFD, contains characters of the language's script, and is not already a whole token in the pruned GigaChat vocabulary.
6. **Initialization of embeddings with FVT** (Gee et al., 2022). A new token is split by the old tokenizer, and its embedding is the mean of the embeddings of the resulting pieces. Output embeddings are initialized analogously. Only the new tokens are initialized, the rest of the vocabulary is unchanged.

| Language | Share of slots | Slots | Already in vocabulary | Selected |
| --- | --- | --- | --- | --- |
| Hindi | 88.3 % | 5,053 | 26 | 4,686 |
| Arabic | 2.7 % | 156 | 7,758 | 156 |
| Chinese | 9.0 % | 515 | 1,963 | 515 |

Examples of selected tokens: Hindi `ं`, ` ह`, `य`; Arabic `يث`, ` ولك`, `رح`; Chinese `這`, `來`, `們`.

### Results

Values are given as before → after adaptation.

| Language | Fertility | Rényi eff. | STRR, % | Premium |
| --- | --- | --- | --- | --- |
| Russian | 1.716 → 1.716 | 0.398 → 0.398 | 18.8 → 18.8 | 1.000 → 1.000 |
| English | 1.385 → 1.385 | 0.385 → 0.385 | 71.9 → 71.9 | 0.899 → 0.899 |
| Chinese | 1.658 → 1.487 | 0.437 → 0.429 | 49.0 → 56.2 | 1.330 → 1.278 |
| Arabic | 1.794 → 1.878 | 0.493 → 0.487 | 19.3 → 19.3 | 1.099 → 1.160 |
| Hindi | 4.939 → 3.170 | 0.274 → 0.320 | 0.2 → 8.6 | 4.243 → 2.722 |

* Hindi: fertility decreased by about 36 %, premium from 4.24 to 2.72, STRR from 0.2 to 8.6 %. The gap to the other languages is reduced but not closed.
* Chinese: a small improvement in fertility, STRR and premium.
* Arabic: a slight deterioration in fertility and premium. The cause has not been investigated yet.
* Russian and English: unchanged, as intended.

### Known limitations and open questions

* No extrinsic evaluation. The adapted embeddings and tokenizer exist as separate files and have not been assembled into a checkpoint, so translation quality, forgetting on the original languages and inference speed are unmeasured. Intrinsic metrics can be misleading on their own (Cognetta et al., 2024), and fertility and similar measures correlate weakly with downstream quality in the Russian adaptation study of Andriushchenko et al.
* FVT is the only initialization method. The comparison with random initialization was planned but could not be run without extrinsic evaluation, and CLP+ (Yamaguchi et al., 2024) needs a helper model whose tokenizer matches the target vocabulary.
* For Hindi, 367 of the 5,053 allocated slots were not filled (4,686 selected).
* Slot allocation by (premium − 1) is a heuristic. It was not compared with alternatives.
* Arabic became slightly worse. Possible causes are the small number of new tokens and the effect of pruning on existing Arabic merges.
* The examples of Chinese tokens above appear to be traditional characters, while FLORES+ uses simplified Chinese. The share of traditional and simplified characters among the 515 tokens should be checked.
* Verification prompts are in English, which may underestimate the probability of tokens from other scripts.

## Next steps

* Assemble a checkpoint with the adapted vocabulary and run extrinsic evaluation (chrF and COMET on FLORES+, MMLU for forgetting, latency).
* Compare FVT with random initialization and with at least one method that does not depend on a helper model.
* Check that every inserted token is actually produced by the final tokenizer, since a token inserted from a separate BPE model can conflict with the merge order of the target tokenizer.
* Treat the division of free slots between scripts as an explicit optimization problem and compare it with the proportional rule.

## Repository structure

This is the planned layout and will be adjusted as the code is added.

```
.
├── README.md
├── step1_internship/
│   ├── notebooks/        baseline metrics, result tables
│   ├── src/              vocabulary analysis, under-trained token detection,
│   │                     pruning, tokenizer training, FVT initialization
│   └── results/          metric tables, lists of removed and added tokens
└── notes/                short notes on papers and meeting decisions
```

## References

* Petrov, La Malfa, Torr, Bibi. Language Model Tokenizers Introduce Unfairness Between Languages. NeurIPS 2023.
* Zouhar et al. Tokenization and the Noiseless Channel. ACL 2023.
* Cognetta et al. Two Counterexamples to Tokenization and the Noiseless Channel. 2024.
* Nayeem et al. Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation. arXiv:2510.09947, 2025.
* Arnett, Hudspeth, O'Connor. Evaluating Morphological Alignment of Tokenizers in 70 Languages. arXiv:2507.06378, 2025.
* Land, Bartolo. Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models. EMNLP 2024.
* Gee, Zugarini, Rigutini, Torroni. Fast Vocabulary Transfer for Language Model Compression. EMNLP 2022, Industry Track.
* Yamaguchi, Villavicencio, Aletras. An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Language Model Inference. Findings of EMNLP 2024.
* Andriushchenko et al. A Multi-Aspect Evaluation of Tokenizer Adaptation Methods for LLM on Russian. Doklady Mathematics, 2025.
