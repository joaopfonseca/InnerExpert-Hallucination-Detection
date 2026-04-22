# TODO

## Datasets

- [x] **TruthfulQA** (contains best answer, correct answers, and incorrect answers for each question)
- [x] [NQ-Open](https://huggingface.co/datasets/google-research-datasets/nq_open) (question-answer pairs only)
- [x] [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) (very similar to NQ-Open)
- [x] [FreshQA](https://github.com/freshllms/freshqa/tree/main) (also contains a method for evaluating the correctness of the answer, which is not just based on keyword matching)
- [x] **[RealtimeQA](https://github.com/realtimeqa/realtimeqa_public)** (contains question-answer pairs, the questions are based on current events, therefore the answers are more likely to be hallucinated if no evidence is provided - could be a good dataset for a case study)
- [x] [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) (task is summarization, interesting for hallucination detection)
- [ ] [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [ ] HaluEval
- [ ] DefAn
- [ ] ToTTo
- [ ] DialFact
- [ ] FactCC
- [ ] TriviaQA (General question answering, quite a large database - correct answer only with keywords, some of which are not correct or are very inaccurate, e.g., after a couple minutes exploration: (1) first european country to abolish capital punishment is not correct and (2) Prince Henry of Prussia patented the windshield wiper in 1908, not 1911)
- [ ] WikiFact (contains claims and evidence, goal is fact extraction, which is a different task)

## Baselines (see docs/baselines.md for rationale)

- [x] Predictive Entropy (simplest baseline, sets the floor)
- [x] Semantic Uncertainty (Kuhn et al., 2023 / Nature 2024 - gold standard for training-free hallucination detection)
- [x] SelfCheckGPT (Manakul et al., EMNLP 2023 - generation consistency baseline)
- [x] LLM-Check (Mitchell et al., 2023 - internal signal baseline; natural ablation point)
- [x] Semantic Energy (Ma et al., 2025 - logit-space energy, improves over semantic entropy)
- [x] HaluNet (Tong et al., 2025 - trainable upper bound)

## Evaluation Metrics

- [x] ROUGE
- [x] BLEU
- [x] BERTScore — Although this is a popular metric this is not working well for our use case. I would like to use it since it is appealing, but the absolute values are always relatively close to 1, even for very bad answers. In addition, the difference between correct and incorrect answers is basically negligible.
- [ ] LLM evaluation — Must be combined with human evaluation to quantify alignment with human judgement.
- [ ] AUROC / AUPRC (for hallucination detection evaluation)
- [ ] Calibration curves

## Pipeline

- [ ] Sampling generation script for multi-response baselines (Semantic Uncertainty, SelfCheckGPT, Semantic Energy)
- [ ] Run full pipeline end-to-end on 2-3 MoE models
- [ ] Decide on cumulative vs non-cumulative default for experiment scripts