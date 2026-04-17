

To run this project, you will need to create a .env file with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## TODOs

- [ ] Add datasets
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

- [ ] Add baselines
    - [ ] FacLens (hallucination detection)
    - [ ] SelfCheckGPT (hallucination detection)
    - [ ] LLM-Check (hallucination detection)
    - [ ] ? (SafeNudge? Safety evaluation - maybe?)
    - [ ] Semantic Uncertainty (Kuhn et al., 2023 - entropy over semantic clusters of generations)
    - [ ] Predictive Entropy (standard token-level entropy baseline)
    - [ ] DegLM (Dettmers et al., 2022 - density-based confidence estimation)
    - [ ] In-Context Confidence Prompting (ask the model to self-assess confidence)
    - [ ] Semantic Energy (Ma et al., 2025 - Boltzmann energy on penultimate logits + semantic clustering, improves over semantic entropy)
    - [ ] HaluNet (Tong et al., 2025 - multi-granular uncertainty: fuses token-level probability, semantic embeddings, and distributional uncertainty)
    - [ ] Bayesian MoE Routing (Li, 2025 - Bayesian distribution over routing decisions for calibration and OoD detection; directly comparable to our approach)
    - [ ] Unconditional Truthfulness (Vazhentsev et al., EMNLP 2025 - learns unconditional uncertainty signal without requiring input-specific generation)

- [x] Add evaluation metrics
    - [x] ROUGE
    - [x] BLEU
    - [x] BERTScore - Although this is a popular metric this is not working well for our use case. I would like to use it since it is appealing, but the absolute values are always relatively close to 1, even for very bad answers. In addition, the difference between correct and incorrect answers is basically negligible.
    - [ ] LLM evaluation - Must be combined with human evaluation to quantify alignment with human judgement.
