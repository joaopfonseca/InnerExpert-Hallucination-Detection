""" 
This script creates the labels for the dataset, which will be used to trained
ML models to detect when a model hallucinates.

The labels are created as a mask of the same shape as the output tokens, where
each token will be labeled as 1 if a given token is part of a hallucination and
0 otherwise.

Generating answer-level labels
------------------------------

To create answer-level labels, we use a combination of the metrics collected in
previous steps. Specifically, we will combine the following metrics to
determine whether there is a hallucination at each answer:

- Whether evidence was provided (binary)
- BertScore (precision, recall, F1)
- ROUGE (1, 2, L and Lsum)
- BLEU
- LLM-based evaluation (e.g., GPT-4 evaluation scores), using the OpenAI API to
  query the model with a prompt that asks it to evaluate the answer based on
  the provided evidence and assign a binary label (1 for hallucinated, 0 for
  grounded/correct).

We set a threshold for each non-binary metric (BertScore, ROUGE and BLEU) based
on the threshold that best separates evidence-supported answers (likely
correct) from non-evidence-supported answers (likely hallucinated) in the
training data. See ``3.1-analyze-metrics.py`` for details on how to determine
these thresholds.

We can then train a simple classifier using the evidence binary metric,
BertScore, ROUGE and BLEU metrics as features to predict the binary label of
the LLM-based evaluation. This will produce an answer-level confidence score
indicating how likely the model’s answer is likely to be hallucinated (1) or
grounded/correct (0).


Generating token-level labels
-----------------------------

To create token-level labels, we will use LLM-based querying to determine which
specific tokens in the model's answer are hallucinated. We can use a prompt
that asks the LLM to identify which tokens in the answer are not supported by
the provided evidence. The LLM outputs the passage containing the hallucinated
tokens, and we can use this information to create a token-level mask indicating
which tokens are hallucinated (1) and which are grounded (0).

All these labels will be saved in the same format as the original dataset, with
additional columns for the answer-level and token-level labels. This will allow
us to easily use these labels for training ML models. 

"""

# Imports

# functions

# if __name__ == "__main__":

# flags

# Read data (use pandas, set up dataset path the same way as in the previous scripts)

# Generate answer-level labels

# Generate token-level labels

# Save the dataset with the new labels
