# Modeling Gender Bias in Online Discourse

This project combines large language models and lightweight predictive models to simulate and analyze votin behavior and biases in gender-related discourse found on the social platform, Reddit. Our goal is to model and predict how users upvote/downvote to content under different social and cognitive contexts, and to understand emergent patterns of bias.

## Research Goals
1. Model the **cognitive and social mechanisms** driving onlne voting behavior
2. Produce a novel, reproducible framework of LLM-based social simulations

## Fine-tuning A Lightweight Model
1. Predict upvote/downvotes
2. Analyze/simulate toxicty, bas, or alignment in gender-related discourse
&rarr; Serve as a core module in the social simulation

| Task                        | Base Model          | Why                                  |
| --------------------------- | ------------------------------- | ------------------------------------ |
| Score prediction            | RoBERTa / DistilBERT            | Lightweight, text-focused            |
| Toxicity / bias detection   | RoBERTa / BERT + Detox datasets | Well-supported in toxic discourse    |
| Simulation-compatible model | LLaMA / Mistral (optional)      | Smaller LLMs with custom fine-tuning |
