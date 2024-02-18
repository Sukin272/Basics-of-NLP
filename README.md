# Basics-of-NLP

There are no commands to run. Each ipynb can be run and do a certain task.

## ./Word Similarity has 3 ipynb files.

1. VecCheck trains model on "twitter-hate-speech-detection" dataset and creates embeddings for words.
2. ModelCheck checks the accuracy of the trained model.
3. Googlew2v loads a pretrained model and checks its accuracy.

## ./Text Similarity Scores/Normal\ Ways has the ways I tried to predict Sentence similarity using underlying word embeddings.

1. glove uses glove word embeddings and uses its mean to get sentence vector.
2. phraseSim has my various attempts to predict Phrase Similarity.
3. SentenceSim has my various attempts to predict Sentence Similarity.
4. isBestGood directly uses Bert library to predict Sentence Similarity Scores.

## ./Text Similarity Scores/Fine\ Tuning has the ways I tried to fine tune pre trained models to predict sentence similarity.

1. fineTuner makes dataset and fine tunes the model(commented code). It also gets the accuracy of non-tuned and 2 of the fine Tuned models on the test set.
2. ternarySearch is used to find the optimal threshold.

## ./Text Similarity Scores/LLM\ querying has my attempts to query LLM models to do the task of Sentence and pgrase similarity.

1. ChatGptTest has sentence similarity prediction with openai API.
2. ChatGptTestPhrase has phrase similarity prediction with openai API.
3. My attempt to get LLAMA model to predict a text similarity using replicate.

```
.
├── README.md
├── RecruitmentTask.pdf
├── Report.pdf
├── Text Similarity Scores
│   ├── Fine Tuning
│   │   ├── fineTuner.ipynb
│   │   ├── ternarySearch.ipynb
│   │   ├── threshholdFinder.txt
│   │   ├── Tuned
│   │   │   ├── 1_Pooling
│   │   │   │   └── config.json
│   │   │   ├── 2_Normalize
│   │   │   ├── config.json
│   │   │   ├── config_sentence_transformers.json
│   │   │   ├── model.safetensors
│   │   │   ├── modules.json
│   │   │   ├── README.md
│   │   │   ├── sentence_bert_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── vocab.txt
│   │   └── Tuned-v2
│   │       ├── 1_Pooling
│   │       │   └── config.json
│   │       ├── 2_Normalize
│   │       ├── config.json
│   │       ├── config_sentence_transformers.json
│   │       ├── model.safetensors
│   │       ├── modules.json
│   │       ├── README.md
│   │       ├── sentence_bert_config.json
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       ├── tokenizer.json
│   │       └── vocab.txt
│   ├── LLM querying
│   │   ├── ChatGptTest.ipynb
│   │   ├── ChatGptTestPhrase.ipynb
│   │   └── llama.ipynb
│   └── Normal Ways
│       ├── glove.6B.100d.txt
│       ├── glove.ipynb
│       ├── IsBertGood.ipynb
│       ├── phraseSim.ipynb
│       └── SentenceSim.ipynb
└── Word Similarity Scores
    ├── Googlew2v.ipynb
    ├── modelCheck.ipynb
    ├── my_model200.pth
    ├── my_model3.pth
    ├── my_model5.pth
    ├── SimLex-999
    │   ├── README.txt
    │   └── SimLex-999.txt
    ├── vecCheck.ipynb
    └── vocab.json

```
