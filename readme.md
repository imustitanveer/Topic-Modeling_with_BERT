# BERT for Social Media Trend Analysis

This project explores the use of large language models (LLMs), specifically BERT (Bidirectional Encoder Representations from Transformers), for identifying emerging trends and topics in social media discussions, focusing on Twitter data. The aim is to compare the performance of fine-tuned BERT with traditional topic modeling approaches, such as Latent Dirichlet Allocation (LDA), in accurately predicting trends.

## Methodology

### Data Collection
The study utilizes two datasets: a smaller dataset with 25,775 tweets from random Twitter users for model training and fine-tuning, and a larger dataset with 1.6 million tweets from the Sentiment140 dataset for evaluation. Only the tweet text is used for analysis.

### Fine-Tuning BERT
BERT is fine-tuned on the smaller dataset using a masked language modeling objective. Parameters are adjusted for optimal performance, including learning rates and batch sizes.

### Evaluation
The performance of fine-tuned BERT is evaluated using a semantic similarity test. The test involves comparing the semantic similarity of BERT embeddings with LDA embeddings for a set of tweets from the evaluation dataset. K-means clustering is also used to analyze how the models identify topics and their coherence.

## Results

The table below shows the average semantic similarity between BERT and LDA for each iteration of the evaluation:

| Iteration | LDA Semantic Similarity | BERT Semantic Similarity |
|-----------|--------------------------|--------------------------|
| 1         | 0.768798302              | 0.77587591               |
| 2         | 0.773550481              | 0.81008035               |
| 3         | 0.768508145              | 0.770734991              |
| 4         | 0.752353276              | 0.772750738              |
| 5         | 0.778895441              | 0.792108541              |
| 6         | 0.7752804                | 0.818383306              |
| 7         | 0.751067267              | 0.77288803               |
| 8         | 0.772680566              | 0.789506231              |
| 9         | 0.776129356              | 0.785144771              |
| 10        | 0.780811098              | 0.795021537              |
| 11        | 0.73497187               | 0.780327765              |
| 12        | 0.775209758              | 0.855840848              |
| 13        | 0.765785678              | 0.776920556              |
| 14        | 0.741108033              | 0.768066693              |
| 15        | 0.774984042              | 0.792108562              |
| 16        | 0.774543089              | 0.830515263              |
| 17        | 0.773322774              | 0.779570525              |
| 18        | 0.774186163              | 0.830648443              |
| 19        | 0.76998461               | 0.901335327              |
| 20        | 0.775669233              | 0.799434663              |


The results indicate that fine-tuned BERT consistently outperforms LDA in identifying trends and topics in social media discussions, as shown by the higher semantic similarity scores.
