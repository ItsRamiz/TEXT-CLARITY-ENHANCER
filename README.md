A set of algorithms from the world of Natural Language Processing:


Note: The following algorithms were tested on:


- 100x Official Hebrew Govrement Transcripts
- Each transcript contained 5 - 33 pages, discussing various topics
- The transcripts were taken from different years, each year the transcript followed a specific syntax, starting from introduction, moving on to the actual context, ending in votes or an ending sentence
- Parts of the transcript were corrupted on purpose for testing purposes

  
**Pre-processing Corpus:** In the pre-processing phase, I've made sure to exclude pure context out of the document, the speaker's name followed by everything he said.
 This included removing the introduction, headlines, dates, images, irrelevant comments made during the meeting, corrupt text, strange symbols and much more.


**Auto-complete Corupus:** With the help of principles learned in Machine Learning, I was able to predict the missing words found in sentences & auto-complete sentences using Linear Interpolation and Laplace smoothing
, concepts learned in Probability Theory.

**Classification Corpus:** In this part, I've used SVM (Support Vector Machines) and KNN (K-Nearest Neighbors), two popular algorithms used in the field of machine learning and pattern recognition for classification and regression tasks.
Given a set of sentences, I was able to classify each sentence to a document type (2 types).

* Implemeted a similiar algorithm to work with 8 types of documents.

