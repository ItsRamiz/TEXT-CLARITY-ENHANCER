import csv
import random
import sys

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class PlenaryChunk: # used to have 14104 chunks

    def __init__(self):
        self.sentences = []
        self.chunks = []
        self.word_frequency = {}
        self.word_count = 0
        self.trigram_frequency = {}
        self.trigram_count = 0
        self.bigram_frequency = {}
        self.bigram_count = 0

    def add_sentence(self, sentence):
        self.sentences.append(sentence)


    def addSentence(self,text):
        try:
            text = text.split()
            for word in text:
                word = word.strip()
                #print("@",word,"@")
                self.word_count += 1
                if word in self.word_frequency:
                    self.word_frequency[word] += 1
                else:
                    self.word_frequency[word] = 1

            bigrams = zip(text, text[1:])
            for bigram in bigrams:
                #print("@",bigram,"@")
                self.bigram_count += 1
                bigram_key = " ".join(bigram)
                if bigram_key in self.bigram_frequency:
                    self.bigram_frequency[bigram_key] += 1
                else:
                    self.bigram_frequency[bigram_key] = 1

            trigrams = zip(text, text[1:], text[2:])
            for trigram in trigrams:
                #print("@",trigram,"@")
                self.trigram_count += 1
                trigram_key = " ".join(trigram)
                if trigram_key in self.trigram_frequency:
                    self.trigram_frequency[trigram_key] += 1
                else:
                    self.trigram_frequency[trigram_key] = 1
        except Exception as e:
            print("Error occured")
            sys.exit(1)


    def chunk_sentences(self):
        self.chunks = self._chunk_sentences(self.sentences)

    def _chunk_sentences(self, sentences):
        chunks = []
        num_chunks = len(sentences) // 5
        sentences = sentences[:num_chunks * 5]
        for i in range(0, len(sentences), 5):
            chunks.append(sentences[i:i+5])
        return chunks
class CommitteeChunk: # used to have 5895 chunks
    def __init__(self):
        self.sentences = []
        self.chunks = []
        self.word_frequency = {}
        self.word_count = 0
        self.trigram_frequency = {}
        self.trigram_count = 0
        self.bigram_frequency = {}
        self.bigram_count = 0

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def addSentence(self,text):
        try:
            text = text.split()
            for word in text:
                word = word.strip()
                #print("@",word,"@")
                self.word_count += 1
                if word in self.word_frequency:
                    self.word_frequency[word] += 1
                else:
                    self.word_frequency[word] = 1

            bigrams = zip(text, text[1:])
            for bigram in bigrams:
                #print("@",bigram,"@")
                self.bigram_count += 1
                bigram_key = " ".join(bigram)
                if bigram_key in self.bigram_frequency:
                    self.bigram_frequency[bigram_key] += 1
                else:
                    self.bigram_frequency[bigram_key] = 1

            trigrams = zip(text, text[1:], text[2:])
            for trigram in trigrams:
                #print("@",trigram,"@")
                self.trigram_count += 1
                trigram_key = " ".join(trigram)
                if trigram_key in self.trigram_frequency:
                    self.trigram_frequency[trigram_key] += 1
                else:
                    self.trigram_frequency[trigram_key] = 1
        except Exception as e:
            print("Error occured")
            sys.exit(1)


    def chunk_sentences(self):
        self.chunks = self._chunk_sentences(self.sentences)

    def _chunk_sentences(self, sentences):
        chunks = []
        num_chunks = len(sentences) // 5
        sentences = sentences[:num_chunks * 5]
        for i in range(0, len(sentences), 5):
            chunks.append(sentences[i:i+5])
        return chunks

def retrieveChunks():
    sentence_endings = ['.', '?', '!']
    chunks = []
    # Open the file
    with open('knesset_text_chunks.txt', 'r', encoding='utf-8') as file:
        # read each line in the file
        for line in file:
            sentences = []

            current_sentence = ''

            for char in line:

                current_sentence += char

                if char in sentence_endings:
                    sentences.append(current_sentence.strip())
                    current_sentence = ''  # reset the current sentence

            # add the last sentence if it exists
            if current_sentence.strip():
                sentences.append(current_sentence.strip())

            # check if the sentence list is non-empty before appending it to the chunks list
            if sentences:
                chunks.append(sentences)

    return chunks

def PrepareData():
    plenary_chunk = PlenaryChunk()
    committee_chunk = CommitteeChunk()
    chunks = retrieveChunks()

    with open('example_knesset_corpus.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentence = row['sentence_text']
            protocol_type = row['protocol_type']
            if protocol_type == 'plenary':
                plenary_chunk.add_sentence(sentence)
            elif protocol_type == 'committee':
                committee_chunk.add_sentence(sentence)

    plenary_chunk.chunk_sentences()
    committee_chunk.chunk_sentences()

    num_chunks_to_keep = len(committee_chunk.chunks)
    random_plenary_chunks = random.sample(plenary_chunk.chunks, num_chunks_to_keep)  # balancing the objects
    plenary_chunk.chunks = random_plenary_chunks  # we changed the number of chunks in plenary to that of commitee, by choosing randomly.
    return plenary_chunk,committee_chunk,chunks

def compare_word_frequencies(inst1, inst2):

    list = []
    for word in inst1.word_frequency:
        count1 = inst1.word_frequency.get(word, 0)
        count2 = inst2.word_frequency.get(word, 0)
        if (abs(count1 - count2) > 5 and (count2 == 0 or count1 == 0)):
            list.append(word)
            #print(word)
    for word in inst1.bigram_frequency:
        count1 = inst1.bigram_frequency.get(word, 0)
        count2 = inst2.bigram_frequency.get(word, 0)
        if(abs(count1 - count2) > 5 and (count2 == 0 or count1 == 0)):
            list.append(word)
            #print(word)
    list.append("!")
    return list

random.seed(42)
np.random.seed(42)

plenary_chunk, committee_chunk, chunks = PrepareData()

committee_connected = [' '.join(inner_list) for inner_list in committee_chunk.chunks]
plenary_connected = [' '.join(inner_list) for inner_list in plenary_chunk.chunks]


for i in committee_connected:
    committee_chunk.addSentence(i)
for i in plenary_connected:
    plenary_chunk.addSentence(i)

#compare_word_frequencies(plenary_chunk,committee_chunk)
#sys.exit(1)
def count_tokens(sentence):

    tokens = sentence.split()
    return len(tokens)

max = 5000
#ngram_range=(1),
vectorizer = TfidfVectorizer(max_features= max)
all_texts = committee_connected + plenary_connected
all_vectors = vectorizer.fit_transform(all_texts)

vocabulary = vectorizer.get_feature_names_out()
vocabulary = list(vocabulary)

add_words = compare_word_frequencies(committee_chunk,plenary_chunk)
for i in add_words:
    if i not in vocabulary:
        vocabulary.append(i)
vocabulary = np.array(vocabulary)

vectorizer = TfidfVectorizer(vocabulary=vocabulary, max_features= max)

print("Vocabulary = " , len(vectorizer.get_feature_names_out()))
all_vectors = vectorizer.fit_transform(all_texts)
all_labels = ["Committee"] * len(committee_connected) + ["Plenary"] * len(plenary_connected)





knn = KNeighborsClassifier(n_neighbors=50)
svm = svm.SVC(kernel = 'linear')

print("KNN - Cross Validation")
predictions = cross_val_predict(knn, all_vectors, all_labels, cv=10, n_jobs=-1)
print(classification_report(all_labels, predictions))

print("KNN - Train & Split")
X_train, X_test, y_train, y_test = train_test_split(all_vectors, all_labels, test_size=0.1, stratify=all_labels, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))


print("SVM - Cross Validation")
predictions = cross_val_predict(svm, all_vectors, all_labels, cv=10, n_jobs=-1)
print(classification_report(all_labels, predictions))

print("SVM - Train & Split")
X_train, X_test, y_train, y_test = train_test_split(all_vectors, all_labels, test_size=0.1, stratify=all_labels, random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

test_connected = [' '.join(inner_list) for inner_list in chunks]
all_vectors_test = vectorizer.transform(test_connected)

svm.fit(all_vectors,all_labels)
predictions = svm.predict(all_vectors_test)

with open("classification_results.txt", 'w') as file:
    for i in predictions:
        file.write(i + '\n')
