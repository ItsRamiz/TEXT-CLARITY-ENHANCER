import csv
import math
import sys
import pandas as pd
import re
from collections import Counter
import math


class Trigram_LM:
    def __init__(self):
        self.word_frequency = {}
        self.word_count = 0
        self.trigram_frequency = {}
        self.trigram_count = 0
        self.bigram_frequency = {}
        self.bigram_count = 0
        self.masked_sentences = []
        self.generate_token_flag = 0

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
    def cut_last_three_words(self,sentence):
        try:
            words = sentence.split()
            return ' '.join(words[-3:]) if len(words) >= 3 else ' '.join(words)
        except Exception as e:
            print("Error occured")
            sys.exit(1)
    def calculate_prob_of_sentence(self, text,method="Linear"):
        try:
            probability = 1
            oneProb = 0
            twoProb = 0
            threeProb = 0
            all_Sentences = []
            if self.generate_token_flag == 1:
                all_Sentences.append(self.cut_last_three_words(text))
            else:
                text = text.split()
                current = text[0]
                all_Sentences.append(current)
                for word in text[1:]:
                    current = current + " " + word
                    all_Sentences.append(current)

            if method == "Laplace":
                for sentence in all_Sentences:
                    words = sentence.split()
                    number_of_words = len(words)
                    oneProb = 1
                    twoProb = 1
                    threeProb = 1

                    if number_of_words == 1:
                        divide = sum(self.word_frequency.values()) + len(self.word_frequency)
                        if(divide == 0):
                            oneProb = 1 / divide
                        else:
                            if(sentence in self.word_frequency):
                                oneProb = (self.word_frequency[sentence] + 1) / divide
                            else:
                                oneProb = 1 / divide
                    if number_of_words == 2:
                        first_word = sentence.split()[0]
                        if sentence in self.bigram_frequency:
                            if first_word in self.bigram_frequency:
                                twoProb = (self.bigram_frequency[sentence] + 1) / (self.word_frequency[first_word] + len(self.bigram_frequency))
                            else:
                                twoProb = (self.bigram_frequency[sentence] + 1) / (1 + len(self.bigram_frequency))
                        else:
                            if first_word in self.word_frequency:
                                twoProb = (1) / (self.word_frequency[first_word] + len(self.bigram_frequency))
                            else:
                                twoProb = (1) / (1 + len(self.bigram_frequency))

                    if number_of_words > 2:
                        two_words = sentence.split()[-3:-1]
                        two_words = ' '.join(two_words)
                        two_words = two_words.strip()

                        last_three_words = sentence.split()[-3:]
                        last_three_words = ' '.join(last_three_words)
                        last_three_words = last_three_words.strip()
                        if(last_three_words in self.trigram_frequency):
                            if(two_words in self.bigram_frequency):
                                threeProb = (self.trigram_frequency[last_three_words] + 1) / (self.bigram_frequency[two_words] + len(self.trigram_frequency))
                            else:
                                threeProb = (self.trigram_frequency[last_three_words] + 1) / (1 + len(self.trigram_frequency))
                        else:
                            if(two_words in self.bigram_frequency):
                                threeProb = (1) / (self.bigram_frequency[two_words] + len(self.trigram_frequency))
                            else:
                                threeProb = (1) / (1 + len(self.trigram_frequency))

                    probability = probability * oneProb * twoProb * threeProb
                printProb = math.log2(probability)
                print(f"{printProb:.3f}")
                return math.log2(probability)

            elif method == "Linear":
                for sentence in all_Sentences:
                    words = sentence.split()
                    number_of_words = len(words)
                    lambdaOne = 0.2
                    lambdaTwo = 0.4
                    lambdaThree = 0.4

                    if number_of_words == 1:
                        firstProbability = lambdaOne * (2 ** self.calculate_prob_of_sentence(sentence,"Laplace"))
                        secondProbability = 0
                        thirdProbability = 0

                    if number_of_words == 2:
                        last_word = sentence.split()[-1]

                        two_words = sentence.split()[-2:]
                        two_words = ' '.join(two_words)
                        two_words = two_words.strip()

                        two_words_b = sentence.split()[-2:-1]
                        two_words_b = ' '.join(two_words_b)
                        two_words_b = two_words_b.strip()

                        firstProbability = lambdaOne * (2 ** self.calculate_prob_of_sentence(sentence, "Laplace"))
                        if two_words_b in self.word_frequency:
                            if two_words in self.bigram_frequency:
                                secondProbability = lambdaTwo * (
                                            self.bigram_frequency[two_words] / self.word_frequency[two_words_b])
                            else:
                                secondProbability = 0
                        else:
                            secondProbability = 0
                        thirdProbability = 0


                    if number_of_words > 2:
                        last_word = sentence.split()[-1]

                        two_words = sentence.split()[-2:]
                        two_words = ' '.join(two_words)
                        two_words = two_words.strip()

                        two_words_b = sentence.split()[-2:-1]
                        two_words_b = ' '.join(two_words_b)
                        two_words_b = two_words_b.strip()

                        last_three_words = sentence.split()[-3:]
                        last_three_words = ' '.join(last_three_words)
                        last_three_words = last_three_words.strip()

                        last_three_words_b = sentence.split()[-3:-1]
                        last_three_words_b = ' '.join(last_three_words_b)
                        last_three_words_b = last_three_words_b.strip()
                        firstProbability = lambdaOne * (2 ** self.calculate_prob_of_sentence(sentence,"Laplace"))
                        if two_words_b in self.word_frequency:
                            if two_words in self.bigram_frequency:
                                secondProbability = lambdaTwo * (self.bigram_frequency[two_words] / self.word_frequency[two_words_b])
                            else:
                                secondProbability = 0
                        else:
                            secondProbability = 0

                        if last_three_words_b in self.bigram_frequency:
                            if last_three_words in self.trigram_frequency:
                                thirdProbability = lambdaThree * (self.trigram_frequency[last_three_words] / self.bigram_frequency[last_three_words_b])
                            else:
                                thirdProbability = 0
                        else:
                            thirdProbability = 0

                    probability = firstProbability + secondProbability + thirdProbability
            printProb = math.log2(probability)
            print(f"{printProb:.3f}")
            return math.log2(probability)
        except Exception as e:
            print("Error occured")
            sys.exit(1)
    def generate_next_token(self,text):
        try:
            max_prob = -10000
            max_token = ""
            text = text + " [*]"
            text = text.replace("[*]", "^")
            text = text.replace("^^", "^ ^")
            split = text.split()
            i = 0
            addedTokens = []
            self.generate_token_flag = 1
            while i < len(split):
                if (split[i] == "^"):
                    if i == 0:
                        max_prob = -100000
                        for word in self.word_frequency:
                            probability = self.calculate_prob_of_sentence(word, "Linear")
                            if probability > max_prob:
                                max_prob = probability
                                max_token = word
                    if i == 1:
                        max_prob = -100000
                        original_text = split[i - 2] + " " + split[i - 1] + " "
                        for word in self.word_frequency:
                            sentence = original_text + word
                            probability = self.calculate_prob_of_sentence(sentence, "Linear")
                            if probability > max_prob:
                                max_prob = probability
                                max_token = word
                    if i > 1:
                        max_prob = -100000
                        original_text = split[i - 2] + " " + split[i - 1] + " "
                        for word in self.word_frequency:
                            sentence = original_text + word
                            probability = self.calculate_prob_of_sentence(sentence, "Linear")
                            if probability > max_prob:
                                max_prob = probability
                                max_token = word
                    split[i] = max_token
                    addedTokens.append(max_token)
                i += 1
            print(max_token)
            self.generate_token_flag = 0
            return max_token
        except Exception as e:
            print("Error occured")
            sys.exit(1)

    def generate_next_token_Corrupt(self,text):
        try:
            max_prob = -10000
            max_token = ""
            text = text.replace("[*]", "^")
            text = text.replace("^^","^ ^")
            split = text.split()
            i = 0
            addedTokens = []
            self.generate_token_flag = 1
            while i < len(split):
                if(split[i] == "^"):
                    if i == 0:
                        max_prob = -10000
                        for word in self.word_frequency:
                            probability = self.calculate_prob_of_sentence(word,"Linear")
                            if probability > max_prob:
                                max_prob = probability
                                max_token = word
                    if i == 1:
                        max_prob = -10000
                        original_text = split[i - 2] + " " + split[i - 1] + " "
                        for word in self.word_frequency:
                            sentence = original_text + word
                            probability = self.calculate_prob_of_sentence(sentence, "Linear")
                            if probability > max_prob:
                                max_prob = probability
                                max_token = word
                    if i > 1:
                        max_prob = -10000
                        original_text = split[i - 2] + " " + split[i - 1] + " "
                        for word in self.word_frequency:
                            sentence = original_text + word
                            probability = self.calculate_prob_of_sentence(sentence, "Linear")
                            if probability > max_prob:
                                max_prob = probability
                                max_token = word
                    split[i] = max_token
                    addedTokens.append(max_token)
                i += 1
            sentence = " ".join(split)
            self.generate_token_flag = 0
            return sentence, max_prob , addedTokens
        except Exception as e:
            print("Error occured")
            sys.exit(1)


def read_sentences_from_csv(file_path, column_name):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        sentences = df[column_name].tolist() # only use sentence_text col

        return sentences

    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def tokenize(text):
    try:
        tokens = text.split()
        return tokens
    except Exception as e:
        print("Error occured")
        sys.exit(1)

def calculate_pmi(ngram, n, ngram_count_dict, ngrams_number, unigram_dict, corpus_size): # Function to calculate PMI of a given ngram
    try:
        sum = 0
        current_ngram_count = ngram_count_dict[ngram]  # Get the count of the current ngram
        for i in range(n):
            word_i_count = unigram_dict[ngram[i]]  # count of each word
            sum += math.log2(word_i_count / corpus_size)  #log probability of each word


        pmi = math.log2(current_ngram_count / ngrams_number) - sum #log(P(ngram) / P(w1) * P(w2) * ... * P(wn))
        return pmi
    except Exception as e:
        print("Error occured")
        sys.exit(1)

def get_k_n_collocations(sentences, k, n):
    try:
        n_grams = [tuple(sentence[i:i + n]) for sentence in sentences for i in range(len(sentence) - n + 1)]

        n_gram_counts = Counter(n_grams) #n_gram_counts[n_gram] gives number of times this ngram appeared
        collocations = []
        ngrams_number = len(n_grams) #number of ngrams in corpus
        corpus_size=len(sentences)
        all_words = [word for sentence in sentences for word in sentence]
        unigram_dict = Counter(all_words)  # unigram_dict[word] number of appearances

        for n_gram, count in n_gram_counts.items():

            pmi = calculate_pmi(n_gram, n, n_gram_counts, ngrams_number, unigram_dict,corpus_size)
            collocations.append((n_gram, pmi))

        sorted_collocations = sorted(collocations, key=lambda x: x[1], reverse=True) # sorting by pmi, from high to low.

        return sorted_collocations[:k]
    except Exception as e:
        print("Error occured")
        sys.exit(1)

if __name__ == '__main__':
    try:
        test = Trigram_LM()
        # Example usage:
        file_path = "example_knesset_corpus.csv"  # Replace with the path to your CSV file
        column_name = "sentence_text"  # Replace with the column name containing your sentences
        protocol_column = "protocol_type"
        df = pd.read_csv(file_path)
        plenary_sentences = df[df[protocol_column] == "plenary"][column_name].tolist()
        committee_sentences = df[df[protocol_column] == "committee"][column_name].tolist()

        plenary_tokenized = [tokenize(sentence) for sentence in plenary_sentences]
        committee_tokenized = [tokenize(sentence) for sentence in committee_sentences]

        plenary = Trigram_LM()
        committe = Trigram_LM()
        for sentence in plenary_sentences:
            plenary.addSentence(sentence)
        for sentence in committee_sentences:
            committe.addSentence(sentence)
        file_path = "knesset_collocations.txt"
        with open('masked_sentences.txt', 'r', encoding='utf-8') as file:
            sentences = []
            for line in file:
                sentences.append(line.strip())





        with open('sentences_results.txt', 'w', encoding='utf-8') as file:
            for i in range(len(sentences)):
                file.write("Original sentence: " + sentences[i] + "\n")
                sentences[i] = sentences[i].replace("[*] [*] [*]", "^^^")
                sentences[i] = sentences[i].replace("[*] [*]", "^^")
                sentences[i] = sentences[i].replace("[*]", "^")

                committe_sentence, committe_probability, adddedTokens = committe.generate_next_token_Corrupt(sentences[i])

                adddedTokens = ",".join(adddedTokens)

                file.write("Commitee sentence: " + committe_sentence + "\n")
                file.write("Committee tokens: " + adddedTokens + "\n")
                a = committe.calculate_prob_of_sentence(committe_sentence)
                b = plenary.calculate_prob_of_sentence(committe_sentence)
                file.write("Probability of committee sentence in committee corpus: " + f"{a:.3f}" + "\n")
                file.write("Probability of committee sentence in plenary corpus: " + f"{b:.3f}" + "\n")
                if b > a:
                    file.write("This sentence is more likely to appear in corpus: Plenary\n")
                else:
                    file.write("This sentence is more likely to appear in corpus: Commitee\n")


                plenary_sentence, plenary_probability, addedTokens = plenary.generate_next_token_Corrupt(sentences[i])
                file.write("Plenary sentence: " + plenary_sentence + "\n")

                addedTokens = ",".join(addedTokens)

                file.write("Plenary tokens: " + addedTokens + "\n")
                a = committe.calculate_prob_of_sentence(plenary_sentence)
                b = plenary.calculate_prob_of_sentence(plenary_sentence)
                file.write("Probability of plenary sentence in plenary corpus: " + f"{b:.3f}" + "\n")
                file.write("Probability of plenary sentence in committee corpus: " + f"{a:.3f}" + "\n")

                if b > a:
                    file.write("This sentence is more likely to appear in corpus: Plenary\n")
                else:
                    file.write("This sentence is more likely to appear in corpus: Commitee\n")




                file.write("\n")  # Add an extra newline for spacing between entries

        file_path = "knesset_collocations.txt"
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("Two-gram collocations:\n")
            if committee_sentences:
                file.write("Committee corpus:\n")
                top_2gram_collocations_committee = get_k_n_collocations(committee_tokenized, 10, 2)
                for n_gram, _ in top_2gram_collocations_committee:  # dont take pmi value
                    collocation_str = ' '.join(map(str, n_gram))  # convert each element to string
                    file.write(collocation_str + '\n')

            if plenary_sentences:
                file.write("\nPlenary corpus:\n")
                top_2gram_collocations_plenary = get_k_n_collocations(plenary_tokenized, 10, 2)
                for n_gram ,_ in top_2gram_collocations_plenary:  # dont take PMI values
                    collocation_str = ' '.join(map(str, n_gram))  # convert each element to string
                    file.write(collocation_str + '\n')

            file.write("\nThree-gram collocations:\n")
            if committee_sentences:
                file.write("Committee corpus:\n")
                top_3gram_collocations_committee = get_k_n_collocations(committee_tokenized, 10, 3)
                for n_gram, _ in top_3gram_collocations_committee:
                    collocation_str = ' '.join(map(str, n_gram))
                    file.write(collocation_str + '\n')

            if plenary_sentences:
                file.write("\nPlenary corpus:\n")
                top_3gram_collocations_plenary = get_k_n_collocations(plenary_tokenized, 10, 3)
                for n_gram, _ in top_3gram_collocations_plenary:
                    collocation_str = ' '.join(map(str, n_gram))
                    file.write(collocation_str + '\n')

            file.write("\nFour-gram collocations:\n")
            if committee_sentences:
                file.write("Committee corpus:\n")
                top_4gram_collocations_committee = get_k_n_collocations(committee_tokenized, 10, 4)
                for n_gram , _ in top_4gram_collocations_committee:
                    collocation_str = ' '.join(map(str, n_gram))
                    file.write(collocation_str + '\n')

            if plenary_sentences:
                file.write("\nPlenary corpus:\n")
                top_4gram_collocations_plenary = get_k_n_collocations(plenary_tokenized, 10, 4)
                for n_gram , _ in top_4gram_collocations_plenary:
                    collocation_str = ' '.join(map(str, n_gram))
                    file.write(collocation_str + '\n')
    except Exception as e:
        print("Error occured")
        sys.exit(1)