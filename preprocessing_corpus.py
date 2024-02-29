import csv
import math
from docx import Document
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

class Speaker:
    protocol_name = ""
    knesset_number = ""
    protocol_type = ""
    name = ""
    paragraph = ""
    sentences = []
    tokenized_sentences = []
    joined_tokens = []
    def __init__(self,name,paragraph,protocol_name,knesset_number,protocol_type):
        self.name = name
        self.paragraph = paragraph
        self.sentences = []
        self.tokenized_sentences = []
        self.protocol_name = protocol_name
        self.knesset_number = knesset_number
        self.protocol_type = protocol_type
    def __str__(self):
        result = self.name,"==",self.paragraph
        return result



def Section11(fileName):
    kinesset_number = ""
    protocolType = ""
    protocolName = ""
    i = 0
    try:
        while (fileName[i] != '_'):
            kinesset_number = kinesset_number + fileName[i]
            i += 1
        i += 1
        kinesset_number = int(kinesset_number)
    except Exception as e:
        kinesset_number = 1

    try:
        if (fileName[i + 2] == 'm'):
            protocolType = "plenary"
        else:
            protocolType = "committee"
    except Exception as a:
        protocolType = "unknown"
    return fileName,kinesset_number,protocolType
#FOR PMT
def excludeStart(text):
    start_index = text.find("תוכן העניינים")

    if start_index != -1:
        new_text = text[start_index:]

    start_index = text.find('היו"ר')
    if start_index != -1:
        start_index = start_index - 1
        new_text = text[start_index:]

    return new_text

# FOR PMT
def separate_text_by_patternPMT(text):
    result = []
    current_chunk = ""
    try:
        lines = text.splitlines()
    except Exception as e:
        lines = lines
    try:
        for line in lines:
            if ":" in line and line.endswith(":") and len(line) < 31:
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += ' ' + line

        if current_chunk:
            result.append(current_chunk.strip())
    except Exception as a:
        return text

    return result

#FOR PVT
def separate_text_by_pattern(text):
    try:
        result = []
        current_chunk = ""
        lines = text.splitlines()

        for line in lines:
            if ":" in line:
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += ' ' + line

        if current_chunk:
            result.append(current_chunk.strip())
        return result
    except Exception as e:
        return text

#FOR PMT
def extract_text_from_docx(doc):
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    text.replace("<", "")
    text.replace(">", "")
    return text
#FOR PMT
def extract_name_before_colon(chunks):
    result = []

    for chunk in chunks:
        index = chunk.find(":")
        if index != -1:
            sentence_before_colon = chunk[:index].strip()
            result.append(sentence_before_colon)

    return result

#FOR PMT
def separate_par_into_sentencesPMT(speaker):
    try:
        result = []
        current_sentence = ""
        for char in speaker.paragraph:
            current_sentence += char
            if char in ['!', '?', '.']:
                if char == ".":
                    if current_sentence.endswith("..."):
                        continue

                    # Check for a space or another period before the last character
                    if len(current_sentence) >= 3 and (current_sentence[-3] == " " or current_sentence[-3] == "."):
                        continue

                result.append(current_sentence)
                current_sentence = ""

        return result
    except Exception as e:
        return speaker

#FOR PMT
def separate_par_into_sentences(speaker):
    try:
        result = []
        current_sentence = ""
        for char in speaker.paragraph:
            current_sentence += char
            if char in ['!', '?', '.']:
                if char == ".":
                    if current_sentence.endswith("..."): # it causes much trouble, now we keep it on the side and then, the whole sentence will be deleted
                        continue

                    # Check for a space or another period before the last character
                    if len(current_sentence) >= 3 and (current_sentence[-3] == " " or current_sentence[-3] == "."):
                        continue

                result.append(current_sentence)
                current_sentence = ""

        return result
    except Exception as e:
        return speaker

#FOR PVT
def separate_par_into_sentences(speaker):
    try:
        result = []
        current_sentence = ""
        for char in speaker.paragraph:
            current_sentence += char
            if char in['!', '?', '.']:
                result.append(current_sentence)
                current_sentence = ""
        return result
    except Exception as e:
        return speaker

#FOR PMT
def is_hebrew(speaker):
    try:
        hebrew_letters=0
        counter = 0  #counter to know which sentence to delete
        sentences_to_remove = []
        hebrew = set("אבגדהוזחטיכךלמנסעפצקרשתםןףץ" + " ")
        for sentence in speaker.sentences:
            hebrew_letters = 0

            for char in sentence:
                if char in hebrew:
                    hebrew_letters += 1

            if "– – –" in sentence:
                sentences_to_remove.append(counter)
            elif "- - -" in sentence:
                sentences_to_remove.append(counter)
            elif" – –" in sentence:
                sentences_to_remove.append(counter)
            elif "--" in sentence:
                sentences_to_remove.append(counter)
            elif "..." in sentence:
                sentences_to_remove.append(counter)
            elif hebrew_letters == 0:
                sentences_to_remove.append(counter)
            speaker.sentences[counter] = sentence.strip("– ") #remove - from begging of sentence
            speaker.sentences[counter] = sentence.strip("- ") # almost the same
            counter += 1
        for index in reversed(sentences_to_remove):
            del speaker.sentences[index]
        return speaker
    except Exception as e:
        return speaker

#FOR PMT
def nameFilteringPMT(paragraph):
    try:
        listOfSpeakers = []
        for par in paragraph:
            splitted = par.split(":", 1)
            if len(splitted) == 2:
                listOfSpeakers.append(Speaker(splitted[0], splitted[1],protocolName,kinesset_number,protocolType))
        for speaker in listOfSpeakers:
            #print(speaker.name)
            speaker.name = speaker.name.strip()
            #print("##")
        ## UNTIL THIS STAGE EACH SPEAKER HAS THE ATTRIBUTES NAME AND PARAGRAPH

        ## FILTERS THE ( )
        for i in listOfSpeakers:
            filtered_name = i.name
            if (")" in filtered_name):
                start = 0
                end = 0
                while (filtered_name[start] != "("):
                    start += 1
                while (filtered_name[end] != ")"):
                    end += 1
                filtered_name = filtered_name[:start] + filtered_name[end + 1:]
                i.name = filtered_name


        # FILTERS THE SPEAKER's POSITION , E.G. Where we find "
        for i in listOfSpeakers:
            filtered_name = i.name
            unfiltered_name = i.name
            if ('"' in filtered_name):
                start = 0
                end = 0
                while start < len(filtered_name) and (filtered_name[start] != '"'):
                    start += 1
                while start < len(filtered_name) and (filtered_name[start-1] != " "):
                    start += 1
                filtered_name = filtered_name[start:]
                i.name = filtered_name
        for speaker in listOfSpeakers:
            speaker.sentences = separate_par_into_sentences(speaker)
        for speaker in listOfSpeakers:
            uf_name = speaker.name
            uf_name = uf_name.strip()
            spaces_allowed = 2
            for i in range(len(uf_name) - 1, -1, -1):
                if (uf_name[i] == " "):
                    spaces_allowed = spaces_allowed - 1
                if (spaces_allowed == 0):
                    uf_name = uf_name[i:len(uf_name)]
                    break
            speaker.name = uf_name

        return listOfSpeakers
    except Exception as e:
        return paragraph

#FOR PVT
def nameFiltering(paragraph):
    try:
        listOfSpeakers = []
        for par in paragraph:
            splitted = par.split(":", 1)
            if len(splitted) == 2:
                listOfSpeakers.append(Speaker(splitted[0], splitted[1], protocolName, kinesset_number, protocolType))
        for speaker in listOfSpeakers:
            #print(speaker.name)
            speaker.name = speaker.name.strip()
            #print("##")
        ## UNTIL THIS STAGE EACH SPEAKER HAS THE ATTRIBUTES NAME AND PARAGRAPH

        ## FILTERS THE ( )
        for i in listOfSpeakers:
            filtered_name = i.name
            if (")" in filtered_name):
                start = 0
                end = 0
                while (filtered_name[start] != "("):
                    start += 1
                while (filtered_name[end] != ")"):
                    end += 1
                filtered_name = filtered_name[:start] + filtered_name[end + 1:]
                i.name = filtered_name


        # FILTERS THE SPEAKER's POSITION , E.G. Where we find "
        for i in listOfSpeakers:
            filtered_name = i.name
            unfiltered_name = i.name
            if ('"' in filtered_name):
                start = 0
                end = 0
                while (filtered_name[start] != '"'):
                    start += 1
                while (start < len(filtered_name) and filtered_name[start] != " "):
                    start += 1
                filtered_name = filtered_name[start:]
                i.name = filtered_name
        for speaker in listOfSpeakers:
            speaker.sentences = separate_par_into_sentences(speaker)
        for speaker in listOfSpeakers:
            uf_name = speaker.name
            uf_name = uf_name.strip()
            spaces_allowed = 2
            for i in range(len(uf_name) - 1, -1, -1):
                if (uf_name[i] == " "):
                    spaces_allowed = spaces_allowed - 1
                if (spaces_allowed == 0):
                    uf_name = uf_name[i:len(uf_name)]
                    break
            speaker.name = uf_name


        return listOfSpeakers
    except Exception as e:
        return paragraph

# Replace '13_ptm_532058.docx' with your actual file
#FOR PMT
def excludeVoting(text):
    try:
        start_i = ""
        end_i = ""
        start_i = text.find("הצבעה מס")
        if(start_i == -1):
            return False, text
        end_i = start_i
        while(text[end_i] != '.'):
            end_i = end_i + 1
        end_i = end_i + 1
        return True,text[:start_i] + text[end_i:]
    except Exception as e:
        return text

def printFinalized(listOfSpeakers):
    for i in listOfSpeakers:
        print(i.name,":@@@@@@@@")
        for j in i.sentences:
            stripped_sentence = j.strip()
            print(stripped_sentence)

def excludeStartPTV(text,number = 0):
    try:
        start_index = text.find("סדר היום")
        if start_index != -1:
            text = text[start_index:]
            start_index = text.find('היו"ר')
            start_index = start_index - 1
            text = text[start_index:]

        elif (-1 != text.find("הישיבה")):
            start_index = text.find("הישיבה")
            if start_index != -1:
                text = text[start_index:]
                start_index = text.find('היו"ר')
                start_index = start_index - 1
                text = text[start_index:]
        if(number == 16 or number == 17):
            start_index = text.find("קצרנית")
            if start_index != -1:
                while(text[start_index] != ' '):
                    start_index = start_index + 1
                text = text[start_index:]
                start_index = text.find('היו"ר')
                start_index = start_index - 1
                text = text[start_index:]


        if(number == 16 or number == 17 or number == 18 or number == 19 or number == 20 or number == 23):
            start_index = text.find("רשמת פרלמנטרית")
            start_indexx = text.find("רישום פרלמנטרי")
            if(start_indexx > start_index):
                start_index = start_indexx
            if start_index != -1:
                while(text[start_index] != '\n'):
                    start_index = start_index + 1
                text = text[start_index:]
                start_index = text.find('היו"ר')
                start_indexx = text.find('היו”ר')
                if(start_indexx != -1 and start_indexx < start_index):
                    start_index = start_indexx
                start_index = start_index - 1
                text = text[start_index:]
        if (number == 16 or number == 17 or number == 18):
            start_index = text.find("רשמה")
            if start_index != -1:
                while (text[start_index] != '\n'):
                    start_index = start_index + 1
                text = text[start_index:]
                start_index = text.find('היו"ר')
                start_index = start_index - 1
                text = text[start_index:]

        if(number >= 19 ):
            text = text.replace('<< יור >>',"")
            text = text.replace("<< דובר >>", "")
            text = text.replace('(יש עתיד-תל"ם)',"")
            text = text.replace('<< אורח >>',"")
            text = text.replace('<< דובר_המשך >>',"")
            text = text.replace('<< קריאה >>',"")
            text = text.replace('<',"")
            text = text.replace('>', "")

        return text
    except Exception as e:
        return text

def is_not_hebrew(word):
    hebrew_letters = set("אבגדהוזחטיכךלמנסעפצקרשתםןףץ" + "0123456789 ")
    # Check if the word contains at least one non-Hebrew character
    return any(char not in hebrew_letters for char in word)


def tokenize_sentences(speaker):
    try:
        tokenized_sentences = []
        sentences_to_remove = []

        for index, sentence in enumerate(speaker.sentences):
            # Split the sentence into a list of words
            tokens = sentence.split()

            # Initialize a list to store the final tokens with separated punctuation
            final_tokens = []

            # Iterate through each token in the list
            for token in tokens:
                # Separate punctuation from words
                separated_tokens = [word.strip('.,!?()[]{}') for word in re.split(r'([.,!?()[]{}])', token) if word]

                # Check if any separated token is not Hebrew
                if any(is_not_hebrew(word) for word in separated_tokens):
                    sentences_to_remove.append(index)
                    break  # Break the loop if any non-Hebrew token is found

                # Add the separated tokens to the final list
                final_tokens.extend(separated_tokens)

            # Add the list of tokens to the tokenized_sentences
            tokenized_sentences.append(final_tokens)

        for index in reversed(sentences_to_remove):
            del speaker.sentences[index]

        # Add the tokenized sentences to the speaker object
        speaker.tokenized_sentences = tokenized_sentences
        """""
        sentences_with_whitespaces = []
        for tokens in speaker.tokenized_sentences:  # merges the tokens and puts whitespace between them in each sentence
            if len(tokens) >= 4:
                sentence = ' '.join(tokens)
                sentences_with_whitespaces.append(sentence)
        speaker.tokenized_sentences.clear()
        speaker.tokenized_sentences.append(sentences_with_whitespaces)
        """""
        return speaker
    except Exception as e:
        return speaker

if __name__ == "__main__":
    try:
        path = sys.argv[1]
        csv_path = sys.argv[2]
        totalSpeakers = []
        listOfSpeakers = []
        file_name = os.path.basename(path)
        doc = Document(path)
        text_from_doc = extract_text_from_docx(doc)
        protocolName, kinesset_number, protocolType = Section11(file_name)
        if (protocolType == "plenary"):
            excluded = (excludeStart(text_from_doc))  # Deletes Everything Before "תוכן העניינים"
            while True:
                r, excluded = excludeVoting(excluded)
                if (r == False):
                    break
            paragraphs_with_names = separate_text_by_patternPMT(excluded)
            listOfSpeakers = nameFilteringPMT(paragraphs_with_names)  # Filters names from positions & extra stuff
            for i in listOfSpeakers:
                i = is_hebrew(i)
                totalSpeakers.append(tokenize_sentences(i))
        else:
            text_from_doc = excludeStartPTV(text_from_doc, kinesset_number)
            text_from_doc = separate_text_by_pattern(text_from_doc)
            listOfSpeakers = nameFiltering(text_from_doc)  # Filters names from positions & extra stuff
            for i in listOfSpeakers:
                i = is_hebrew(i)
                totalSpeakers.append(tokenize_sentences(i))

    except Exception as e:
        print("Error occured")
        sys.exit(1)


    words_array = []
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["protocol_name", "knesset_number", "protocol_type", "speaker_name", "sentence_text"])
            try:
                for speaker in totalSpeakers:
                    for sentence in speaker.tokenized_sentences:
                        for word in sentence:
                            words_array.append(word)
                            if len(sentence) >= 4:
                                sentenceA = ' '.join(sentence)
                        writer.writerow(
                            [speaker.protocol_name, speaker.knesset_number, speaker.protocol_type, speaker.name, sentenceA])
            except KeyboardInterrupt:
                print("Data collection stopped by user.")
    except Exception as e:
        print("Error occured")
        sys.exit(1)



# TODO: Files that have ":" which are not names, are identified as names 058 & 066 (PMT)
# TODO: PMT 066 , Page 5, The same problem, sometimes the speaker can only be inferred logically which makes it hard for computers to identify
# TODO: 18_ptv_139299
# TODO: 19_ptv & 20_ptv show < > in names, even though it does not appear in the word document.
# TODO: In 23_ptv some texts start with underline bold and then ":" , cannot
# TODO: In conclusion, texts with 20,23 PTV are dog shit
# TODO: Speakers with 0 sentences were removed
# TODO: Some names have more than 2 spaces and we sometimes cut the name
# TODO: Check א. ב.
# TODO: When we have a time, like the last sentence in 18_ptv_139299.docx, it is considered a name, besha3a 10:40.