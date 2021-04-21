from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk import ngrams
import operator
from nltk.stem import WordNetLemmatizer
import nltk
import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Jake Gillenwater
# CSCI-5260-940
# Artifical Intelligence
# Dr. Brian Bennett
# 4/20/21

# ==========================================================
#                          Part 1
# ==========================================================

# Step 1 - Reading a corpus from the web
print("Reading Text and Tokenizing...")
# "Frankenstein" by Merry Shelley
response = urllib.request.urlopen("https://www.gutenberg.org/files/42324/42324-h/42324-h.htm")
html = response.read()

# Step 2 - Cleaning the html
soup = BeautifulSoup(html, "html5lib")
text = soup.get_text(strip=True)

# Step 3 - Tokenization
naive_tokens = [t for t in text.split()]
sent_tokens = sent_tokenize(text)
word_tokens = word_tokenize(text)

# ==========================================================
#                          Part 2
# ==========================================================

# Step 1 - Remove puncuation
print("Removing Puncuation...")
tokenizer = RegexpTokenizer(r'\w+')
regex_tokens = tokenizer.tokenize(text)

# Step 2 - Make everything lowercase
regex_tokens = [t.lower() for t in regex_tokens]


# ==========================================================
#                          Part 3
# ==========================================================

# Step 0 - Assume a list of tokens
tokens = regex_tokens

# Step 2 - Stop Word Removal
print("Removing Stop Words...")
clean_tokens = tokens[:]
sr = stopwords.words("english")
for token in tokens:
    if(token in sr):
        clean_tokens.remove(token)

# Step 1 - Obtaining word counts
print("Frequency Analysis (Part 3)...")
freq = nltk.FreqDist(clean_tokens)  # lowercase, non-punctuated tokens
print("Length of Unique Items:", len(freq.items()))

# rm for key, val in freq.items():
# rm     print(str(key) + ':' + str(val))
# rm     print("Length of Unique Items:", len(freq.items()))
# rm     freq.plot(20, cumulative=False)

# ==========================================================
#                          Part 4
# ==========================================================

# Step 0 - Assume a list of tokens
tokens = clean_tokens

# Step 1 - Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Step 2 - Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

print("Frequency Analysis (Part 4)...")
freq = nltk.FreqDist(lemmatized_tokens)
print("Length of Unique Items:", len(freq.items()))
# rm for key, val in freq.items():
# rm     print(str(key) + ':' + str(val))
# rm     print("Length of Unique Items:", len(freq.items()))
# rm     freq.plot(20, cumulative=False)

# ==========================================================
#                          Part 5
# ==========================================================

# Step 0 - Assume a list of tokens
tokens = lemmatized_tokens

# Step 1 - Parts of speech analysis
print("POS Analysis...")
pos = nltk.pos_tag(tokens)
pos_counts = {}
for key, val in pos:
    # rm print(str(key) + ':' + str(val))
    if val not in pos_counts.keys():
        pos_counts[val] = 1
    else:
        pos_counts[val] += 1

print(pos_counts)
# rm plt.bar(range(len(pos_counts)), list(pos_counts.values()), align='center')
# rm plt.xticks(range(len(pos_counts)), list(pos_counts.keys()))
# rm plt.show()


# ==========================================================
#                          Part 6
# ==========================================================

# Step 1 - Group together tri-grams
print("Tri-Grams...")
trigrams = ngrams(text.split(), 3)
# rm for gram in trigrams:
# rm     print(gram)


# ==========================================================
#                          Part 7
# ==========================================================
print("Document-Term Matrix...")
print("Processing 2nd document...")
# "The Last Man" by Merry Shelley
response = urllib.request.urlopen('https://www.gutenberg.org/cache/epub/18247/pg18247.html')
html = response.read()
soup = BeautifulSoup(html, "html5lib")
text2 = soup.get_text(strip=True)
print("Back to the DTM...")
docs = [text, text2]
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print("Instances of 'fear' in both documents:")
print(df["fear"])   # Show the count for this word in both documents
print("Instances of 'hope' in both documents:")
print(df["hope"])   # Show the count for this word in both documents
print(df)           # Show the full data frame
