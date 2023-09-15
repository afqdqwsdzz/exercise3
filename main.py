import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
import string

# Read the file "Moby Dick" from the Gutenberg dataset
from nltk.corpus import gutenberg
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stop words and punctuation filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

# Part-of-speech tagging
tagged_tokens = pos_tag(filtered_tokens)

# Part-of-speech frequency analysis
pos_freq = FreqDist(tag for word, tag in tagged_tokens)
common_pos = pos_freq.most_common(5)

# Lemmatization
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to Noun

lemmatized_tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos)) for token, pos in tagged_tokens[:20]]

# Output results
print("Most common parts of speech:")
for pos, count in common_pos:
    print(f"{pos}: {count}")

print("Lemmatized tokens:")
print(lemmatized_tokens)