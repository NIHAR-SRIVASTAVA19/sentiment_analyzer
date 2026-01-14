import re
import nltk 
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import emoji

nltk.download('stopwords')
nltk.download('punkt')

# Get the standard list
stop_words = set(stopwords.words('english'))


# Words to keep because they change the meaning of the sentence
negation_words = {'not', 'no', 'never', 'neither', 'nor', 'but'}

# Remove negation words from stop_words
customized_stop_words = stop_words - negation_words



def preprocess_text(text):

    text=emoji.demojize(text)
    text=text.lower()
    pattern='[^a-z:]'
    text=re.sub(pattern, ' ', text)
    text=text.split()
    clean_text=[WordNetLemmatizer().lemmatize(word) for word in text if word not in customized_stop_words]
    processed_text=' '.join(clean_text)
    return processed_text