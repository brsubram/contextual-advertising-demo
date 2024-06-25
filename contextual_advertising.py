import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')


def download_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading the page: {e}")


def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    return ' '.join(chunk for chunk in chunks if chunk)


def extract_keywords(text):
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
    X = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores


def main(url):
    html = download_html(url)
    cleaned_text = clean_html(html)
    keywords = extract_keywords(cleaned_text)
    print("Keywords and their scores:")
    for keyword, score in keywords:
        print(f"{keyword}: {score:.2f}")


if __name__ == "__main__":
    url = input("Enter the URL: ")
    main(url)
