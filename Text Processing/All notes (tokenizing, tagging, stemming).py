import requests

# Fetch a web page
r = requests.get("https://www.udacity.com/courses/att")
print(r.text)


import re
# Remove HTML Tags with RegEx
pattern = re.compile(r'<.*?>') # tags look like <...>
print(pattern.sub('', r.text)) # replace them with a blank

# OR

# Much better with bs4
from bs4 import beautifulSoup
soup = BeautifulSoup(r.text, 'html5lib')
print(soup.get_text())


# After looking up 'Source Page' and finding that needed content is identified by pairs of 'div's
# Find all course summaries
summaries = soup.find_all("div", class_="course-summary-card")
print(summaries[0])

# extract title surrounded by <a> tags, surrounded by <h3> tags
summaries[0].select_one("h3 a") # .get_text() to clean in it up and .strip() to clean the white spaces from both ends

# Extract description tagged with <div data-course-short-summary='">
summaries[0].select_one("div[data-course-short-summary]").get_text().strip()

# Find all courses summaries
courses = []
summaries = soup.find_all("div", class_="course-summary-card")
for summary in summaries:
    title = summary.select_one("h3 a").get_text.strip()
    description = summary.select_one("div[data-course-short-summary]").get_text().strip()
    courses.append((title, description))
print(len(courses), "course summaries found. Sample:")
print(courses[0][0])
print(courses[0][1])







# Another way to remove punctuation:
words = re.sub(r"[^a-zA-Z0-9]", " ", text) # better to replace with space than to remove

# And 1 more
import string

words = text.translate(str.maketrans('', '', string.punctuation)))






# Another way to tokenize:
words = words.split()

# Smarter way to tokenize both words and sentences:
from nltk.tokenize import word_tokenize, sent_tokenize
words = word_tokenize(text)
sentences = sent_tokenize(text)






# Removing Stop words
from nltk.corpus import stopwords
print(stopwords.words("english"))

words = [w for w in words if w not in stopwords.words("english")] # words are normalized and tokenized
print(words)



# Tag parts of speach
from nltk import pos_tag
sentence = word_tokenize("I always lie down to tell a lie.")
pos_tag(sentence)


# Named Entity Recognition
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

ne_chunk(pos_tag(word_tokenize("Antonio joined Udacity Inc. in California.")))



# Stemming
from nltk.stem.porter import PorterStemmer

# Reduce words to their stem
stemmed = [PorterStemmer().stem(w) for w in words]
print(stemmed)

# OR Reduce to their root form (using dictionary)
from nltk.stem.porter import WordNetLemmatizer

lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed] # for verbs
print(lemmed)
lemmed 
