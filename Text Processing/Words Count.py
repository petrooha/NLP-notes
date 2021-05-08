"""Count words."""
#from string import punctuation
import re

def count_words(text):
    """Count how many times each unique word occurs in text."""
    counts = dict()  # dictionary of { <word>: <count> } pairs to return
    
    # TODO: Convert to lowercase
    text = text.lower()
    # TODO: Split text into tokens (words), leaving out punctuation
    # (Hint: Use regex to split on non-alphanumeric characters)
    punctuation = "[\s.,!?:;'\"-]+"
    text = re.split(punctuation, text)
    text = [word for word in text if word != '']

    # TODO: Aggregate word counts using a dictionary
    for word in text:
        if word in counts:
            counts[word] +=1
        else:
            counts[word] = 1
    
    return counts

# Another way to remove punctuation:
words = re.sub(r"[^a-zA-Z0-9]", " ", text) # better to replace with space than to remove

# Another way to tokenize:
words = words.split()
# Smarter way to tokenize both words and sentences:
from nttk.tokenize import word_tokenize, sent_tokenize
words = word_tokenize(text)
sentences = sent_tokenize(text)


text = "As I was waiting, a man came out of a side room, and at a glance I was sure he must be Long John. His left leg was cut off close by the hip, and under the left shoulder he carried a crutch, which he managed with wonderful dexterity, hopping about upon it like a bird. He was very tall and strong, with a face as big as a ham—plain and pale, but intelligent and smiling. Indeed, he seemed in the most cheerful spirits, whistling as he moved about among the tables, with a merry word or a slap on the shoulder for the more favoured of his guests."

counts = count_words(text)
sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        
print("10 most common words:\nWord\tCount")
for word, count in sorted_counts[:10]:
    print("{}\t{}".format(word, count))

print("\n10 least common words:\nWord\tCount")
for word, count in sorted_counts[-10:]:
    print("{}\t{}".format(word, count))
