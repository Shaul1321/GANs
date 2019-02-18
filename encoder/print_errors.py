from collections import Counter
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib
from collections import defaultdict

with open("gen2.txt", "r") as f:

	gen = f.readlines()

with open("dev_wiki.txt", "r") as f:

	wiki = f.readlines()
	
count  = 0.
total_words = 0.
length_count_true = Counter()
length_count_false = Counter()
num_errors_count = Counter()
sents_by_lengths = Counter()

for sent1, sent2 in zip(gen,wiki):

	
	sent1 = sent1.split(" ")
	sent2 = sent2.split(" ")
	sent1 = sent1[:-1]
	sent2 = sent2[1:-1]
	sent1 = " ".join(sent1)
	sent2 = " ".join(sent2)
	sents_by_lengths[len(sent2.split(" "))] += 1
	
	if sent1 != sent2:
		print "ERROR"
		length_count_false[len(sent2.split(" "))] += 1
		
		for w in sent1.split(" "):
		
			if w not in sent2.split(" "):
			
				num_errors_count[len(sent2.split(" "))] += 1
	else:
		print "CORRECT"
		length_count_true[len(sent2.split(" "))] += 1
	print sent1
	print sent2
	print "-------------------------------------------"
		
lengths, counts = zip(*length_count_true.items())
bar_width = 0.35
plt.bar(lengths, counts, alpha = 0.4, label = "full reconstruction")
plt.xticks(np.array(range(len(lengths))) + bar_width, lengths)

lengths, counts = zip(*length_count_false.items())
bar_width = 0.35
plt.bar(lengths, counts, alpha = 0.4, color = "red", label = "error in reconstruction")
plt.xticks(np.array(range(len(lengths))) + bar_width, lengths)
plt.legend()
plt.xlabel("sentence length (words)")
plt.ylabel("# sentences")
plt.title("number of sentences vs. sentence lengths, correct and incorrect reconstruction")
plt.show()


lengths, num_words_diff = zip(*num_errors_count.items())
num_words_diff = list(num_words_diff)
for i, (length, diff) in enumerate(zip(lengths, num_words_diff)):
	print length, diff, type(num_words_diff)
	num_words_diff[i] = diff/(1. * sents_by_lengths[length])

print num_words_diff

bar_width = 0.35
plt.bar(lengths, num_words_diff, alpha = 0.4)
plt.xticks(np.array(range(len(lengths))) + bar_width, lengths)
plt.show()
print count / total_words
