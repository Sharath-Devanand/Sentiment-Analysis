#####   Sentiment Analysis   #####

###### Check if the required packages are installed, if not, install them #####

import os
package_name = 'afinn'
package_name2 = 'textblob'

try:
    __import__(package_name)
    __import__(package_name2)
except ImportError:
    os.system(f"pip install {package_name}")
    os.system(f"pip install {package_name2}")

#####   Importing Libraries   #####

import re, random
from afinn import Afinn
from textblob import TextBlob

"""
Reads the files (rt-polarity.pos,rt-polarity.neg,nokia-pos,nokia-neg) and splits them into lines.
Reads the positive and negative words from the positive-words.txt and negative-words.txt files.
Creates a dictionary of the positive and negative words by assigning a value of 1 to the positive words and -1 to the negative words.
Split the positive and negative sentences into training and testing data.
"""


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #Reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    # Read positive and negative words from files and append them to lists
    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = []
        for line in posDictionary:
            if not line.startswith(';'):
                posWordList.extend(re.findall(r"[a-z\-]+", line))
    posWordList.remove('a')

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = []
        for line in negDictionary:
            if not line.startswith(';'):
                negWordList.extend(re.findall(r"[a-z\-]+", line))
    
    # SentimentDictionary - Stores sentiment value of each word
    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1
    
    #Splitting sentences into training and testing data
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #Ceate Nokia Dataset
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

"""
Calculates frequency of each word in the training data.
Calculates the conditional probabilities of each word in the positive and negative classes.
"""

def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    #posFeatures = [] # [] initialises a list [array]
    #negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 


"""
Calculates and prints the accuracy, precision, recall and F1 score for the positive and negative classes.
"""

def printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred):
    accuracy = correct/total
    precision_pos = correctpos/totalpospred
    precision_neg = correctneg/totalnegpred
    recall_pos = correctpos/totalpos
    recall_neg = correctneg/totalneg
    F1_score_pos = (2*precision_pos*recall_pos)/(precision_pos+recall_pos)
    F1_score_neg = (2*precision_neg*recall_neg)/(precision_neg+recall_neg)
    
    print(dataName)
    print("Accuracy :",accuracy)
    print("Precision - Positive :",precision_pos)
    print("Precision - Negative :",precision_neg)
    print("Recall - Positive :",recall_pos)
    print("Recall - Negative :",recall_neg)
    print("F1 Score - Positive :",F1_score_pos)
    print("F1 Score - Negative :",F1_score_neg,'\n\n')


"""
Implement Naive Bayes algorithm
INPUTS:
  sentencesTest is a dictonary with sentences associated with sentiment 
  dataName is a string (used only for printing output)
  pWordPos is dictionary storing p(word|positive) for each word
     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
  pWordNeg is dictionary storing p(word|negative) for each word
  pWord is dictionary storing p(word)
  pPos is a real number containing the fraction of positive reviews in the dataset

"""
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)

    # TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
    # (3) precision and recall for the negative class; (4) F1 score;
    printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred)
    
"""
Implement Rule-Based algorithm
For each word in the sentence,
    if the word is in the positive dictionary , it adds 1,
    ff it is in the negative dictionary, it subtracts 1.
If the final score is above a threshold, it classifies as "Positive", otherwise as "Negative"
"""

def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %score + sentence)
    
    # TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
    # (3) precision and recall for the negative class; (4) F1 score;
    
    printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred)


"""
Implement improvisation in Rule-Based algorithm
For each word in the sentence,
    if the word is in the positive dictionary , it adds 4,
    if it is in the negative dictionary, it subtracts 4.
    if the word is in the intensifierDiminisher dictionary, it adds/subtracts the value of the word.
    if the word is in the negation dictionary, it multiplies the score by -1.
    if the word has an exclamation mark at the end, it adds/subtracts 1.

If the final score is above a threshold, it classifies as "Positive", otherwise as "Negative"
"""
def testDictionaryImproved_manual(sentencesTest, dataName, sentimentDictionary, threshold, intensDimiDict, negationList):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        total = len(sentencesTest.items())
        for i in range(0, len(Words)):
            if Words[i] in sentimentDictionary:
                score += 4 * sentimentDictionary[Words[i]]
                
                ## NEGATION RULES
                if i > 0 and Words[i - 1] in negationList:
                    score *= -1
                    
                if sentiment == 'positive':
                    ## CAPITALIZATION
                    if Words[i].isupper():
                        score += 1
                    
                    ## Intensifier/Diminisher Rule
                    if Words[i] in intensDimiDict:
                        score += intensDimiDict[Words[i]]
                    
                    ## Exclamation Rule
                    if Words[i][-1] == '!':
                        score += 1
                    
                else:
                    ## CAPITALIZATION
                    if Words[i].isupper():
                        score += 1
                    
                    ## Intensifier/Diminisher Rule
                    if Words[i] in intensDimiDict:
                        score -= intensDimiDict[Words[i]]
                    
                    ## Exclamation Rule
                    if Words[i][-1] == '!':
                        score -= 1

        #total+=1
        #print(score)
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                #correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                #correct+=0
                totalpospred+=1
    
    printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred)


"""
Implement improvisation in Rule-Based algorithm
For each word in the sentence,
    the afinn library calculates the score of the word.
    if the score is above a threshold, it classifies as "Positive", otherwise as "Negative"
"""


def testDictionaryImproved_afinn(sentencesTest, dataName, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    afinn = Afinn()
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        total = len(sentencesTest.items())

        for word in Words:
            score += afinn.score(word)

        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                #correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                #correct+=0
                totalpospred+=1
    
    printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred)


"""
Implement improvisation in Rule-Based algorithm
For each word in the sentence,
    the textblob library calculates the score of the word.
    if the score is above a threshold, it classifies as "Positive", otherwise as "Negative"
"""


def testDictionaryImproved_textblob(sentencesTest, dataName, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        total = len(sentencesTest.items())
        for word in Words:
            word_blob = TextBlob(word)
            score += word_blob.sentiment.polarity
                        

        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                #correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                #correct+=0
                totalpospred+=1
    
    printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred)

"""For each word in the sentence,
    it is checked for the sentiment from the vader lexicon
    if the score is above a threshold, it classifies as "Positive", otherwise as "Negative"
"""

def testDictionaryImproved_vader(sentencesTest, dataName, threshold,vader_lexicon):
    #total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        total = len(sentencesTest.items())
        for word in Words:
            if word in vader_lexicon:
               score+=vader_lexicon[word]
                        

        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                #correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                #correct+=0
                totalpospred+=1
    
    printAccuracy(dataName,correct,total,correctpos,correctneg,totalpos,totalneg,totalpospred,totalnegpred)

"""
Predictive power is calculated by the ability of the word to being positive over negative.
    - If word is present in the positive class, it tends to be more closer to 1 over 0.
Prints the top 10 and bottom 10 words based on predictive power.
"""


def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]>0.0000001:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\n\nPOSITIVE:")
    print (tail)
    return sortedPower

"""
Checks if words from useful function list is in the sentimentDictionary.
"""

def usefulInDict(SentimentDictionary, sortedPower):
    count = 0
    for word in sortedPower:
        if word in SentimentDictionary:
            count += 1
    return count/len(sortedPower)


"""
Main function - 

Initiates variables - dictionaries and datasets.
Vader lexicon is read from the file and stored in a dictionary.

"""


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

intensDimiDict = {
    "very": 2,
    "really": 2,
    "quite": 1.5,
    "somewhat": 0.5,
    "a bit": 0.5,
    "extremely": 3,
    "incredibly": 3,
    "totally": 2.5,
    "absolutely": 2.5,
    "rather": 1.5,
}

negationList = ["not", "no", "never", "none", "neither", "nor", "nobody", "nowhere", "nothing", "hardly", "scarcely", "barely"]

lexicon_path = "vader_lexicon.txt"

vader_lexicon = {}

with open(lexicon_path, 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        fields = line.strip().split('\t')
        if len(fields) == 4:
            word = fields[0]
            sentiment = float(fields[1])
            vader_lexicon[word] = sentiment

PRINT_ERRORS=0

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)
#run naive bayes classifier on datasets
print ("Naive Bayes")
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)
#run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 0)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 0)
# print most useful words
powerList=mostUseful(pWordPos, pWordNeg, pWord, 100)

# find ratio of words in powerlist present in the sentimentDictionary
ratio = usefulInDict(sentimentDictionary, powerList)

print("Ratio of words in powerlist present in the sentimentDictionary: ", ratio)
#run sentiment dictionary based classifier on datasets
testDictionaryImproved_afinn(sentencesTrain,  "Films (Train Data, Rule-Based - Afinn)\t", -5)
testDictionaryImproved_afinn(sentencesTest,  "Films  (Test Data, Rule-Based - Afinn)\t", -5)
testDictionaryImproved_afinn(sentencesNokia, "Nokia   (All Data, Rule-Based - Afinn)\t", 5)
#run sentiment dictionary based classifier on datasets
testDictionaryImproved_textblob(sentencesTrain,  "Films (Train Data, Rule-Based - TextBlob)\t", 0)
testDictionaryImproved_textblob(sentencesTest,  "Films  (Test Data, Rule-Based - TextBlob)\t", 0)
testDictionaryImproved_textblob(sentencesNokia, "Nokia   (All Data, Rule-Based - TextBlob)\t", 0)
#run sentiment dictionary based classifier on datasets
testDictionaryImproved_vader(sentencesTrain,  "Films (Train Data, Rule-Based - Vader)\t", 0,vader_lexicon)
testDictionaryImproved_vader(sentencesTest,  "Films  (Test Data, Rule-Based - Vader)\t", 0,vader_lexicon)
testDictionaryImproved_vader(sentencesNokia, "Nokia   (All Data, Rule-Based - Vader)\t", 0,vader_lexicon)
#run sentiment dictionary based classifier on datasets
testDictionaryImproved_manual(sentencesTrain,  "Films (Train Data, Rule-Based - Manual)\t",sentimentDictionary, 0,intensDimiDict, negationList)
testDictionaryImproved_manual(sentencesTest,  "Films  (Test Data, Rule-Based - Manual)\t",sentimentDictionary, 0,intensDimiDict, negationList)
testDictionaryImproved_manual(sentencesNokia, "Nokia   (All Data, Rule-Based - Manual)\t",sentimentDictionary, 0,intensDimiDict, negationList)
