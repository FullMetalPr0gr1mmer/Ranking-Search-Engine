import nltk
import string
import sys
import tkinter as tk
from tkinter import *
from collections import defaultdict
import math
from tkinter import ttk
import numpy as np
from itertools import chain
import os
import pandas as pd
from tabulate import tabulate
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


#######FIRST PART FUNCTIONS########
def readingFile(file):
    with open(file, 'r') as f:#f becomes the file obj
        words = f.read()
    f.close()
    return words

def Tokenize(words):
    wordsAfterTokenization=word_tokenize(words)
    return wordsAfterTokenization

def applyStopWords(words):
    stopWords=set(stopwords.words("english"))
    #print( stopWords)
    tokensAfterApplyingStopWords=[]
    for w in words:
        if w not in stopWords or (w=="in" or w=="to" or w=="where"):
            tokensAfterApplyingStopWords.append(w)
    return tokensAfterApplyingStopWords



#######SECOND PART FUNCTIONS########
filenames =['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt', '10.txt']

def buildingPositionalIndex():
    wordsDictionary={}
    for file in filenames:
        if file.endswith(".txt"):
            words=readingFile(file)
            tokens=Tokenize(words)
            tokensAfterApplyingStopWords=applyStopWords(tokens)
            wordPosition=0
            for word in tokensAfterApplyingStopWords:
                if word not in wordsDictionary:
                    wordsDictionary[word]=(word,[(file,[wordPosition + 1])])
                elif wordsDictionary[word][1][-1][0]==file :
                    wordsDictionary[word][1][-1][1].append(wordPosition + 1)
                else:
                    wordsDictionary[word][1].append((file,[wordPosition + 1]))
                wordPosition += 1
    positionalIndex={}
    to_be_Printed={}
    for word in wordsDictionary:
        positionalIndex[word]=[wordsDictionary[word][0],len(wordsDictionary[word][1]),wordsDictionary[word][1]]
        to_be_Printed[word]=[len(wordsDictionary[word][1]),wordsDictionary[word][1]]
    return positionalIndex,to_be_Printed
####GLOBAL VAR.#####
positionalIndex=buildingPositionalIndex()[0]

def Query(phraseQuery):
    try:

        phraseQuery=phraseQuery.lower()
        phraseQuery = Tokenize(phraseQuery)
        phraseQuery = applyStopWords(phraseQuery)
        answers=[]
        files=[]
        for token in phraseQuery:
            files.append(positionalIndex[token][2])

        if len(phraseQuery)==1:
            for f in files[0]:
                answers.append(f[0])
            return answers
        else:
            answers=rec(files[0],files[1:],len(phraseQuery))
            if len(answers)==0:
                return "Sorry, Query not found"
            else:
                return answers

    except:
      return "Sorry, Query not found"

def rec(files,TotalFiles,phraseLen):
    #print(files)

    TotalFiles=sum(TotalFiles, [])
    #print(TotalFiles)
    answers=[]
    for f in files:
        for p in f[1]:
            i=p+1
            c=1
            Flag=True
            for t in TotalFiles:
                if f[0]==t[0]:
                    if i in t[1]:
                        i+=1
                        c+=1
                    if c == phraseLen:
                        answers.append(f[0])
    return answers



#######THIRD PART FUNCTIONS########

def buildTf(): 
    Table = defaultdict(dict)
    for term in positionalIndex:
        record=positionalIndex[term]
        for f in filenames:
            for ff in record[2]:
                if f == ff[0]:
                    Table[term][f]=len(ff[1])
                    break
                else:
                    if f in Table[term].keys():
                        continue
                    else:
                        Table[term][f]=0


    return Table

def buildWeightedTf():
    tableTF = buildTf()
    tableWeightedTF= defaultdict(dict)

    for term in positionalIndex:
        for file in filenames:
            if tableTF[term][file]>0:
                tableWeightedTF[term][file]=1+math.log10(tableTF[term][file])
            else :
                tableWeightedTF[term][file] =0

    return tableWeightedTF

def buildDf():
    tableDf={}
    for term in positionalIndex:
        tableDf[term]=(positionalIndex[term][1],math.log10(len(filenames)/positionalIndex[term][1]))
    return tableDf


def buildTf_Idf():
    tableTF = buildWeightedTf()
    tableDf = buildDf()
    tableTf_Idf = defaultdict(dict)

    for term in positionalIndex:
        for file in filenames:
            tableTf_Idf[term][file]=tableTF[term][file]*tableDf[term][1]
    return tableTf_Idf

def buildDocLength():
    tableTf_Idf=buildTf_Idf()
    docLength={}
    for file in filenames:
        len=0
        for term in positionalIndex:
            len+=(math.pow(tableTf_Idf[term][file],2))
        docLength[file]=math.sqrt(len)
    return docLength


def buildNormalizedTf_Idf():
    normalizedTf_Idf= defaultdict(dict)
    docLength=buildDocLength()
    TtableTf_Idf=buildTf_Idf()
    for term in positionalIndex:
        for file in filenames:
            normalizedTf_Idf[term][file]=TtableTf_Idf[term][file]/docLength[file]
    return normalizedTf_Idf

def cosineSimilarity(phraseQuery):
    try:
        answers = Query(phraseQuery)
        phraseQuery = phraseQuery.lower()
        phraseQuery = Tokenize(phraseQuery)
        phraseQuery = applyStopWords(phraseQuery)
        cosineSimilarityQueryTable=defaultdict(dict)

        #headers=["TF","WeightedTF","IDF","WeightedTF*IDF","Normalized"]
        for term in phraseQuery:###BUILDING TF FOR QUERY
            if term in cosineSimilarityQueryTable.keys():
                cosineSimilarityQueryTable[term]["TF"]+=1
            else:
                cosineSimilarityQueryTable[term]["TF"]=1

        for term in phraseQuery:  ###BUILDING WeightedTF FOR QUERY
            if cosineSimilarityQueryTable[term]["TF"]>0:
                cosineSimilarityQueryTable[term]["WeightedTF"] = 1 + math.log10(cosineSimilarityQueryTable[term]["TF"])
            else:
                cosineSimilarityQueryTable[term]["WeightedTF"] = 0
        tableIDF=buildDf()
        for term in phraseQuery:  ###BUILDING IDF FOR QUERY
            cosineSimilarityQueryTable[term]["IDF"]=tableIDF[term][1]

        for term in phraseQuery:  ###BUILDING TF*IDF FOR QUERY
            cosineSimilarityQueryTable[term]["WeightedTF*IDF"]=cosineSimilarityQueryTable[term]["IDF"]*cosineSimilarityQueryTable[term]["WeightedTF"]
        querylen=0
        for term in phraseQuery:
            querylen += (math.pow(cosineSimilarityQueryTable[term]["WeightedTF*IDF"], 2))
        querylen=math.sqrt(querylen)

        for term in phraseQuery:
            cosineSimilarityQueryTable[term]["Normalized"]=cosineSimilarityQueryTable[term]["WeightedTF*IDF"]/querylen
        normaliztionTable=buildNormalizedTf_Idf()
        cosineAnswers=[]
        for f in answers:
            score=0
            for term in phraseQuery:
                score+=(cosineSimilarityQueryTable[term]["Normalized"]*normaliztionTable[term][f])
                cosineSimilarityQueryTable[term][f]=cosineSimilarityQueryTable[term]["Normalized"]*normaliztionTable[term][f]
            cosineAnswers.append((score,f))
            cosineSimilarityQueryTable["SUM:"][f]=score
        cosineSimilarityQueryTable["QueryLength:"]["TF"]=querylen
        cosineAnswers.sort(reverse=True)
        #print(cosineAnswers)
        return cosineSimilarityQueryTable,cosineAnswers
    except:
        return "Sorry, Query not found"



def printWeightedTf():
    wtf=pd.DataFrame(buildWeightedTf())
    print(tabulate(wtf.T, headers="keys"))


def printDf():
    df=pd.DataFrame(buildDf())
    print(tabulate(df.T, headers=["DF","IDF"]))

def printTf():
    tf = pd.DataFrame(buildTf())
    print(tabulate(tf.T, headers="keys"))

def printTf_Idf():
    tf_idf = pd.DataFrame(buildTf_Idf())
    print(tabulate(tf_idf.T, headers="keys"))

def printDocLength():
    docLength = pd.DataFrame.from_dict(buildDocLength(), orient='index')
    print(tabulate(docLength,headers=["Documeny","DocLength"]))

def printNormalizedTf_Id():
    normalizedTf_Idf = pd.DataFrame(buildNormalizedTf_Idf())
    print(tabulate(normalizedTf_Idf.T, headers="keys"))

def printCosineSimilarity(cs):
    cs=pd.DataFrame(cs)
    cs=cs.round(6)
    cs=cs.fillna('')

    print(tabulate(cs.T, headers="keys"))

def printPositionalIndex():
    df = pd.DataFrame(buildingPositionalIndex()[1])
    print(tabulate(df.T,headers=["Term","DF","Files -> Positions"]))



#########FOR CONSOLE ############
'''
print("Positional Index:")
printPositionalIndex()
print("")
print("")
print("")

print("TERM FREQUENCY:")
printTf()
print("")
print("")
print("")
print("WEIGHTED TERM FREQUENCY:")
printWeightedTf()
print("")
print("")
print("")
print("DOCUMENT FREQUENCY - INVERSED DOCUMENT FREQUENCY:")
printDf()
print("")
print("")
print("")
print("TF*IDF:")
printTf_Idf()
print("")
print("")
print("")
print("Document Length:")
printDocLength()
print("")
print("")
print("")
print("Normalized TF*IDF:")
printNormalizedTf_Id()


while True:
    print("")
    print("")
    print("")
    print("Please enter your Query:")
    pQuery = input()
    answers = Query(pQuery)
    if answers == "Sorry, Query not found":
        print(answers)
    else:
        print("Cosine Similarity for The Queries:")
        tableCS = cosineSimilarity(pQuery)
        print(f"Results:{answers}")
        CA = []
        for t in tableCS[1]:
            CA.append(t[1])
        print(f"Results based on Cosine Similarity:{CA}")
        printCosineSimilarity(tableCS[0])
    print("Do you want to insert another query (yes/no):")
    again = input().lower()
    if again=="no":
        break
'''


#########GUI##################
win= Tk()
win.geometry("1000x750")

def display_text():
   global Query_input
   x=250                        ####Displays Documents when button is clicked
   y=150
   s=Query_input.get()
   answers = Query(s)
   if answers=="Sorry, Query not found":
       label.configure(text=answers)
       label2.configure(text=answers)
   else:
       tableCS = cosineSimilarity(s)
       label.configure(text=answers)
       CA = []
       for t in tableCS[1]:
           CA.append(t[1])
       label2.configure(text=CA)
   #####Calls for Cosine Function

       printCosineSimilarity(tableCS[0])
       print("")
       print("")
       print("")




label=Label(win, text="", font=("Courier 10 bold"))
label.pack()
label.place(x=258,y=150)
label2=Label(win, text="", font=("Courier 10 bold"))
label2.pack()
label2.place(x=408,y=170)
Result=Label(win,text="Results:")
Result.pack()
Result.place(x=208,y=150)
Result2=Label(win,text="Results based on Cosine Similarity:")
Result2.pack()
Result2.place(x=208,y=170)

Query_input= Entry(win, width = 60)                   #############TAKES QUERY
Query_input.focus_set()
Query_input.pack()
Query_input.place(relx=0.5,y=100,anchor='center')
Please_Enter_Query = Label(win,text = "Please Enter Query :").place(x=478,y = 100,anchor='center')
ttk.Button(win, text= "Search",width= 20, command= display_text ).place(relx=0.5,y=130,anchor='center')


#############DISPLAY DIFFERENT TABLES###############
txt = Text(win,width=130)
txt.pack()
txt.place(relx=0.5,y=400,anchor='center')
class PrintToTXT(object):
 def write(self, s):
     txt.insert(END, s)
sys.stdout = PrintToTXT()
print("POSITIONAL INDEX:")
printPositionalIndex()
print("")
print("")
print("")
print("TERM FREQUENCY:")
printTf()
print("")
print("")
print("")
print("WEIGHTED TERM FREQUENCY:")
printWeightedTf()
print("")
print("")
print("")
print("DOCUMENT FREQUENCY - INVERSED DOCUMENT FREQUENCY:")
printDf()
print("")
print("")
print("")
print("TF*IDF:")
printTf_Idf()
print("")
print("")
print("")
print("Document Length:")
printDocLength()
print("")
print("")
print("")
print("Normalized TF*IDF:")
printNormalizedTf_Id()
print("")
print("")
print("")
print("Cosine Similarity for The Queries:")
win.state('zoomed')
win.mainloop()
