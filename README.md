# Ranking-Search-Engine
The project has three parts:

**First part :**
1. Read 10 files (.txt)
2. Apply tokenization 
3. Apply Stop words 

**Secondpart :** 
1. Build positional index  and displays each term as the following :
<term, number of docs containing term;doc1: position1, position2 ... ;doc2: position1, position2 ... ;etc.>


![image](https://user-images.githubusercontent.com/68661572/148378827-c0562c13-9b18-4938-b84f-05134ae31b75.png)


2. Allow users to write phrase query on positional index and system returns the matched documents for the query.



**Thirdpart :** 
1. Computeterm frequency for each term in each document.
2. Compute IDF for each term.
3. Displays TF.IDF matrix.
4. Compute cosine similarity between the query and matched documents.
5. Rank documents based on cosinesimilarity

![image](https://user-images.githubusercontent.com/68661572/148379013-1fbcd1cc-e16d-471c-9627-633ef265401c.png)
