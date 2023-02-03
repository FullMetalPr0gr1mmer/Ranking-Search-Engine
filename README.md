# Ranking-Search-Engine
The project has three parts:

<strong>First part :</strong><br><br>
<pre>  1.Read 10 files (.txt)</pre>
<pre>  2.Apply tokenization</pre>
<pre>  3.Apply Stop words</pre>
<strong>Secondpart : </strong>

Build positional index and displays each term as the following : <term, number of docs containing term;doc1: position1, position2 ... ;doc2: position1, position2 ... ;etc.>
image

Allow users to write phrase query on positional index and system returns the matched documents for the query.
Thirdpart :

Computeterm frequency for each term in each document.
Compute IDF for each term.
Displays TF.IDF matrix.
Compute cosine similarity between the query and matched documents.
Rank documents based on cosinesimilarity
image
