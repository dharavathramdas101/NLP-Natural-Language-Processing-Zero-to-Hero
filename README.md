## NLP-Natural-Language-Processing

### Step 1.Text Preprocessing Level-1 : 
Remove Special Characters, Convert Lowercase, Stop word removal, Tokenization, Lemmatization, Stemming
#### Description
In NLP, we have the text data, which our Machine Learning algorithms cannot directly use, so we have first to preprocess it and then feed the preprocessed data to our Machine Learning algorithms. So, In this step, we will try to learn the same basic processing steps which we have to perform in almost every NLP problem.
### Step 2. Advanced level Text Cleaning:
👉 Normalization,
👉 Correction of Typos, etc.

#### Description
These advanced-level techniques help our text data give our model better performance. Let’s take an advanced understanding of some of these techniques straightforwardly.
##### Normalization:
Map the words to a fixed language word.
For Example, according to human beings, Let’s have words like b4, and ttyl which can be understood as “before” and “talk to you later” respectively. Still, machines cannot understand these words the same way, so we have to map these words to a particular language word. This map is known as Normalization.

##### Correction of typos:
There are a lot of mistakes in writing English text or for other languages text, like Fen instead of a fan. The accurate map necessitates using a dictionary, which we used to map words to their correct forms based on similarity. Correction of typos is the term for this procedure.
### Step 3.Text Preprocessing Level-2(word to vec): 
👉 Bag Of Words(BOW), 
👉 TFIDF, 
👉 Unigrams, Bigrams, and Ngrams  
#### Description:
> All these are the primary methods to convert our Text data into numerical data (Vectors) to apply a Machine Learning algorithm to it. 
> These techniques are useful to store the meaning of the tokens in relationship with other tokens that will be useful for predictive machine learning models.

> Whenever we work with text data, we need numeric data so that the machine can understand. These methods are useful because they convert the text tokens to numeric values/vectors so that the machine learning models process this semantic information between the information.

#### Bag of Words (BOW) or Count Vectorizer
##### Models can be used:
Sklearn library provides the model as Count Vectorizer
Gensim library provides the model as Doc2vec
Bag of words is also called DTM i.e. Document Term Matrix, the vectors are stored in matrix form. The bag word is used to store the token’s vocabulary.

##### Pros of BOW:

> This process counts the frequency of tokens.
> The implementation is very easy.
> The classification and feature extraction applications can be based on this technique.
##### Cons of BOW:
> The tokens increases in the bag as the length of the data increases.
> The sparsity in the matrix will also increase as the size of the input data increases. The number of zero’s in the sparsity matrix is more than non-zero numbers.
> There is no relationship/semantic connection with each other because the text is split into independent words.
> Term Frequency-Inverse Document Frequency (TF-IDF)
There are two methods in TF-IDF in the sklearn library

##### TF-IDF Vectorizer and TF-IDF Transformer
The TF-IDF implementation tries to get information from the uncommon words.

##### Type of TF-IDF methods:

The output of the bag of words is used by TF-IDF Transformer and does the further process.
This TF-IDF Vectorizer method takes the raw data as input and does further process.
Pros of TF-IDF:

It slightly overcomes the semantic information between tokens.
##### Cons of TF-IDF:

This method gives chance for the model to overfit.
Not so much a semantic relationship between the tokens.
Use cases of Level BOW and TF-IDF

The classification model of machine learning can use these techniques.
Code Flow of BOW


Code Flow of TF-IDF


#### One Hot representation of Words

The encoding of the text can be done with the help of a one-hot method to map the text into numeric.

One-hot encoding representation in Text
Example: You are very Brave


The one-hot encoding makes the token into vectors. Each word gets its index position to represent different vectors for different words. This example shows only with 4 words but if the vocabulary size increases it increases the vector size increases.

##### Cons:
The more big-size vectors are complicated to train in the machine learning model.

### Step 4. Text Preprocessing Level-3:  
Word2vec, AvgWord2vec, Glove, Fast Text

#### Description
    All these are advanced techniques to convert words into vectors.
### Step 5. Solve Machine Learning Use cases
    Hands-on Experience on a use case
#### Description 
    After following all the above steps, now at this step, you can implement a typical or straightforward NLP use case using machine learning algorithms like Naive Bayes Classifier, etc. To have a clear understanding of all the above and understand the next steps.
Email Spam or Ham classification
### Step 6. Get an advanced level understanding of Artificial Neural Network
    While going much deeper into NLP, you do not take Artificial Neural Networks (ANN) very far from your view; you have to know about the basic deep learning algorithms, including backpropagation, gradient descent, etc.
### Step 7. Deep Learning Models:
RNN, LSTM RNN, GRU RNN 
    RNN is mainly used when we have the data sequence in hand, and we have to analyze that data. We will understand LSTM and GRU, conceptually succeeding topics after RNN.
### Step 8.Advanced Text Preprocessing level -4:
Word Embedding,
Word2Vec
    Now, we can do moderate-level projects related to NLP and makeproso in this domain. Below are some stepsthat will differentiate you from otherse who have also worked in this field. So, to take an edge over all those people learning these topics are a must.
### Step 9:
Bidirectional LSTM RNN, Encoders, and Decoders, Self Attention Models
    
### Step 10.
Transformers Learning in NLP
    The Transformer in NLP is an architecture that seeks to handle sequence-to-sequence tasks while handling long-range relationships with ease. It leverages self-attention models.
### Step 11:
👉 BERT(Bidirectional Encoder Representations from Transformers)

Description 
    It is a variation of the transformer, and it converts a sentence into a vector. It is a neural network-based technique used for natural language processing pre-training.
