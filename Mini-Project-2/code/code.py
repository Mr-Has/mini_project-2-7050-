#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from collections import defaultdict
import re


# # Cleaning function:

# In[2]:


def preprocess_string(str_arg):
    
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) 
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) 
    cleaned_str=cleaned_str.lower() 
    
    return cleaned_str 


# # Naive Bayes from scratch:

# In[3]:


class NaiveBayes:
    
    def __init__(self,unique_classes):
        self.classes=unique_classes
        

    def addToBow(self,example,dict_index):
        
        if isinstance(example,np.ndarray): example=example[0]
            
        for token_word in example.split():
            self.bow_dicts[dict_index][token_word]+=1
            
            
    def fit(self,dataset,labels):
        
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
            
        for cat_index,cat in enumerate(self.classes):
            all_cat_examples=self.examples[self.labels==cat] 
            cleaned_examples=[preprocess_string(cat_example) for cat_example in all_cat_examples]
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            np.apply_along_axis(self.addToBow,1,cleaned_examples,cat_index)
            
        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        
        
        for cat_index,cat in enumerate(self.classes):
            prob_classes[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            count=list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1                                
            all_words+=self.bow_dicts[cat_index].keys()
                                                        
        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
                                                                     
        denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])  
        
        self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],
                         denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]  
        self.cats_info=np.array(self.cats_info)                                 
                                              
                                              
    def getExampleProb(self,test_example):  
        
        likelihood_prob=np.zeros(self.classes.shape[0]) 
        for cat_index,cat in enumerate(self.classes):                  
            for test_token in test_example.split():                           
                test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1                           
                test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])                              
                likelihood_prob[cat_index]+=np.log(test_token_prob)
        post_prob=np.empty(self.classes.shape[0])
        
        
        for cat_index,cat in enumerate(self.classes):
            post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])            
        return post_prob
    
   
    def predict(self,test_set):
        
        predictions=[] 
        for example in test_set:                                  
            cleaned_example=preprocess_string(example)                               
            post_prob=self.getExampleProb(cleaned_example) 
            predictions.append(self.classes[np.argmax(post_prob)])     
        return np.array(predictions) 


# # Training data-set:

# In[4]:


training_set=pd.read_csv ("train.csv")
training_set.info()
training_set.shape


# In[5]:


training_set.head()


# # Prediction on train data-set:

# In[6]:


y_train=training_set['subreddit'].values
x_train=training_set['comment'].values


from sklearn.model_selection import train_test_split
train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,
                                                               shuffle=True,
                                                               test_size=0.2
                                                               ,random_state=1
                                                               ,stratify=y_train)
classes=np.unique(train_labels)

nb=NaiveBayes(classes)
nb.fit(train_data,train_labels)

y_pred_train=nb.predict(test_data)
test_acc=np.sum(y_pred_train==test_labels)/float(test_labels.shape[0])

print ("Test Set Accuracy: ",test_acc) 


# # Testing data-set:

# In[7]:


testing_set=pd.read_csv ("test.csv")
testing_set.info()
testing_set.shape


# In[8]:


testing_set.head()


# # Prediction on test data-set:

# In[9]:


X_test=testing_set.comment.values

y_pred_test=nb.predict(X_test) 


# # Submission:

# In[10]:


submission = zip(list(range(len(y_pred_test))), y_pred_test)
test_df = pd.DataFrame(submission, columns=['Id','Category'])
test_df.to_csv('submission.csv', index = False, header=True)


# # Naive Bayes cross validation by using KFold:

# In[11]:


from sklearn.model_selection import KFold

X = x_train
y = y_train
kf = KFold(n_splits=5, random_state=42, shuffle=True)

print(kf)


# In[12]:


main_cross_val_accuracy = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    nb.fit(X_train,y_train)
    y_pred_cross_train=nb.predict(X_test)
    test_cross_acc=np.sum(y_pred_cross_train==y_test)/float(y_test.shape[0])
    main_cross_val_accuracy.append(test_cross_acc)


# In[13]:


main_cross_val_accuracy


# ###############################################################################################################################

# # Preparing the data for selective classifiers:

# In[14]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(training_set):
    
    all_comments = list()
    lines = training_set["comment"].values.tolist()
    for text in lines:
        text = text.lower()
        
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub("", text)       
    
        
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        
        tokens = word_tokenize(text)
        
        table = str.maketrans('', '', string.punctuation)
        
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        
        words = [w for w in words if not w in stop_words]
        words = ' '.join(words)
        
        all_comments.append(words)
    return all_comments

all_comments = clean_text(training_set)
all_comments[0:2]


# # Most frequent used words

# In[15]:


from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

c = all_comments
filtered_sentence = [] 
freq_count_limit = FreqDist()
lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

for i in c:
    comment_tokens = word_tokenize(i)
    
    for words in comment_tokens:
        if words not in stop_words: 
            filtered_sentence.append(words) 
        
            limit_words = lemmatizer.lemmatize(words)
#     for word in root_words:
            freq_count_limit[limit_words.lower()]+=1
freq_count_limit


# In[16]:


import matplotlib.pyplot as plt

freq_count_limit.plot(30,cumulative=False)
plt.show()


# # Vectorizing and transforming the text:

# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(ngram_range=(1,1), max_features=30000, strip_accents='ascii')
vect.fit(all_comments)
vocabulaire = vect.get_feature_names()


# In[18]:


bag_of_words = vect.transform(all_comments)
bag_of_words.shape


# # Random Forest:

# In[19]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=15, random_state=42)
clf_rf.fit(bag_of_words, training_set['subreddit'])


# In[20]:


clf_rf.score(bag_of_words, training_set['subreddit'])


# ### Random Forest - Cross Validation:

# In[21]:


from sklearn.model_selection import cross_val_score

scores_rf = cross_val_score(clf_rf, bag_of_words, training_set['subreddit'], cv=5)
scores_rf


# # Logistic Regression:

# In[22]:


from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(max_iter=10, random_state=42)
clf_lr.fit(bag_of_words, training_set['subreddit'])


# ### Logistic Regression - Cross Validation:

# In[23]:


clf_lr.score(bag_of_words, training_set['subreddit'])


# ### Logistic Regression - Cross Validation:

# In[25]:


scores_lr = cross_val_score(clf_lr, bag_of_words, training_set['subreddit'], cv=5)
scores_lr


# In[ ]:




