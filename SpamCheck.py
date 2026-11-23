import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
path = r"C:\\Users\\umuty\\Downloads\\archive\\spam.csv"


df = pd.read_csv(path)
print(df.head())

print(df.isnull().sum())
print()


print(df['label'].value_counts())



# x is input --> training data
# y is output -> training results 

X = df['text']
Y = df['label_num']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(X_train)
print(X_test)




# model should use fit only on training set  --> fit_transform 
# test dataset -> transform 



vectorizer = TfidfVectorizer(stop_words="english") ## for the elimination of "the,is,etc..."

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

print("X training dataset shape :", X_train_vec.shape) # X training dataset shape : (4136, 44934)
print("X test dataset shape     :",X_test_vec.shape)   # X test dataset shape     : (1035, 44934)



# time for training 

model = MultinomialNB()

model.fit(X_train_vec,Y_train)


# time to test the model 


y_pred = model.predict(X_test_vec)

#y_test is the Answer Key for the y_pred

print(accuracy_score(Y_test,y_pred))

print()
print(classification_report(Y_test, y_pred))

# now it has 92% accuracy, but will it detect my own spam mail ? 

new_mails = ["Hey Umut, can you meet me at my office tomorrow morning? We need to talk",
             "Subject: Free money online! Save now and click http://www.getfreeprize.com to win cash"]

newmailvec = vectorizer.transform(new_mails)

predicts = model.predict(newmailvec)
for mail, tahmin in zip(new_mails, predicts):
    stat = "spam" if tahmin == 1 else "ham"
    print(f"\n message : {mail}")
    print(f"prediction: {stat}")


prob = model.predict_proba(newmailvec)

for mail,prob in zip(new_mails,prob):
    print(mail)
    print(f"{prob[0]*100:.2f}") # ham possiblity
    print(f"{prob[1]*100:.2f}") # spam possiblity



print()
# spam word chec 

words = np.array(vectorizer.get_feature_names_out())


coef = model.feature_log_prob_[1]

mostSpam = words[np.argsort(coef)[-15:]]

print(mostSpam)