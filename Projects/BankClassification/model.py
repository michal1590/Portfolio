# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:46:34 2018

model

@author: Michal
"""

#%%############################################################################
#########################ladowanie bibliotek###################################
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale





###############################################################################
#################### funkcje ##################################################
###############################################################################

def rysujHistogramy(df,fig):
        
    
    for i,col in enumerate(df.select_dtypes(exclude=['category'])):
        ax=fig.add_subplot(4,4,i+1)
        
        x_max = max(df.loc[wyb_poz,col].max(),df.loc[wyb_neg,col].max())
        x_min = min(df.loc[wyb_poz,col].min(),df.loc[wyb_neg,col].min())
        
        ax.hist(df.loc[wyb_neg,col],alpha=0.3,label='odrzucenie', density=True,color='r',
                  range=(x_min, x_max))
        ax.hist(df.loc[wyb_poz,col],alpha=0.3,label='akceptacja', density=True,color='c',
                range=(x_min, x_max))
        
        ax.set_xlabel(col)
    plt.tight_layout()    
    fig.legend(loc='lower right')
    
    
    
    
    
    
    
def rysujSlupki(df,fig):
    
    for i, col in enumerate(df.select_dtypes(include=['category'])):
        ax = fig.add_subplot(2,5,i+1)
        
        temp1 = df.loc[wyb_poz,col].value_counts(sort=False) / sum(
                df.loc[wyb_poz,col].value_counts())
        
        temp0 = df.loc[wyb_neg,col].value_counts(sort=False) / sum(
                df.loc[wyb_neg,col].value_counts())
        
        ax.bar(temp0.index,temp0,color='r',alpha=0.3,label='odrzucenie')
        ax.bar(temp1.index,temp1,color='c',alpha=0.3,label='akceptacja')
        
        ax.set_title(col)
        ax.set_xticklabels(temp1.index,rotation='vertical')
        
    plt.tight_layout()    
    fig.legend(loc='lower right')    






def legendaTrans(legenda):
    
    wynik = []
    dict1 = {0:'Odrzucenie. ', 1:'Akceptacja. ' }
    dict2 = {'success':'Klient banku','failure':'Jedynie kontakt',
             'nonexistent':'Brak kontaktu'}
    f = lambda x: (dict1[x[0]]+dict2[x[1]])
    for data in legenda:
        wynik.append(f(data))
        
    return wynik





def plot_learning_curve(estymator, tytul, X, y, ylim,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(tytul)
  
    plt.xlabel("Rozmiar probki")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estymator, X, y, train_sizes=train_sizes,  scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt









#%%############################################################################
#######################przetwarzanie wstepne###################################
###############################################################################


#ładowanie danych
df = pd.read_csv('bank-additional/bank-additional-full.csv',sep=';',na_values='unknown')


#podstawowe informacje 
df.info()
df.describe()

df.drop(['duration'],axis=1, inplace=True)

#zamiana typow
df.default = df.default.map({'yes':1, 'no':0})
df.housing = df.housing.map({'yes':1, 'no':0})
df.loan = df.loan.map({'yes':1, 'no':0})
df.y = df.y.map({'yes':1, 'no':0})

for col in df.select_dtypes(include=['object']):
    df[col] = df[col].astype('category')


#uzupelnianie N/A
df.job = df.job.apply(lambda x: 'admin.' if pd.isnull(x)==True else x )
df.education = df.education.apply(lambda x: 'university.degree' if pd.isnull(x)==True else x )

df.fillna(df.mode().iloc[0],inplace=True)

#przydatne 
wyb_poz = df.y==1
wyb_neg = df.y==0







#%%############################################################################
#######################wizualizacja i analiza##################################
###############################################################################

plt.close('all')

#podstawowa prezentacja danych
histogramy = plt.figure('Histogramy')
rysujHistogramy(df, histogramy)

slupki = plt.figure('Slupki')
rysujSlupki(df, slupki)


#%%histogram pdays jest zaklamany prez rekordy z wartoscia 999

plt.figure('pdays bez 999')
df.pdays[(df.pdays<999) & (wyb_neg)].hist(bins=10,alpha=0.3, color='r', 
         density = True, label='odrzucenie', range=(0,27))
df.pdays[(df.pdays<999) & (wyb_poz)].hist(bins=10,alpha=0.3, color='c', 
         density = True, label='akceptacja')

plt.title('Wplyw ilosci dni ktore uplynely od czasu ostaniego kontaktu na  \n' +
          'decyzje klientow(jedynie ci ktorzy mieli kontakt z bankiem w poprzednich kampaniach')
plt.xlabel('Liczba dni od ostatniego kontaktu w poprzedniej kampanii')
plt.legend()


# procent klientow ktorzy zaakceptowali oferte
print('Procent klientow ktorzy zaakceptowali oferte, majac do czynienia z ' +
      'bankiem poprzednio: {0:.2f}%'.format( 
      sum((df.poutcome!='nonexistent') & (wyb_poz)) / sum(df.poutcome!='nonexistent')*100)  )

#dla kontrastu 
print('Procent klientow ktorzy zaakceptowali oferte, nie majac do czynienia z '+
      'bankiem poprzednio: {0:.2f}%'.format( 
      sum((df.poutcome=='nonexistent') & (wyb_poz)) / sum(df.poutcome=='nonexistent')*100)  )


#Sprawdzmy czy sam fakt wczesniejszego kontaktu byl decydujacy, a moze istotniejszy
# jest wynik (czy osoba otworzyla lokate) w poprzedniej kampanii
print('Procent klientow ktorzy zaakceptowali oferte, majac juz wczesniej ' +
      'lokate w banku: {0:.2f}%'.format( 
      sum((df.poutcome=='success') & (wyb_poz)) / sum(df.poutcome=='success')*100 ))

print('Procent klientow ktorzy zaakceptowali oferte, nie majac wczesniej ' +
      'lokaty w banku, ale mieli kontakt w poprzednich kampaniach: {0:.2f}%'.format( 
      sum((df.poutcome=='failure') & (wyb_poz)) / sum(df.poutcome=='failure')*100 ))


#porownanie liczby telefonow wykonywanych do klientow
data_box = []
legenda = []
for i in range(2):
    for j in ['success','failure','nonexistent']:
        data_box.append(df.campaign[(df.y==i) & (df.poutcome==[j])])
        legenda.append( (i,j) )

legenda = legendaTrans(legenda)     
plt.figure('Liczba telefonow vs historia klienta')   
plt.boxplot(data_box, labels=legenda, sym='', meanline=True, showmeans=True)
plt.title('Wyniki kampanii w zaleznosci od historii klienta')        
plt.ylabel('Liczba wykonanych telefonow')



#%%
#sprawdzmy czy wykluczajac ludzi ktorzy mieli juz kontakt z bakiem, beda widoczne
#jakies prawidlowosci

hist2 = plt.figure('Histogramy dla klientow bez historii kontaktow')
rysujHistogramy(df[df.poutcome=='nonexistent'], hist2)

slupki2 = plt.figure('Slupki dla klientow bez historii kontaktow')
rysujSlupki(df[df.poutcome=='nonexistent'], slupki2)


hist3 = plt.figure('Histogramy dla klientow z historia kontaktow')
rysujHistogramy(df[df.poutcome!='nonexistent'], hist3)

slupki3 = plt.figure('Slupki dla klientow z historia kontaktow')
rysujSlupki(df[df.poutcome!='nonexistent'], slupki3)



# %%
#zmiany danych makroekonomicznych 
grouped = df.groupby(['month'], sort=False).mean()
grouped = grouped[['campaign','euribor3m','nr.employed','y']]
grouped = (grouped-grouped.mean())/(grouped.max()-grouped.min())

grouped.plot.bar()
plt.title('Odchyl poszczegolnych parametrow od sredniej w zaleznosci od miesiaca')


#%%

#Z powyzszych wynika ze najbardzoej atrakcyjna grupa sa ludzie w wieku mniej niz
#30 lat lub powyzej 60, o statusie zawodowym student, emeryt lub pracujacy w dzialach
# administracyjnych. Preferowane powinno byc wyksztalcenie wyzsze. Mniejsze znaczenie
#z kolei ma stan cywilny (singiel), oraz dzien tygodnia (preferowany koniec)

idealni_poz = sum(   ( (df.age<=30) | (df.age>=60)) & 
                 ( (df.job=='student') | (df.job=='retired') | (df.job=='admin.')) &
                 ( df.education == 'university.degree' ) &
                 ( df.y==1 ) )

idealni_wszystkie = sum(   ( (df.age<=30) | (df.age>=60)) & 
                 ( (df.job=='student') | (df.job=='retired') | (df.job=='admin.')) &
                 ( df.education == 'university.degree' ) )


idealni_jedna_cecha = sum(   ( (df.age<=30) | (df.age>=60)) | 
                 ( (df.job=='student') | (df.job=='retired') | (df.job=='admin.')) |
                 ( df.education == 'university.degree' ) )

print('''Udzial ludzi spelniajacych idealne warunki w calkowitej liczbie lokat 
      to: {0:.2f}%, przy calkowitym udziale w populacji: {1:.2f}%. Skutecznosc
      rekrutacji wsrod grupy wyniosla {2:.2f}%, przy ogolnej skutecznosci {3:.2f}%'''.
      format( idealni_poz/sum(df.y==1)*100, idealni_poz/len(df)*100,
      idealni_poz/idealni_wszystkie*100, sum(wyb_poz)/len(wyb_poz)*100 ) )

#w jakiej mierze sa to ci sami ludzie (spelniajacy wszystkie wymagania)
print('''{0:.2f}% ludzi posiadajacych przynajmniej jedna ceche ze zbioru 
      idealnych, posiada rowniez pozostale cechy'''.format(
      idealni_wszystkie / idealni_jedna_cecha *100))






#%%#############################################################################
##########################model################################################
###############################################################################


#sprowadzenie wszystkich zmiennych do postaci liczbowej
df.month = df.month.map({'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,
                         'sep':9,'oct':10, 'nov':11, 'dec':12})

df.education = df.education.map({'illiterate':0,'basic.4y':1,'basic.6y':2,
 'basic.9y':3, 'professional.course':4, 'high.school':5, 'university.degree':6})

df.day_of_week = df.day_of_week.map({'mon':1,'tue':2,'wed':3,'thu':4,'fri':5}) 

df.pdays=pd.cut(df.pdays,[0,5,10,15,20,25,999],[5,10,15,20,25,999]).cat.codes

df = pd.get_dummies(df,columns=['job','marital','contact','poutcome'])


#%% wizualizacja ze wspolrzednymi rownoleglymi
plt.figure('wspolrzedne rownolegle')

pd.plotting.parallel_coordinates(pd.DataFrame(scale(df[::3]),columns=df.columns),
                                 'y', alpha=0.01, color=['r','c'])

plt.xticks(rotation='vertical')
plt.ylim((-10, 10))
plt.tight_layout()    


#%%korelacja pomiedzy zmiennymi

plt.figure('macierz korelacji')
plt.imshow(df.corr(), cmap=plt.cm.plasma, interpolation='nearest')
plt.colorbar()

tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)


#%% odrzucanie cech

target = df.y
zbedne_cechy = ['y','default', 'campaign', 'loan', 'housing', 'nr.employed', 'emp.var.rate',]
df.drop(zbedne_cechy,axis=1,inplace=True)


#%% wybor modelu
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



modele = dict()

modele['SVC'] = [SVC(),{'C':[0.1,0.3,1,], 'kernel':['rbf'] } ]   

modele['LogisticRegression'] = [LogisticRegression(), {'C':[0.1,0.3,1,3,10],
      'solver':['saga'], 'max_iter':[1000]}]
        
modele['RandomForestClassifier'] = [RandomForestClassifier(),{'n_estimators':[10,20],
      'criterion':['gini','entropy'],'max_depth':[3,7,10], 'min_samples_split':[7,20],
      'min_samples_leaf':[3,10]}]    

modele['DecisionTreeClassifier'] = [DecisionTreeClassifier(), {'criterion':['gini','entropy'],
      'max_depth':[3,7,10], 'min_samples_split':[7,20],'min_samples_leaf':[3,10]}]



X_train, X_test, y_train, y_test = train_test_split(df,target, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

wyniki=dict()
best_model = 0
best_score = 0

for model in modele:
    print(model)

    classifier = GridSearchCV(modele[model][0], modele[model][1], scoring='f1',
                              verbose=True)
    classifier.fit(X_train, y_train)
    
    wyniki[model] = [classifier.score(X_test,y_test),classifier.best_score_, 
          classifier.best_params_]
    
    if wyniki[model][0] > best_score:
        best_score = wyniki[model][0]
        best_model = model


#%%learning curve
estimator = modele[best_model][0].set_params(**wyniki[best_model][2])
plot_learning_curve(estimator,best_model, X_train, y_train, ylim=(0.7, 1.01))


#%%istotnosc cech
estimator.fit(X_train,y_train)
waznosc_cech = estimator.feature_importances_
