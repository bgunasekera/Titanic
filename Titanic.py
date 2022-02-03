import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data['train'] = 1
test_data['train'] = 0

all_data = pd.concat([train_data,test_data])

train_data.head()
train_data.info()
train_data.describe()
train_data.describe().columns


#Numeric variables  
numeric = train_data[['Age','SibSp','Parch','Fare']]
categorical = train_data[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

for i in numeric.columns:
    plt.hist(numeric[i])
    plt.title(i)
    plt.show()

print(numeric.corr())
sns.heatmap(numeric.corr())


pd.pivot_table(train_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])

#Categorical variables  

for i in categorical.columns:
    sns.barplot(categorical[i].value_counts().index,categorical[i].value_counts()).set_title(i)
    plt.show()
  
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))


#Feature engineering

categorical.Cabin
train_data['cabin_multiple'] = train_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

train_data['cabin_multiple'].value_counts()


pd.pivot_table(train_data, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket' ,aggfunc ='count')


train_data['cabin_adv'] = train_data.Cabin.apply(lambda x: str(x)[0])


print(train_data.cabin_adv.value_counts())
pd.pivot_table(train_data,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count')


train_data['numeric_ticket'] = train_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
train_data['ticket_letters'] = train_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)

train_data['numeric_ticket'].value_counts()

pd.set_option("max_rows", None)
train_data['ticket_letters'].value_counts()

pd.pivot_table(train_data,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')

train_data.Name.head(50)
train_data['name_title'] = train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
train_data['name_title'].value_counts()

#create all categorical variables that we did above for both training and test sets 
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data 
all_data.Age = all_data.Age.fillna(train_data.Age.median())
all_data.Fare = all_data.Fare.fillna(train_data.Fare.median())

all_data.dropna(subset=['Embarked'],inplace = True)

all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()


all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

all_data.Pclass = all_data.Pclass.astype(str)

all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train']])


X_train = all_dummies[all_dummies.train == 1].drop(['train'], axis =1)
X_test = all_dummies[all_dummies.train == 0].drop(['train'], axis =1)

y_train = all_data[all_data.train==1].Survived
y_train.shape

#Random forrest 

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

#Logistic regression
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
