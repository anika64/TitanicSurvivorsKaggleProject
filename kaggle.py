import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler




if __name__ == '__main__':
    df = pd.read_csv ('titanic/train.csv')
    #print (df[['Age']])
    #print(df.describe())
    np_train = np.array(df)
    # print(np_train)
    np_survived = np.array(df[['Survived']])
    #print(np_survived)
    survived_by_gender = df[['Survived', 'Sex']]
    # print(pd.pivot_table(df, index = 'Survived', values = ['Age', 'Fare']))
    #print(pd.pivot_table(df, index = 'Survived', columns = ['Sex']))
    # print(pd.pivot_table(df, index = 'Sex', values = ['Survived']))
    #print(pd.pivot_table(df, index = 'Fare'))
    ## got mean age (29.699118) by looking at df.describe()
    df_fix = df[['Survived', 'Fare', 'Age', 'SibSp', 'Parch', 'Sex', 'Pclass']].fillna(30)
    ## get_dummies = putting numerical for categorical data
    df_fix = pd.get_dummies(df_fix, columns = ['Pclass', 'Sex'], drop_first = True)
    #print(fixed_age)
    # plt.hist(fixed_age)
    # plt.show()
    #print(df.describe())
    print(df_fix)

    df_fix['Survived_cat'] = df_fix['Survived'].astype('category').map({1: '1', 0: '0'})
    boundary = 70 ## idk random num
    
    #sns.catplot(x = 'Fare', y = 'Survived_cat', data = df, order = ['1', '0'])
    #plt.plot([boundary, boundary], [-.2, 1.2], 'g', linewidth = 2)
    #plt.show()

    def boundary_classifier(target_boundary, fare_series):
        predictions = []
        for fare in fare_series:
            if fare > target_boundary:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    #y_pred = boundary_classifier(boundary, df_fix['Age'])
    #df_fix['predicted'] = y_pred
    #y_true = df_fix['Survived']
    #sns.catplot(x = 'Fare', y = 'Survived_cat', hue = 'predicted', data = df_fix, order = ['1', '0'])
    #plt.plot([boundary, boundary], [-.2, 1.2], 'g', linewidth = 2)
    #plt.show()

    

    ### REGRESSION MODEL
    
    train_df, test_df = train_test_split(df_fix, test_size = 0.2, random_state = 1)
    X = ['Age', 'Fare', 'SibSp', 'Parch', 'Sex_male', 'Pclass_2', 'Pclass_3']
    y = 'Survived'
    X_train = train_df[X]
    print(X_train.head())
    y_train = train_df[y]
    print(y_train.head())


    logreg_model = linear_model.LogisticRegression()
    logreg_model.fit(X_train, y_train)

    X_test = test_df[X]
    y_test = test_df[y]
    y_pred = logreg_model.predict(X_test)
    test_df['predicted'] = y_pred.squeeze()

    print('Accuracy of the Model on Training Data: ')
    print(logreg_model.score(X_train, y_train))

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of the Model on Test Data: ')
    print(accuracy)






     ##normalize
    ##ss = StandardScaler()
    #ft_values = ['Fare']
    #X_train[ft_values] = ss.fit_transform(X_train[ft_values])
    ##idek plotting something
    #sns.catplot(x = X[0], y = 'Survived_cat', hue = 'predicted', data = test_df, order = ['1', '0'])
    # plt.show()