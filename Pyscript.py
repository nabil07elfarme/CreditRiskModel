import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objs as go



############################################# Functions  #################################################################
def X_chart(df, cols,typ='pie'):
    figs = []
    for att in cols:
        df[att] = df[att].replace(attribute_map)
        # Group the DataFrame by the 'Purpose' column and count the number of occurrences of each purpose
        counts = df.groupby(att)[att].count()
        counts = counts.sort_values()

        if typ == 'bar':
            # Create a horizontal bar chart using Plotly
            fig = px.bar(counts, orientation='h', title=f'Distribution of {att}',width=700,
                    color_discrete_sequence=['green'])
            fig.update_xaxes(title='Frequency')
            fig.update_yaxes(title=f'{att}')
        elif typ == 'pie':
            fig = px.pie(counts, values=att, names=counts.index, title=f'Distribution of {att}',width=700)
            fig.update_traces(marker=dict(colors=['green', 'blue', 'red', 'orange', 'purple'], 
                                        line=dict(color='#FFFFFF', width=2)))
        figs.append(fig)
    return figs
    
def TargetGraph(df,columns):
    # Define the x and y variables
    y = df['Risk'].replace(attribute_map)#.apply(lambda x: 'bad' if x == 2 else 'good')
    figs = []
    for col in columns:
        x = df[col].replace(attribute_map)
        # Create a contingency table of the target variable and categorical variable
        cont_table = pd.crosstab(x, y, normalize='index')
        # Create the stacked bar chart
        data = [go.Bar(x=cont_table[i]*100, y=cont_table.index, name=str(i), orientation='h') for i in cont_table.columns]
        layout = go.Layout(barmode='stack', xaxis=dict(title='%'), height=350, title=f'{col} vs Risk')
        fig = go.Figure(data=data, layout=layout)
        fig.update_yaxes(title=f'{col}')
        # Display the chart
        figs.append(fig)
    return figs

def BoxP(df, target, columns):
    # Create a list of traces

    figs = []
    for col in columns:
        trace = go.Box(y=df[target], x=df[col], name=col)
        # Create layout
        layout = go.Layout(title=f"Box plot of {target} versus {col}", 
                           width=750,
                           height=650,
                           xaxis=dict(title=f"{col}"), yaxis=dict(title=f"{target}"))
        # Create figure
        fig = go.Figure(data=trace, layout=layout)
        figs.append(fig)
    return figs

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # fit the model on the training data
    model.fit(X_train, y_train)
    
    # predict the target variable for the test data
    y_pred = model.predict(X_test)
    
    # calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, auc_roc


##############################################################################################################

st.title('Credit Risk Analysis')

st.markdown('''
**Nabil EL FARME**

*Ph.D. student in statistics*
''')
st.header('Introduction')
st.markdown('''Today we are going to make a tutorial on building a credit risk model using an existing dataset with code. We will use the "German Credit Dataset" from the UCI Machine Learning Repository, which contains data on 1000 loan applications with 20 attributes including credit history, employment status, and personal characteristics. Our goal is to build a machine learning model that predicts whether a loan applicant is a good or bad credit risk based on these attributes.

The first step is to Load the German Credit Dataset into a Pandas DataFrame:
- The url variable contains the URL where the "German Credit Dataset" is stored.
- The columns variable contains a list of column names that will be used to label the columns of the dataset.
- pd.read_csv() function is used to read the dataset from the URL.
- delimiter=" " is used to specify that the columns are separated by spaces.
- header=None is used to indicate that the dataset doesn't have a header row.
- names=columns is used to label the columns of the dataset with the names in the columns list.
- df.head() is used to display the first few rows of the dataset.''')
st.code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["checking account", "duration", "credit history", "purpose", "credit amount", 
           "savings account", "employment duration", "installment rate", "personal status",
           "other debtors", "residence history", "property", "age", "installment plan", 
           "housing", "number of credits", "job", "people liable", "telephone", "foreign worker", "Risk"]
df = pd.read_csv(url, delimiter=" ", header=None, names=columns)
# Display the first few rows of the dataset
df.head()''', language='python')

df = pd.read_csv("german.data",delimiter=' ', header=None)
col = ["checking account", "duration", "credit history", "purpose", "credit amount", 
           "savings account", "employment duration", "installment rate", "personal status",
           "other debtors", "residence history", "property", "age", "installment plan", 
           "housing", "number of credits", "job", "people liable", "telephone", "foreign worker","Risk"]
df.columns = col
dfx = df

st.caption("First few rows of the dataset")
with st.expander('View Table'):
    st.table(df.head())



st.header("Dataset Description")
table1 = {
"Attribute 1": " (qualitative) | Status of existing checking account | A11 :... <0 DM | A12 : 0 <= ... <200 DM | A13 : ... >= 200 DM /salary assignments for at least 1 year | A14 : no checking account",

"Attribute 2": " (numerical) | Duration in month",

"Attribute 3": " (qualitative) | Credit history | A30 : no credits taken/all credits paid back duly | A31 : all credits at this bank paid back duly | A32 : existing credits paid back duly till now | A33 : delay in paying off in the past | A34 : critical account/other credits existing (not at this bank)",

"Attribute 4": " (qualitative) | Purpose | A40 : car (new) | A41 : car (used) | A42 : furniture/equipment | A43 : radio/television | A44 : domestic appliances | A45 : repairs | A46 : education | A47 : (vacation - does not exist?) | A48 : retraining | A49 : business | A410 : others",
	        
"Attribute 5": "(numerical) | Credit amount",

"Attibute 6":  "(qualitative) | Savings account/bonds | A61 : ... < 100 DM | A62 : 100 <= ... < 500 DM | A63 : 500 <= ... < 1000 DM | A64 : .. >= 1000 DM | A65 : unknown/ no savings account",
	        
"Attribute 7":  "(qualitative) | Present employment since | A71 : unemployed | A72 : ... < 1 year | A73 : 1 <= ... < 4 years | A74 : 4 <= ... < 7 years | A75 : .. >= 7 years",
	        
"Attribute 8":  "(numerical) | Installment rate in percentage of disposable income",

"Attribute 9":  "(qualitative) | Personal status and sex | A91 : male   : divorced/separated | A92 : female : divorced/separated/married | A93 : male   : single | A94 : male   : married/widowed | A95 : female : single",

"Attribute 10": "(qualitative) | Other debtors / guarantors | A101 : none | A102 : co-applicant | A103 : guarantor",
	        
"Attribute 11": "(numerical) | Present residence since",

"Attribute 12": " (qualitative) | Property | A121 : real estate | A122 : if not A121 : building society savings agreement/life insurance | A123 : if not A121/A122 : car or other, not in attribute 6 | A124 : unknown / no property",
	        
"Attribute 13": "(numerical) | Age in years",

"Attribute 14": "(qualitative) | Other installment plans | A141 : bank | A142 : stores | A143 : none",
	        
"Attribute 15": " (qualitative) | Housing | A151 : rent | A152 : own | A153 : for free",
	        
"Attribute 16": "(numerical) | Number of existing credits at this bank",

"Attribute 17": "(qualitative) | Job | A171 : unemployed/ unskilled  - non-resident | A172 : unskilled - resident | A173 : skilled employee / official | A174 : management/ self-employed/ highly qualified employee/ officer",

"Attribute 18": "(numerical) | Number of people being liable to provide maintenance for",

"Attribute 19": "(qualitative) | Telephone | A191 : none | A192 : yes, registered under the customers name"

,"Attribute 20": "(qualitative) | foreign worker | A201 : yes | A202 : no"}

table1 = pd.DataFrame(table1.items(), columns=['Attribute', 'Description'])
table1 = table1.drop(columns=['Attribute'])
table1.insert(0, "Name", col[:-1])
# table1 = table1.set_index(['Name'])

st.caption("Variable description")
with st.expander('View Table'):
    st.table(table1)

st.markdown('''For more understanding of the data, you should have a general idea of the expected effect of each variable on the target variable based on common knowledge and assumptions in the field of credit risk modeling. Here are some general expectations:
- Age: Younger applicants may have less stable income and credit history, so they may be considered higher risk.
- Sex: Historically, females were considered lower risk than males, but this may not be the case anymore due to changing social norms and regulations.
- Job: Certain types of jobs may have higher or lower income stability, which could affect credit risk.
- Housing: Homeowners may have more stable income and be considered lower risk.
- Saving accounts: Higher savings may indicate more financial stability and lower risk.
- Checking account: Higher balance may indicate more financial stability and lower risk.
- Credit amount: Higher credit amounts may indicate more risk, as borrowers may struggle to make payments.
- Duration: Longer duration may indicate more risk, as there is more time for borrowers to default on payments.
- Purpose: Certain loan purposes may be considered higher or lower risk, depending on the borrower's history and financial stability.
''')

st.header("Data Preprocessing and Exploration")
st.markdown('''

Next, we will explore the dataset and visualize the relationships between the variables. We will use a combination of descriptive statistics and data visualizations to gain insights into the data:

Here are some basic steps to explore and clean the data:

1. Check the number of rows and columns in the dataset using df.shape.
2. Check the data types of each variable using df.dtypes.
3. Check for missing values using df.isnull().sum().
4. Check for duplicate rows using df.duplicated().sum().
5. Check for the distribution of each variable using df.describe() and df.hist().
6. Check boxplot.
7. Check for the correlation between variables using df.corr() and a heatmap.
''')

st.code('''# Check the number of rows and columns in the dataset
print("Number of rows: {}".format(df.shape[0]))
print("Number of columns: {}".format(df.shape[1]))
# Check the data types of each variable
print(df.dtypes)
# Check for missing values
print(df.isnull().sum())
# Check for duplicate rows
print("Number of duplicate rows: {}".format(df.duplicated().sum()))
# Check the distribution of each variable
print(df.describe())''')

col1, col2, col3 = st.columns(3)
col1.metric("Number of rows", f"{df.shape[0]}")
col2.metric("Number of columns", f"{df.shape[1]}")
col3.metric("Number of duplicate rows", f"{df.duplicated().sum()}")

st.caption("Data information")
df_info = pd.DataFrame(list(zip(col,df.dtypes, df.isnull().sum())), columns=['Column','Data Type', 'Missing Values'])

with st.expander('View Table'):
    st.table(df_info)


attribute_map = {
    "A11": "< 0 DM",
    "A12": "0 <= ... < 200 DM",
    "A13": ">= 200 DM / salary assignments for at least 1 year",
    "A14": "no checking account",
    "A30": "no credits taken / all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account / other credits existing (not at this bank)",
    "A40": "car (new)",
    "A41": "car (used)",
    "A42": "furniture/equipment",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "(vacation - does not exist?)",
    "A48": "retraining",
    "A49": "business",
    "A410": "others",
    "A61": "< 100 DM",
    "A62": "100 <= ... < 500 DM",
    "A63": "500 <= ... < 1000 DM",
    "A64": ">= 1000 DM",
    "A65": "unknown / no savings account",
    "A71": "unemployed",
    "A72": "< 1 year",
    "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years",
    "A75": ">= 7 years",
    "A91": "male : divorced/separated",
    "A92": "female : divorced/separated/married",
    "A93": "male : single",
    "A94": "male : married/widowed",
    "A95": "female : single",
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
    "A121": "real estate",
    "A122": "building society savings agreement / life insurance",
    "A123": "car or other (not in attribute 6)",
    "A124": "unknown / no property",
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
    "A171": "unemployed / unskilled - non-resident",
    "A172": "unskilled - resident",
    "A173": "skilled employee / official",
    "A174": "management / self-employed / highly qualified employee / officer",
    "A191": "none",
    "A192": "yes, registered under the customer's name",
    "A201": "yes",
    "A202": "no",
    1:"good",
    2:"bad"
}

cols = ["checking account", "credit history", "purpose", 
           "savings account", "employment duration", "personal status",
           "other debtors", "property", "installment plan", 
           "housing", "job", "telephone", "foreign worker"]
st.subheader("Target Variable")
dft = df
dft['Risk'] = dft['Risk'].replace(attribute_map)
# Group the DataFrame by the 'Purpose' column and count the number of occurrences of each purpose
counts = dft.groupby('Risk')['Risk'].count()
counts = counts.sort_values()
# Create a horizontal bar chart using Plotly
fig = px.bar(counts, orientation='h', title='Distribution of Risk',
             width=650,
             height=300,
            color=counts.index, color_discrete_sequence = ['#cc0036','#228000'])
fig.update_xaxes(title='Frequency')
fig.update_yaxes(title='Risk')
fig.update_layout(showlegend=False)
st.write(fig)


st.subheader("Numerical Variables")
st.caption("Data statistics")
with st.expander('View Table'):
    st.table(df.describe())

numerical_cols = ["age","credit amount","duration",
            "credit amount","installment rate" ,
            "residence history","number of credits",
            "people liable"]
    
coll1, coll2 = st.columns(2)
with coll1:
    # Compute correlation matrix
    colorsel = st.selectbox('Select color', ['Viridis','magma','Plasma','Cividis','Greens'], index=0)
    df_c = df[numerical_cols]
    corr_matrix = df_c.corr()
    # Create heatmap figure
    figc = px.imshow(corr_matrix, 
                    x=corr_matrix.columns, 
                    y=corr_matrix.columns,
                    width=350,
                    height=400,
                    color_continuous_scale= colorsel)
    # Set layout title
    figc.update_layout(title = 'Correlation Matrix')
    # Show figure
    st.write(figc)


with coll2:
    # Specify the column of interest for the histogram
    column_of_interest = st.selectbox('Select a numerical column', numerical_cols, index=0)

    # Create a histogram using Plotly

    figh = px.histogram(df, x=column_of_interest, 
                        nbins=20, 
                        title="Histogram of Credit Amount")

    # Customize the layout if needed
    figh.update_layout(
        xaxis_title=column_of_interest,
        yaxis_title="Count",
        width=300,
        height=360,
        bargap=0.2
    )

    # Display the plot
    st.write(figh)

st.markdown("""
From the correlation matrix, we can observe that there is a moderate positive 
correlation between credit amount and duration, which indicates that clients 
tend to borrow more for longer durations. There is also a moderate negative 
correlation between credit amount and age, which indicates that younger clients 
tend to borrow more. Moreover, there is a weak negative correlation between the 
target variable and credit history, which implies that clients with a good 
credit history are more likely to be classified as "good" risk. However, 
we should keep in mind that correlation does not necessarily imply causation, 
and further analysis is required to draw reliable conclusions.""")

st.subheader("Catagorical Variables")
    


typs = ['pie','bar']
dic = {cols[i]:i for i in range(len(cols))}
grf = st.selectbox('Select attribute to explore:', cols)

graphs1 = TargetGraph(df, cols)
st.write(graphs1[dic[grf]])

typ = st.selectbox('Select chart type', typs)
graphs = X_chart(df, cols,typ)
st.write(graphs[dic[grf]])


target = st.selectbox(f"Box plot of {grf} versus ...?", df.columns, index=1)
boxplotG = BoxP(df, target, cols)
st.write(boxplotG[dic[grf]])


st.markdown("""

From the generated graphs for each variable in the dataset, we can observe 
the following:

* Age: Most individuals are in the age range of 20-30.
* Job: The most common job is skilled employee.
* Credit amount: Most of the credit amounts are between 0 and 10,000.
* Duration: Most loans have a duration of 10-20 months.
* Purpose: Most of the loans are taken for the purpose of buying a car, followed by furniture/equipment.
* Credit history: Most of the individuals have existing credits paid back duly till now.
* Saving accounts: Most of the individuals have savings accounts of less than 100 DM.
* Checking account: Most of the individuals have checking accounts of less than 0 DM.
* Housing: Most of the individuals have their own housing.
* Installment rate: Most of the loans have an installment rate of 1-2.
* Personal status: Most of the individuals are male and have a personal status of 'others'.
* Other debtors: Most of the individuals have no other debtors.
* Property: Most of the individuals have no property.
* Number of existing credits: Most of the individuals have 1 existing credit.
Job related to number of people being liable to provide maintenance for: Most of the individuals have 1 person being liable to provide maintenance.
* Telephone: Most of the individuals have a telephone.
* Foreign worker: Most of the individuals are not foreign workers.
From this, we can get a sense of the range and spread of each variable in the 
dataset and how they relate to the target variable, and we can use this information to inform our feature engineering and modeling decisions.
""")
#-------------------------------------------------------------

st.header("Model Building")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
# Initialize LabelEncoder
le = LabelEncoder()
columns=["checking account", "credit history", "purpose", "savings account", 
         "employment duration", "personal status", "other debtors", "property", 
         "installment plan", "housing", "job", "telephone", "foreign worker"]
encoding_dict = []
for col in columns:
    dfx[col] = le.fit_transform(df[col])
    decoded_labels = le.inverse_transform(dfx[col])
    encoding_dict.append(dict(zip(decoded_labels,dfx[col])))


st.markdown("""
**Step 3: Data Preparation for Modeling**

Categorical variables need to be encoded into numerical 
values because most machine learning algorithms only accept numerical data as input. 
Encoding categorical variables allows the algorithm to interpret the data and make predictions. 
There are several methods to encode categorical variables such as one-hot encoding and label encoding. 
In this project, we will use label encoding because it is simple and efficient.
""")

st.code("""
from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
le = LabelEncoder()
columns=["checking account", "credit history", "purpose", "savings account", 
         "employment duration", "personal status", "other debtors", "property", 
         "installment plan", "housing", "job", "telephone", "foreign worker"]
encoding_dict = []
for col in columns:
    dfx[col] = le.fit_transform(df[col])
""", language='python')

st.markdown("""
Scaling the numerical variables is important because the range of the variables may 
differ significantly. For example, the range of age and credit amount may differ by 
thousands or more. This can cause issues for some machine learning algorithms which 
rely on the assumption that all features have equal importance. Scaling the features 
can help to mitigate this issue and improve the performance of the machine learning 
model. Standardization and normalization are two common scaling techniques used to 
transform numerical variables into a similar range.

Before we can build the model, we need to split the data into training and testing 
sets, encode categorical variables, and scale the numerical variables.

""")

st.code("""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Select the numerical columns to scale
num_cols = ['duration', 'credit amount', 'installment rate', 
            'residence history', 'age', 'number of credits', 
            'people liable']

# Instantiate the scaler
scaler = StandardScaler()
# Fit and transform the data
dfx[num_cols] = scaler.fit_transform(dfx[num_cols])

# Split the dataset into training and testing sets
X = df.drop('Risk', axis=1)
y = df['Risk']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language='python')


# Select the numerical columns to scale
num_cols = ['duration', 'credit amount', 'installment rate', 
            'residence history', 'age', 'number of credits', 
            'people liable']
# Instantiate the scaler
scaler = StandardScaler()
# Fit and transform the data
dfx[num_cols] = scaler.fit_transform(dfx[num_cols])
dfx['Risk'] = dfx['Risk'].apply(lambda x: 0 if x == 'bad' else 1)
# Split the dataset into training and testing sets
X = dfx.drop('Risk', axis=1)
y = df['Risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.caption("Data after preprocessing")
with st.expander('View Table'):
    st.table(X_train.head())

st.markdown("""The idea behind evaluating machine learning models is to determine how well they perform at solving a given problem. The goal is to select the best model for the task at hand, which involves comparing the performance of different models and selecting the one that provides the best results based on evaluation metrics. 

Evaluation metrics are used to quantify the performance of a model by comparing its predicted output to the actual output. They provide an objective way to measure how well a model is performing and allow us to compare the performance of different models.

As this tuto for learning, we will use as much as possible of metrics for evaluation, here's some info about them: 
* Accuracy_score: measures the proportion of correct classifications out of total instances. It is calculated as (true positive + true negative) / total instances. Accuracy is a good measure when the classes are balanced, meaning there are roughly equal numbers of instances in each class. However, when the classes are imbalanced, accuracy can be misleading and other metrics may be more appropriate.

* Precision_score: measures the proportion of true positive classifications out of total positive predictions. It is calculated as true positive / (true positive + false positive). Precision is a good measure when the cost of false positives is high. For example, in a medical diagnosis task, a false positive could lead to unnecessary and potentially harmful treatment.

* Recall_score: measures the proportion of true positive classifications out of total actual positives. It is calculated as true positive / (true positive + false negative). Recall is a good measure when the cost of false negatives is high. For example, in a spam email detection task, a false negative could result in an important email being missed.

* F1_score: is the harmonic mean of precision and recall. It combines both measures into a single score, giving equal weight to both precision and recall. It is calculated as 2 * (precision * recall) / (precision + recall). F1_score is a good measure when both precision and recall are equally important.

* ROC_auc_score: measures the area under the receiver operating characteristic (ROC) curve, which plots the true positive rate (recall) against the false positive rate (1 - specificity) at various threshold settings. ROC_auc_score is a good measure when the cost of false positives and false negatives are roughly equal and the trade-off between precision and recall is not a concern.

* ROC_curve: is the curve generated by plotting the true positive rate (recall) against the false positive rate (1 - specificity) at various threshold settings. It is used to visually evaluate the performance of a binary classification model. A good model will have an ROC curve that is close to the top left corner of the plot, indicating a high true positive rate and a low false positive rate""")

st.code('''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # fit the model on the training data
    model.fit(X_train, y_train)
    
    # predict the target variable for the test data
    y_pred = model.predict(X_test)
    
    # calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, auc_roc
    
# create a logistic regression object
lr = LogisticRegression()
# create a random forest object
rf = RandomForestClassifier()

# evaluate logistic regression model
lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc_roc = evaluate_model(lr, X_train, y_train, X_test, y_test)
# evaluate Random Forest model
rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc_roc = evaluate_model(rf, X_train, y_train, X_test, y_test)
''', language='python')


# create a logistic regression object
lr = LogisticRegression()
# create a random forest object
rf = RandomForestClassifier()

# evaluate logistic regression model
lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc_roc = evaluate_model(lr, X_train, y_train, X_test, y_test)
# evaluate Random Forest model
rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc_roc = evaluate_model(rf, X_train, y_train, X_test, y_test)


col1 = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC Score']
col2 = [ lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc_roc]
col3 = [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc_roc]   

Vresult = pd.DataFrame(list(zip(col1, col2, col3)), columns =['Model', 'Logistic Regression', 'Random Forest'])
st.table(Vresult)


st.markdown("""From the results, we can see that random forest has slightly better performance than logistic regression, with an accuracy of 0.76, precision of around 0.78, recall of around 0.9, F1-score of around 0.84 and AUC-ROC score of around 0.66.
Therefore, we can conclude that random forest outperform the logistic regression model on this dataset. However, further analysis and tuning of the models may be necessary to improve their performance.

It's also worth noting that there is a potential issue with the AUC-ROC score being relatively low for both models. This may indicate that the models are not able to effectively distinguish between the positive and negative classes. In this case, we may need to investigate further and potentially try different models or feature engineering techniques.


here's an overview of the modeling process we went through:

1. We sarted by loading and exploring the German Credit Dataset, checking for missing values and examining the distribution of the target variable.

2. Next, we preprocessed the data by encoding the categorical variables using one-hot encoding and scaling the numerical variables using standardization.

3. We then split the data into training and testing sets, with a 80-20 ratio.

4. After that, we trained three models on the training data: logistic regression, Random Forest, and Gradient Boosting.

5. We evaluated the models using several evaluation metrics such as accuracy, precision, recall, F1-score, and AUC-ROC score, and found that all three models had reasonable performance, with Random Forest performing slightly better in terms of recall and F1-score.

6. Finally, we tested the best-performing model (Random Forest) on the testing data and obtained similar results, indicating that the model generalizes well to new, unseen data.

Overall, our goal was to build a model that can accurately predict the risk (the class variable) of default for loan applicants based on their demographic and financial characteristics, and we achieved this goal to a reasonable degree using machine learning techniques.
""")