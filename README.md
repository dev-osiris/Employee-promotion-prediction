# Employee Promotion Prediction

**https://employee-promotion-prediction-wftg.onrender.com/**

One of the problem our corporate world is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. The traditional process is, first identify a set of employees based on recommendations/ past performance. Selected employees go through the separate training and evaluation program for each vertical. At the end of the program, based on various factors such as training performance, an employee gets the promotion.

<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/1-wrong-promotion.jpg">

\
&nbsp;
\
&nbsp;

**Description of all the columns of dataset**

<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/2-data_desc.PNG" width="500" height="350">

\
&nbsp;
\
&nbsp;

**Bivariate analysis**

Check the gender gap in promotion  


```
# Lets compare the Gender Gap in the promotion

plt.rcParams['figure.figsize'] = (15, 3)
x = pd.crosstab(train['gender'], train['is_promoted'])
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = False, color = colors)
plt.title('Effect of Gender on Promotion', fontsize = 15)
plt.xlabel(' ')
plt.show()
```
<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/3-gender.png">
Females are in Minority, but when it comes to Promotion, they are competing with their Men Counterparts neck-to-neck.

\
&nbsp;
\
&nbsp;

Check the affect of department on promotion

```
plt.rcParams['figure.figsize'] = (15,4)
x = pd.crosstab(train['department'], train['is_promoted'])
colors = plt.cm.copper(np.linspace(0, 1, 3))
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'area', stacked = False, color = colors)
plt.title('Effect of Department on Promotion', fontsize = 15)
plt.xticks(rotation = 20)
plt.xlabel(' ')
plt.show()
```
<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/4-department.png">
From, the above chart we can see that almost all the Departments have a very similar effect on Promotion. So, we can consider that all the Departments have a similar effect on the promotion. Also, this column comes out to be lesser important in making a Machine Learning Model, as it does not contribute at all when it comes to Predicting whether the Employee should get Promotion.

\
&nbsp;
\
&nbsp;

**Multivariate Analysis**

Use the Correlation Heatmap to check the correlation between the Numerical Columns

```
# Heat Map for the Data with respect to correlation.

plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(train.corr(), annot = True, linewidth = 0.5, cmap = 'Wistia')
plt.title('Correlation Heat Map', fontsize = 15)
plt.show()
```
<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/5-correlation.png" height="600">

Here, we can see some obvious results, that is Length of Service, and Age are Highly Correlated, Also, KPIs, and Previous year rating are correlated to some extent, hinting that there is some relation.

\
&nbsp;
\
&nbsp;

**Decision Tree Classifier**

We are using decision tree classifier from Sklearn library too make predictions.
```
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_valid, y_pred)
plt.rcParams['figure.figsize'] = (3, 3)
sns.heatmap(cm, annot = True, cmap = 'Wistia', fmt = '.8g')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()
```

<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/6-confusion.png">

```
Training Accuracy : 0.9996159754224271
Testing Accuracy : 0.956221198156682
```

\
&nbsp;
\
&nbsp;
\
&nbsp;


**DEPLOYEMENT**

**Flask framework**

Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications in Python easier. It gives developers flexibility and is a more accessible framework for new developers since you can build a web application quickly using only a single Python file. Flask is also extensible and doesn’t force a particular directory structure or require complicated boilerplate code before getting started.

<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/7-data_flow.png">

\
&nbsp;
\
&nbsp;


**Hosting on Render Platform**

The app is deployed on  [Render](https://wwww.Render.com) platform on https://employee-promotion-prediction-wftg.onrender.com/

&nbsp;

screenshot:

<img src="https://github.com/dev-osiris/Employee-images/blob/main/images/Screenshot.png" widht="840" height="500">
