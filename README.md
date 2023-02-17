# finding_donors

## Udacity Machine Learning Engineer Nanodegree

### Unit 2: Supervised Learning

## Project: Finding Donors for CharityML

This project requires Python, the Python file ‘visuals.py’ provided by Udacity and the following libraries:
-	NumPy
-	Pandas
-	Matplotlib
-	Seaborn
-	Scikit-learn

### Data
1994 U.S. Census data are provided by Udacity as a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. The original dataset is hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).
There are more than 45000 data points and 13 features:
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

The Target Variable is:
- `income`: Income Class (<=50K, >50K)

## Project Overview
In this project, supervised learning techniques are applied on data collected for the 1994 U.S. census to help CharityML (a fictitious charity organisation) identify people with higher incomes and therefore most likely to donate to their cause.
The project contains several sections:
-	Data exploration (including plots of features and target)
-	Data transformation and pre-processing (transformation of skewed continuous features, normalization of numerical features, one-hot encoding of categorical features, converting target label to numerical values)
-	Establishing a benchmark: the Naïve predictor
-	Evaluation of three supervised classifier models with default hyperparameters on different training set sizes: Gaussian Naïve Bayse, Support Vector Machines, ensemble method AdaBoost (based on accuracy and f_score)
-	Exercise on optimisation of AdaboostClassifier model with GridSearchCV (final result is: accuracy 0.8588, F_score 0.7263)
-	Extraction of the five most important features

This second Udacity project of the Machine Learning Engineer Nanodegree was designed for students to become familiar with some of the supervised learning algorithms available in sklearn. It was designed also to provide a method for evaluating how the selected models work and perform on the data.
