# Starbucks Portfolio Exercise

## Project Description
This portfolio exercise was originally used as a take-home assignment provided by Starbucks for their job candidates. This exercise involves an advertising promotion that was tested to see if it would bring more customers to purchase a specific product priced at $10. It costs the company 0.15 to send out each promotion. Ideally, we want to limit that promotion only to those that are most receptive to the promotion.

Our promotion strategy will be evaluated on 2 key metrics
1. Incremental Response Rate (IRR)
2. Net Incremental Revenue (NIR)


## File Description
1. training.csv: Contains training data
2. Test.csv: Contains test data
3. Starbucks.ipynb: Contains code for the promotional strategy based on uplift models
4. test_results.py: Contains functions to evaluate the IRR and NIR values. File is provided by Starbucks.

## Findings
* Purchase rate is obviously higher in promotion group. Number of pepople that don't make a purchase are almost the same regardless of receiving promotions or not. However, for cohort that made purchases, number of people that received promotions is significantly larger than people don't receive pormptions. This indicates promotions have some kind of effect of purchaing behaviors. We need to invesgate p-value of this A/B test to gain further evdience.
* p-value is extremely small in this case (5.5e-36), which means we have a very strong evdience to reject H0 (P_control>=P_test).
* Although we have statistical significant on the a/b test, but NIR suffers a negative value. This means net profit of treatment group is $2335 less than control group because control group doesn't need to bear 0.15/promotion cost. We do not have a practicial significant on this test.
* There are only about 1.2% of target data is 1, leaves these data set is very imblanced. This would casue predicting issue on ML algorithms. We need to handle the imbalanced data first.
> Introduce EasyEnsembleClassifier from imblanced learn library to handle imblanced data

## Result
* Expected irr : 0.0188 and nir : 189.45
* My irr with this strategy is 0.0183.(under expected-2.7%)
My nir with this strategy is 309.40 (over expected 63.3%)


## Software
+ Jupyter Notebook
+ Atom
+ Python 3.7
> + Sklearn
> + Numpy
> + pandas
> + scipy
> + imblearn
> + statsmodels
> + seaborn
> + matplotlib
