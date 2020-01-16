# HR-Analytics

The goal of the competition was to predict which employee will get promoted. To help us in our prediction a csv file containing the following features has been given. 

Features   |employee_id  | department | region | education | gender | recruitment_channel | no_of_trainings | age | previous_year_rating |
| -------- |-------------|------------|--------|-----------|--------|---------------------|-----------------|-----|----------------------|
Description||department in which the employee works|geographical region||||number of training years||previous year rating of performance (1 to 5)|


Features   |length_of_service|KPIs_met|awards_won|avg_training_score|is_promoted|
| -------- |-----------------|--------|----------|------------------|-----------|
Description|number of years of service|did the employee reach the objective performance asked (1=yes and 0=no)|||is the employee promoted ? (1=yes, 0=no)|

A first file is given as a training sample and another as a test where the class (is_promoted) is unknown. Also, for some instances the previous_year_rating and the education were missing. 

By looking at the data (cf. to my Public Tableau account given in the description), I discovered some interesting patterns (ex: employees with an average training score greater than 95% are automatically promoted, after 28 years old below Secondary educated employees are never promoted,..). Si I decided to use a random forest in order to predict is_promoted and to use SMOTE (Synthetic Minority Over-Sampling TEchnique) to oversample the minority class (44428 employees were not promoted and only 4232 were). However first I had to deal with the missing values of edculation and previous_year_rating.

To address this issue, I chose to design a random forest taking as input the training data without the incomplete individuals and to trace the importance of each feature in predicting the target class is_promoted. Here is the graph:

![alt text](https://github.com/Yami-B/HR-Analytics/blob/master/MeanDecreaseGini.png)

From this graph, we can see that education has in fact a limited discriminatory effect on our target class compared to avg_training_score, so we might as well not consider it when implementing our random forest. However, I decided to keep the previous_year_rating since it has a  moderate mean decrease Gini compared to other features. I chose to first impute the mean value of previous_year_rating to these missing values in my R code. In my Python code, I tried to improve the quality of imputation by using a random forest, and for predict if an employee is promoted I used LightGBM classifier since it's a very powerful and fast tool for classification.
