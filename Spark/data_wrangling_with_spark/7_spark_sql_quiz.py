#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling with Spark SQL Quiz
# 
# This quiz uses the same dataset and most of the same questions from the earlier "Quiz - Data Wrangling with Data Frames Jupyter Notebook." For this quiz, however, use Spark SQL instead of Spark Data Frames.

# In[1]:


from pyspark.sql import SparkSession

# TODOS: 
# 1) import any other libraries you might need
# 2) instantiate a Spark session 
# 3) read in the data set located at the path "data/sparkify_log_small.json"
# 4) create a view to use with your SQL queries
# 5) write code to answer the quiz questions 


# In[2]:


spark=SparkSession     .builder     .getOrCreate()


# In[3]:


path="data/sparkify_log_small.json"
user_log=spark.read.json(path)


# In[5]:


user_log.createOrReplaceTempView("user_log_table")


# In[4]:


user_log.printSchema()


# # Question 1
# 
# Which page did user id ""(empty string) NOT visit?

# In[8]:


# TODO: write your code to answer question 1
spark.sql('''
            SELECT DISTINCT page
            FROM user_log_table
            WHERE page NOT IN (SELECT page
                                FROM user_log_table
                                WHERE userId='')''').show()


# # Question 2 - Reflect
# 
# Why might you prefer to use SQL over data frames? Why might you prefer data frames over SQL?
# 
# Both Spark SQL and Spark Data Frames are part of the Spark SQL library. Hence, they both use the Spark SQL Catalyst Optimizer to optimize queries. 
# 
# You might prefer SQL over data frames because the syntax is clearer especially for teams already experienced in SQL.
# 
# Spark data frames give you more control. You can break down your queries into smaller steps, which can make debugging easier. You can also [cache](https://unraveldata.com/to-cache-or-not-to-cache/) intermediate results or [repartition](https://hackernoon.com/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4) intermediate results.

# # Question 3
# 
# How many female users do we have in the data set?

# In[10]:


# TODO: write your code to answer question 3
spark.sql('''
            SELECT COUNT(DISTINCT userId)
            FROM user_log_table
            WHERE gender ='F'
            ''').show()


# # Question 4
# 
# How many songs were played from the most played artist?

# In[25]:


# TODO: write your code to answer question 4
spark.sql('''
            SELECT COUNT(*)
            FROM user_log_table
            GROUP BY artist
            HAVING Artist IS NOT NULL
            ORDER BY 1 DESC
            LIMIT 1
            ''').show()


# # Question 5 (challenge)
# 
# How many songs do users listen to on average between visiting our home page? Please round your answer to the closest integer.

# In[35]:


# SELECT CASE WHEN 1 > 0 THEN 1 WHEN 2 > 0 THEN 2.0 ELSE 1.2 END;

is_home = spark.sql("SELECT userID, page, ts, CASE WHEN page = 'Home' THEN 1 ELSE 0 END AS is_home FROM user_log_table             WHERE (page = 'NextSong') or (page = 'Home')             ")

# keep the results in a new view
is_home.createOrReplaceTempView("is_home_table")

# find the cumulative sum over the is_home column
cumulative_sum = spark.sql("SELECT *, SUM(is_home) OVER     (PARTITION BY userID ORDER BY ts DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS period     FROM is_home_table")

# keep the results in a view
cumulative_sum.createOrReplaceTempView("period_table")

# find the average count for NextSong
spark.sql("SELECT AVG(count_results) FROM           (SELECT COUNT(*) AS count_results FROM period_table GROUP BY userID, period, page HAVING page = 'NextSong') AS counts").show()

