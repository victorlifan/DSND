import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments
import recommender_functions as rf

class Recommender():
    '''
    instantiate Recommender() class,
    attributes:
    .fit()  fit trainig set
    .predict_rating()   predict given dataset user ratings
    .make_recs()    give back recommendations
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''
        


    def fit(self, trainpath,moviepath,latent_features=15, learning_rate=0.005, iters=100):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        INPUT:
        trainpath - train set path
        moviepath - movie data set path
        latent_features - (int) the number of latent features used (defule 15)
        learning_rate - (float) the learning rate (defule 0.005)
        iters - (int) the number of iterations (defule 100)

        OUTPUT: None
        attributes:
        train_df - review df
        movies -  movie df
        train_data_df - unstacked train df
        ratings_mat -rating matrix 
        n_users - num users
        n_movies - num movies
        num_ratings - num ratings
        user_mat - (numpy array) a user by latent feature matrix
        movie_mat - (numpy array) a latent feature by movie matrix
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews, then time, and must have more than 4 ratings
        '''
        self.train_df = pd.read_csv(trainpath)
        self.movies=pd.read_csv(moviepath)
        # Create user-by-item matrix - nothing to do here
        train_user_item = self.train_df[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        train_data_np  = np.array(self.train_data_df)
        self.ratings_mat=train_data_np
        # Set up useful values to be used through the rest of the function
        self.n_users = self.ratings_mat.shape[0]
        self.n_movies = self.ratings_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.ratings_mat))

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, latent_features)
        movie_mat = np.random.rand(latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.ratings_mat[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))
        self.user_mat=user_mat
        self.movie_mat=movie_mat
        self.ranked_movies=rf.create_ranked_df(self.movies,self.train_df)
        
    def predict_rating(self, user_id, movie_id):
        '''
        makes predictions of a rating for a user on a movie-user combo
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        try:
            # User row and Movie Column
            row=np.where(self.train_data_df.index==user_id)[0][0]
            column=np.where(self.train_data_df.columns==movie_id)[0][0]
            # Take dot product of that row and column in U and V to make prediction
            pred=np.dot(self.user_mat[row,:],self.movie_mat[:,column])
            
            name=self.movies.query("movie_id==@movie_id")[['movie','genre']].values[0][0]
            genre=self.movies.query("movie_id==@movie_id")[['movie','genre']].values[0][1]
            try:
                rate=self.train_df.query("user_id==@user_id and movie_id==@movie_id")[['rating','date']].values[0][0]
                date=self.train_df.query("user_id==@user_id and movie_id==@movie_id")[['rating','date']].values[0][1]
                print("user {} rated movie '{}'({}) {}/10 on {}. The prediction is {}".format(user_id, name,genre,rate, date,pred))
            except IndexError:
                print("user {} haven't watch movie '{}'({}). The prediction is {}".format(user_id, name,genre,pred))     
        except:
            print("user_id or movie_id doesn't exist")
            
    def make_recs(self,_id, _id_type='movie', rec_num=5):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str) (defult 'movie')
        rec_num - number of recommendations to return (int) (defult 5)

        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies by id                  
        rec_names - (array) a list or numpy array of recommended movies by name
        '''
        if _id_type=='movie':
            try:
                rec_names=rf.find_similar_movies(_id,self.movies)[:rec_num]
                rec_ids=self.movies[self.movies['movie'].isin(rec_names)]['movie_id'].values[:rec_num]
            except:
                print('movie not in dataset')
                rec_ids, rec_names=None, None
        else:
            if _id in self.train_data_df.index:
                # find row in user_mat
                user=np.where(self.train_data_df.index==_id)[0][0]
                # preidct rateing on user with all movies
                pre= np.dot(self.user_mat[user,:],self.movie_mat)
                # get movies indices of top rec_num records
                indices=np.argsort(pre)[::-1][:rec_num]
                # get movie ids with index
                rec_ids=self.train_data_df.columns[indices].values
                # get movie names
                rec_names=rf.get_movie_names(rec_ids,self.movies)
            else:
                rec_names=rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                rec_ids=self.movies[self.movies['movie'].isin(rec_names)]['movie_id'].values[:rec_num]
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")
        return rec_ids, rec_names

if __name__ == '__main__':
    # test different parts to make sure it works
    import recommender_template as rt
    rec=rt.Recommender()
    rec.fit('data/train_data.csv','data/movies_clean.csv')
    rec.predict_rating(user_id=8, movie_id=2844)
    print(rec.make_recs(8,'user')) # user in the dataset
    print(rec.make_recs(1,'user')) # user not in dataset
    print(rec.make_recs(1853728)) # movie in the dataset
    print(rec.make_recs(1)) # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)