
# coding: utf-8

# # Introduction (688 only)
# In this question, you'll build a basic recommendation system using collaborative filtering to make recommendations on a typical recommendation systems dataset, the MovieLens dataset. The purpose of this question is to become familiar with the internals of recommendation systems: both how they train and how they form recommendations. 
# 
# ### Grading 
# Your submission will be scored in the following manner: 
# * process - 10pts
# * train - 15pts
# * recommend - 10pts
# 
# ## Collaborative Filtering by Matrix Factorization
# In collaborative filtering we wish to factorize our ratings matrix into two smaller feature matrices whose product is equal to the original ratings matrix. Specifically, given some partially filled ratings matrix $X\in \mathbb{R}^{m\times n}$, we want to find feature matrices $U \in \mathbb{R}^{m\times k}$ and $V \in \mathbb{R}^{n\times k}$ such that $UV^T = X$. In the case of movie recommendation, each row of $U$ could be features corresponding to a user, and each row of $V$ could be features corresponding to a movie, and so $u_i^Tv_j$ is the predicted rating of user $i$ on movie $j$. This forms the basis of our hypothesis function for collaborative filtering: 
# 
# $$h_\theta(i,j) = u_i^T v_j$$
# 
# In general, $X$ is only partially filled (and usually quite sparse), so we can indicate the non-presence of a rating with a 0. Notationally, let $S$ be the set of $(i,j)$ such that $X_{i,j} \neq 0$, so $S$ is the set of all pairs for which we have a rating. The loss used for collaborative filtering is squared loss:
# 
# $$\ell(h_\theta(i,j),X_{i,j}) = (h_\theta(i,j) - X_{i,j})^2$$
# 
# The last ingredient to collaborative filtering is to impose an $l_2$ weight penalty on the parameters, so our total loss is:
# 
# $$\sum_{i,j\in S}\ell(h_\theta(i,j),X_{i,j}) + \lambda_u ||U||_2^2 + \lambda_v ||V||_2^2$$
# 
# For this assignment, we'll let $\lambda_u = \lambda_v = \lambda$. 
# 
# ## MovieLens rating dataset
# To start off, let's get the MovieLens dataset. This dataset is actually quite large (24+ million ratings), but we will instead use their smaller subset of 100k ratings. You will have to go fetch this from their website. 
# 
# * You can download the archive containing their latest dataset release from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip (last updated October 2016). 
# * For more details (contents and structure of archive), you can read the README at http://files.grouplens.org/datasets/movielens/ml-latest-README.html
# * You can find general information from their website description located at http://grouplens.org/datasets/movielens/. 
# 
# For this assignment, we will only be looking at the ratings data specifically. However, it is good to note that there is additional data available (i.e. movie data and user made tags for movies) which could be used to improve the ability of the recommendation system. 

# In[33]:


import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la


# In[34]:


# AUTOLAB_IGNORE_START
# movies = pd.read_csv("ml-latest-small/movies.csv")
# movies.head()


# # In[35]:


# ratings = pd.read_csv("ml-latest-small/ratings.csv")
# ratings.head()
# AUTOLAB_IGNORE_STOP


# ## Data preparation
# 
# Matrix factorization requires that we have our ratings stored in a matrix of users, so the first task is to take the dataframe and convert it into this format. Note that in general, typically these matrices are extremely large and sparse (especially if you want to process the 24 million ratings), however for the purposes of this assignment a dense representation of this smaller dataset should fit on any machine. 
# 
# ### Specification
# * Split the data by assigning the first $\mathrm{floor}(9n/10)$ permuted entries to be the training set, and the remaining to be the testing set. Use the order given by the permutation. 
# * Each row of the ratings matrix corresponds to a user. The first row of the matrix should correspond to the first user (by userID), and so on. This is because the set of user IDs already form a consecutive range of numbers. 
# * Each column of the ratings matrix corresponds to a movie. The order of the columns doesn't matter, so long as the resulting list of movie names is accurate. This is because the set of movie IDs does not form a consecutive range of numbers. 
# * Each user and movie that exists in the **ratings** dataframe should be present in the ratings matrix, even if it doesn't have any entries. We will only use the movies dataframe to extract the names of the movies. 
# * Any entry that does not have a rating should have a default value of 0. 

# In[36]:


import math
def process(ratings, movies, P):
    """ Given a dataframe of ratings and a random permutation, split the data into a training 
        and a testing set, in matrix form. 
        
        Args: 
            ratings (dataframe) : dataframe of MovieLens ratings
            movies (dataframe) : dataframe of MovieLens movies
            P (numpy 1D array) : random permutation vector
            
        Returns: 
            (X_tr, X_te, movie_names)  : training and testing splits of the ratings matrix (both 
                                         numpy 2D arrays), and a python list of movie names 
                                         corresponding to the columns of the ratings matrices. 
    """
    
    movieIDs = list(set(list(ratings['movieId'])))
    
    numberofRatings = len(P)
    numberOfUsers = ratings['userId'][numberofRatings - 1]
    numberOfMovies = len(movieIDs)
    
    X_tr = np.zeros((numberOfUsers, numberOfMovies))
    X_te = np.zeros((numberOfUsers, numberOfMovies))
    
    movieNameDict = dict(zip(list(movies['movieId']), list(movies['title'])))
    movieIdDict = dict(zip(movieIDs, list(range(numberOfMovies))))
    
    movieNames = list(map(lambda x: movieNameDict[x], movieIDs))
    
    threshold = int(math.floor(numberofRatings * 0.9))
    
    for idx, idx_rating in np.ndenumerate(P):
        item = ratings.iloc[[idx_rating]]
        idx = idx[0]
        
        row = int(item['userId']) - 1
        col = movieIdDict[int(item['movieId'])]
        val = item['rating']
        
        if idx < threshold:
            X_tr[row][col] = val
        else:
            X_te[row][col] = val
            
    return X_tr, X_te, movieNames

# AUTOLAB_IGNORE_START
# X_tr, X_te, movieNames = process(ratings, movies, np.random.permutation(len(ratings)))
# print(X_tr.shape, X_te.shape, movieNames[:5])
# AUTOLAB_IGNORE_STOP


# For example, running this on the small MovieLens dataset using a random permutation gives the following result: 
#     
#     (671L, 9066L) (671L, 9066L) ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)', 'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)']
# 
# Your actual titles may vary depending on the random permutation given. 

# ## Alternating Minimization for Collaborative Filtering
# Now we build the collaborative filtering recommendation system. We will use a method known as alternating least squares. Essentially, we will alternate between optimizing over $U$ and $V$ by holding the other constant. By treating one matrix as a constant, we get exactly a weighted least squares problem, which has a well-known solution. More details can be found in the lecture notes. 
# 
# ### Specification
# * Similar to the softmax regression on MNIST, there is a verbose parameter here that you can use to print your training err and test error. These should decrease (and converge). 
# * You can assume a dense representation of all the inputs. 
# * You may find it useful to have an indicator matrix W where $W_{ij} = 1$ if there is a rating in $X_{ij}$. 
# * You can initialize U,V with random values. 

# In[128]:


def error(X, U, V):
    """ Compute the mean error of the observed ratings in X and their estimated values. 
        Args: 
            X (numpy 2D array) : a ratings matrix as specified above
            U (numpy 2D array) : a matrix of features for each user
            V (numpy 2D array) : a matrix of features for each movie
        Returns: 
            (float) : the mean squared error of the observed ratings with their estimated values
        """
    X_pred = U @ V.T
    mse = ((X - X_pred) ** 2).mean(axis=None)
    return mse

def train(X, X_te, k, U, V, niters=51, lam=10, verbose=False):
    """ Train a collaborative filtering model. 
        Args: 
            X (numpy 2D array) : the training ratings matrix as specified above
            X_te (numpy 2D array) : the testing ratings matrix as specified above
            k (int) : the number of features use in the CF model
            U (numpy 2D array) : an initial matrix of features for each user
            V (numpy 2D array) : an initial matrix of features for each movie
            niters (int) : number of iterations to run
            lam (float) : regularization parameter
            verbose (boolean) : verbosity flag for printing useful messages
            
        Returns:
            (U,V) : A pair of the resulting learned matrix factorization
    """
    
    from numpy.linalg import inv
    
    lamI = lam * np.identity(k)

    tr_error = []
    te_error = []
    
    def cal(i, k, Xcal, Vcal):
        Vsub = Vcal[Xcal[i]!= 0]
        S = np.nonzero(Xcal[i])[0]
        sum_kxk = Vsub.T @ Vsub + lamI
        sum_kx1 = Vcal.T @ Xcal[i]
#         for j in S:
#             vj = Vcal[j].T
#             sum_kxk += vj @ vj.T
#             sum_kx1 += vj * Xcal[i][j]
#         sum_kxk = lamI
        # print((inv(sum_kxk) @ sum_kx1).T.shape)
        return (inv(sum_kxk) @ sum_kx1)
        
    
    for _ in range(niters):
        for i in range(len(U)):
            U[i] = cal(i, k, X, V)
        for i in range(len(V)):
            V[i] = cal(i, k, X.T, U)
        if verbose:
            _tr = error(X, U, V)
            _te = error(X_te, U, V)
            tr_error.append(_tr)
            te_error.append(_te)
            print("{}\t{}\t{}".format(_, _tr, _te))

    if verbose:
        return U, V, tr_error, te_error
    return U, V


# Training the recommendation system with a random initialization of U,V with 5 features and $\lambda = 10$ results in the following output. Your results may vary depending on your random permutation.  
# 
#     Iter |Train Err |Test Err  
#         0|    1.3854|    2.1635
#         5|    0.7309|    1.5782
#        10|    0.7029|    1.5078
#        15|    0.6951|    1.4874
#        20|    0.6910|    1.4746
#        25|    0.6898|    1.4679
#        30|    0.6894|    1.4648
#        35|    0.6892|    1.4634
#        40|    0.6891|    1.4631
#        45|    0.6891|    1.4633
#        50|    0.6891|    1.4636
#     Wall time: 7min 32s

# In[129]:


# AUTOLAB_IGNORE_START
# k = 5
# m, n = X_tr.shape
# U = np.random.random((m, k))
# V = np.random.random((n, k))

# retU, retV = train(X_tr, X_te, k, U, V, verbose=True)
# AUTOLAB_IGNORE_STOP


# ## Recommendations
# 
# Finally, we need to be able to make recommendations given a matrix factorization. We can do this by simply taking the recommending the movie with the highest value in the estimated ratings matrix. 
# 
# ### Specification
# * For each user, recommend the the movie with the highest predicted rating for that user that the user **hasn't** seen before. 
# * Return the result in a list such that the ith element in this list is the recommendation for the user corresponding to the ith row of the ratings matrix. 

# In[ ]:


def recommend(X, U, V, movieNames):
    """ Recommend a new movie for every user.
        Args: 
            X (numpy 2D array) : the training ratings matrix as specified above
            U (numpy 2D array) : a learned matrix of features for each user
            V (numpy 2D array) : a learned matrix of features for each movie
            movieNames : a list of movie names corresponding to the columns of the ratings matrix
        Returns
            (list) : a list of movie names recommended for each user
    """
    
    import copy
    
    uv_matrix = U @ V.T
    
    X_copy = copy.deepcopy(X)
    
    X_copy[X!=0] = -1.0
    X_copy[X==0] = uv_matrix[X==0]
    
    idxs = np.argmax(X_copy, axis=1).tolist()
    
    ret = list(map(lambda x: movieNames[x], idxs))
    
    return ret
    
# AUTOLAB_IGNORE_START
# recommendations = recommend(X_tr, U, V, movieNames)
# print(recommendations[:10])
# AUTOLAB_IGNORE_STOP


# Our implementation gets the following results (we can see they are all fairly popular and well known movies that were recommended). Again your results will vary depending on the random permutation. 
# 
#     ['Shawshank Redemption, The (1994)', 'Shawshank Redemption, The (1994)', 'Shawshank Redemption, The (1994)', 'Shawshank Redemption, The (1994)', 'Shawshank Redemption, The (1994)', 'Shawshank Redemption, The (1994)', 'Godfather, The (1972)', 'Fargo (1996)', 'Godfather, The (1972)', "Schindler's List (1993)"]
