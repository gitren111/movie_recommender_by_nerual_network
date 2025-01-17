import collections
import pickle
import numpy as np
from sklearn.utils import shuffle

"""
Step 2: Data loading and management, providing batched data for the model
1、Read the preprocessed data from the.p files output by pre_process.py.
2、Implement the batching function (such as the next_batch method) through the Dataset class.
3、Provide specific functions, such as decompression_feature, to restore the data into the form of user and movie features.
4、Output: Return a batch of features and target values for use in the model training or testing phase.
"""
#Batch data management tools, which extract batch data from the already loaded data set for subsequent model training and testing.
class Dataset(object):
    def __init__(self,Xs,ys=None,shuffle=True):
        self._Xs = Xs
        self._ys = ys
        self._shuffle = True
        self._size = self._Xs.shape[0]
        self._index = list(range(self._size))
        self._current = 0
        self._epoch = 0

    #Batch Data Acquisition
    def next_batch(self,batch_size):
        if self._shuffle:
            if self._current >= self._size:
                print(f"Epoch {self._epoch} Complete the cycle, and reset the _current index to 0.")
                self._epoch += 1
                self._current = 0
            if self._current == 0:
                self._index = shuffle(self._index)
                print("Reshuffle the indices")

        start = self._current
        end = min(self._current + batch_size,self._size)
        self._current = end

        Xs = self._Xs[start:end]

        if self._ys is not None:#如果有标签数据
            ys = self._ys[self._index[start:end]]
            return Xs,ys
        else:
            return Xs

    @property
    def epoch(self):
        return self._epoch
    @property
    def size(self):
        return self._size

#Named tuple：Users、Movies
Users = collections.namedtuple('Users',['id','gender','age','job'])
Movies = collections.namedtuple('Movies',['id','genres','titles','title_length'])

#Data decompression
def decompression_feature(Xs):
    bath = len(Xs)
    user_id = np.reshape(Xs.take(0,1),[bath,1])
    user_gender = np.reshape(Xs.take(4,1),[bath,1])
    user_age = np.reshape(Xs.take(5,1),[bath,1])
    user_job = np.reshape(Xs.take(3,1),[bath,1])

    users = Users(user_id,user_gender,user_age,user_job)

    movie_id = np.reshape(Xs.take(6,1),[bath,1])
    movie_genres = np.array(list(Xs.take(10,1)))
    movie_titles = np.array(list(Xs.take(11,1)))
    movie_title_length = (movie_titles != 0).sum(axis=1)

    movies = Movies(movie_id,movie_genres,movie_titles,movie_title_length)

    return users,movies

if __name__ == '__main__':
    with open('data/data.p','rb') as data:
        train_x,_,train_y,_ = pickle.load(data,encoding='utf-8')
    dataset = Dataset(train_x.values,train_y.values)

    #test
    for i in range(2):
        Xs,ys = dataset.next_batch(2)
        users,movies = decompression_feature(Xs)
        #print('封装的用户数据')
        #print(users.index)
        #print('封装的电影数据')
        #print(movies)
        #print('标签数据')
        #print(ys)























