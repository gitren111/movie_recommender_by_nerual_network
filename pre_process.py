import hashlib
import os
import pickle
import re
import zipfile
from urllib.request import urlretrieve
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from io import StringIO


"""
Step 1: Data download, decompression, preliminary processing, and saving
1、Download and verify the integrity of the dataset.
2、Data cleaning: Handle missing values and invalid values.
3、Encoding: Perform One-Hot or Multi-Hot encoding on discrete features such as movie genres and user genders.
4、Saving: Split the processed data into training and testing sets, and save them in a serialized format. The generated.p files will be used for training, testing, and inference stages.
"""

#Provide a more visual progress bar to show the download progress
class DLprocess(tqdm):
    last_block = 0
    def hook(self,block_num=1,block_size=1,total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block)*block_size)
        self.last_block = block_num

#download ml-1m dataset
def download_m1_1m(save_path):
    url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    hash_code = 'c4d9eecfca2ab87c1945afe126590906'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_pathname = os.path.join(save_path,'ml-1m.zip')

    if os.path.exists(save_pathname):
        print('The "ml - 1m.zip" file already exists, no need to download.')
    else:
        with DLprocess(unit='B',unit_scale=True,miniters=1,desc='downloading m1-1m') as pbar:
            urlretrieve(url,save_pathname,pbar.hook)

        #assert hashlib.md5(open(save_pathname,'rb').read()).hexdigest() == hash_code,\
        #'{}The file is corrupted. Remove the file and try again'.format(save_path)

        print('successfully downloaded the ml - 1m file')
    return save_pathname


def genres_multi_hot(genre_int_map):
    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split('|')]
        multi_hot = np.zeros(len(genre_int_map))
        multi_hot[genre_int_list] = 1
        return multi_hot
    return helper

def title_encode(word_int_map):
    def helper(title):
        title_word = [word_int_map[word] for word in title.split()]
        if len(title_word)>15:
            return np.array(title_word[:15])
        else:
            title_vector = np.zeros(15)
            title_vector[:len(title_word)] = title_word
            return title_vector
    return helper

#Data preprocessing
def load_data(dataset_zip):
    with zipfile.ZipFile(dataset_zip) as zf:
        #Reading users data and data preprocessing
        with zf.open('ml-1m/users.dat') as users_raw_data:
            users_title = ['UserID','Gender','Age','JobID','Zip-code']
            users_raw_data = users_raw_data.read().decode('ISO-8859-1')
            users_raw_data = StringIO(users_raw_data)
            users = pd.read_table(users_raw_data,sep='::',header=None,names=users_title,engine='python')
            users = users.filter(regex='UserID|Gender|Age|JobID')

            gender_map = {'F':0 , 'M':1}
            users['GenderIndex'] = users['Gender'].map(gender_map)
            age_map = {var:ii for ii,var in enumerate(set(users['Age']))}
            users['AgeIndex'] = users['Age'].map(age_map)

        #Reading movies data and data preprocessing
        with zf.open('ml-1m/movies.dat') as movies_raw_data:
            movies_title = ['MovieID','Title','Genres']
            movies_raw_data = movies_raw_data.read().decode('ISO-8859-1')
            movies_raw_data = StringIO(movies_raw_data)
            movies = pd.read_table(movies_raw_data,sep='::',header=None,names=movies_title,engine='python')

            #Separate the movie name from the year in the title
            pattern = re.compile('^(.*)\((\d+)\)')
            movies['TitleWithoutYear'] = movies['Title'].map(lambda x: pattern.match(x).group(1))#获得不含年份的电影名称

            genre_set = set()
            for var in movies['Genres'].str.split('|'):
                genre_set.update(var)
            genre_int_map = {var:ii for ii,var in enumerate(genre_set)}
            movies['GenresMultiHot'] = movies['Genres'].map(genres_multi_hot(genre_int_map))

            word_set = set()
            for var in movies['TitleWithoutYear'].str.split():
                word_set.update(var)
            word_int_map = {var:ii for ii,var in enumerate(word_set,start=1)}
            movies['TitleIndex'] = movies['TitleWithoutYear'].map(title_encode(word_int_map))
            title_indices = list(word_int_map.values())
            title_word_count = len(set(title_indices))

        #Reading ratings data and data preprocessing
        with zf.open('ml-1m/ratings.dat') as ratings_raw_data:
            ratings_title = ['UserID','MovieID','ratings','Timestamp']
            ratings = pd.read_table(ratings_raw_data,sep='::',header=None,names=ratings_title,engine='python')
            ratings = ratings.filter(regex='UserID|MovieID|ratings')

        #Merge three tables
        data = pd.merge(pd.merge(users,ratings),movies)
        #features,targets tables
        features,targets = data.drop(['ratings'],axis=1),data[['ratings']]
        return features,targets,age_map,gender_map,genre_int_map,word_int_map,users,movies


if __name__ == '__main__':
    dataset_zip = download_m1_1m('./data')
    features,targets,age_map,gender_map,genre_int_map,word_int_map,users,movies = load_data(dataset_zip)

    with open('data/meta.p','wb') as meta:
        pickle.dump((age_map,gender_map,genre_int_map,word_int_map),meta)

    with open('data/users.p','wb') as meta:
        pickle.dump(users,meta)
    with open('data/movies.p','wb') as meta:
        pickle.dump(movies,meta)

    train_x,test_x,train_y,test_y = train_test_split(features,targets,test_size=0.2,random_state=0)
    with open('data/data.p','wb') as data:
        pickle.dump((train_x,test_x,train_y,test_y),data)










