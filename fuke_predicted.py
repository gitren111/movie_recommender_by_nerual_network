import pickle
import numpy as np
import tensorflow as tf

"""
第七步：模型预测
实现推荐系统，包括用户评分预测和电影、用户相似度计算功能
"""

def relu(x):
    s = tf.maximum(x,0)
    return s

def predict_rating(user_feature,movie_feature,kernel,basis,activate):
    feature = np.concatenate((user_feature,movie_feature))
    xw_b = np.dot(feature,kernel) + basis
    output = activate(xw_b)
    return output

def cosine_similarity(vec_left,vec_right):
    num = np.dot(vec_left,vec_right)
    denom = np.linalg.norm(vec_left) * np.linalg.norm(vec_right)
    cos = -1 if denom == 0 else num / denom
    return cos

def similar_movie(movie_id,top_k,movie_features):
    cosine_similarities = {}
    movie_feature_i = movie_features[movie_id]
    for movie_id_,movie_feature_ in movie_features.items():
        if movie_id_ != movie_id:
            cosine_similarities[movie_id_] = cosine_similarity(movie_feature_i,movie_feature_)
    return sorted(cosine_similarities.items(),key=lambda item:item[1],reverse=True)[:top_k]

def similar_user(user_id,top_k,user_features):
    cosine_similarities = {}
    user_feature_i = user_features[user_id]
    for user_id_,user_feature_ in user_features.items():
        if user_id_ != user_id:
            cosine_similarities[user_id_] = cosine_similarity(user_feature_i,user_feature_)
    return sorted(cosine_similarities.items(),key=lambda item:item[1],reverse=True)[:top_k]

if __name__ == '__main__':
    with open('data/user_features.p','rb') as uf:
        user_features = pickle.load(uf,encoding='utf-8')

    with open('data/movie_features.p','rb') as mf:
        movie_features = pickle.load(mf,encoding='utf-8')

    with open('data/user_movie_fc_param.p','rb') as params:
        kernel,bias = pickle.load(params,encoding='utf-8')

    with open('data/users.p','rb') as usr:
        users = pickle.load(usr,encoding='utf-8')

    with open('data/movies.p','rb') as mv:
        movies = pickle.load(mv,encoding='utf-8')

    rating1 = predict_rating(user_features[12],movie_features[517],kernel,bias,relu)
    print('UserId={},MovieId={},Rating={:.3f}'.format(1,1193,rating1[0]))

    rating2 = predict_rating(user_features[521], movie_features[95], kernel, bias, relu)
    print('UserId={},MovieId={},Rating={:.3f}'.format(5900, 3100, rating2[0]))

    similar_users = similar_user(5900,5,user_features)
    print('这些用户与用户{}最相似'.format(str(users[users['UserID'] == 5900][['UserID','Gender','Age','JobID']].to_dict('records'))))
    for user in similar_users:
        print('最相似的用户{},相似度：{:.4f}'.format(users[users['UserID'] == user[0]][['UserID','Gender','Age','JobID']].to_dict('records')[0],user[1]))

    similar_movies = similar_movie(1401,5,movie_features)
    print('这些电影与电影{}最相似'.format(str(movies[movies['MovieID'] == 1401][['MovieID','Title','Genres']].to_dict('records'))))
    for movie in similar_movies:
        print('最相似的电影{}，相似度：{:.4f}'.format(movies[movies['MovieID'] == movie[0]][['MovieID', 'Title', 'Genres']].to_dict('records')[0],movie[1]))















