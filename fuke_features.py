import os.path
import tensorflow as tf
import numpy as np
import pickle


"""
第六步：提取特征
一、传入模型
二、提取中间层
三、保存中间层
"""
def extract_trained_features(user_data, movie_data,target, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    user_combined_feature_output = tf.keras.Model(inputs=model.input, outputs=model.get_layer('user_combine_fc_layer').output)
    movie_combine_feature_output = tf.keras.Model(inputs=model.input,outputs=model.get_layer('movie_fc_layer').output)

    user_movie_fc_output = tf.keras.Model(inputs=model.input,outputs=model.get_layer('prediction').output)

    user_id = user_data['user_id']
    user_gender = user_data['user_gender']
    user_age = user_data['user_age']
    user_job = user_data['user_job']

    movie_id = movie_data['movie_id']
    movie_genres = movie_data['movie_genres']
    movie_titles = movie_data['movie_titles']
    movie_title_length = movie_data['movie_title_length']

    combined_features = [user_id, user_gender, user_age, user_job, movie_id, movie_genres, movie_titles, movie_title_length, target]

    # Extract user and movie features
    user_features_1 = user_combined_feature_output.predict(combined_features)  # 用户特征
    movie_features_1 = movie_combine_feature_output.predict(combined_features)  # 电影特征

    user_features={}
    movie_features={}
    for idx,user_id_val in enumerate(user_id):
        user_features[user_id_val[0]] = user_features_1[idx]

    for idx,movie_id_val in enumerate(movie_id):
        movie_features[movie_id_val[0]] = movie_features_1[idx]

    #Extract kernel and bias
    weights = user_movie_fc_output.get_weights()
    #print(len(weights))
    kernel = weights[25]
    bias = weights[26]
    #print(len(weights))
    #for i, weight in enumerate(weights):
    #    print(f"Weight {i} shape: {weight.shape}")

    if os.path.exists('data/user_features.p'):
        print('The user_features file already exists and does not need to be saved')
    else:
        print("Saving user features...",)
        with open('data/user_features.p', 'wb') as uf:
            pickle.dump(user_features, uf)
            print("User features saved")

    if os.path.exists('data/movie_features.p'):
        print('The movie_features file already exists and does not need to be saved')
    else:
        print("Saving movie features...")
        with open('data/movie_features.p', 'wb') as mf:
            pickle.dump(movie_features, mf)
            print("Movie features saved.")

    if os.path.exists('data/user_movie_fc_param.p'):
        print('The weight and bias features already exist and do not need to be saved.')
    else:
        print("Saving weight and bias features....")
        with open('data/user_movie_fc_param.p','wb') as param:
            pickle.dump((kernel, bias), param)
            print("Weight and bias features saved.")


if __name__ == '__main__':
    with open('data/data.p','rb') as data:
        train_x,_,train_y,_ = pickle.load(data,encoding='utf-8')

user_data = {
    'user_id': np.reshape(np.array(train_x['UserID'], dtype=np.int32), (-1, 1)),
    'user_gender': np.reshape(np.array(train_x['GenderIndex'], dtype=np.int32), (-1, 1)),
    'user_age': np.reshape(np.array(train_x['AgeIndex'], dtype=np.int32), (-1, 1)),
    'user_job': np.reshape(np.array(train_x['JobID'], dtype=np.int32), (-1, 1))
}

movie_title_length = (np.array(list(train_x['TitleIndex']), dtype=np.int32)!= 0).sum(axis=1)
movie_data = {
    'movie_id': np.reshape(np.array(train_x['MovieID'], dtype=np.int32), (-1, 1)),
    'movie_genres': np.array(list(train_x['GenresMultiHot']), dtype=np.float32),
    'movie_titles': np.array(list(train_x['TitleIndex']), dtype=np.int32),
    'movie_title_length': np.reshape(movie_title_length ,(-1, 1))
}

target = np.reshape(np.array(train_y['ratings'], dtype=np.float32), (-1, 1))

extract_trained_features(user_data, movie_data,target, 'data/model/latest_model.keras')

