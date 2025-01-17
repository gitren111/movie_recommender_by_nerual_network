import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding,Dropout,LSTM
import numpy as np
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape,Multiply,Dot,Permute,GlobalAveragePooling1D
from dropout_rate import DROPOUT_RATE
"""
Step 3: Definition of the inference model structure. The main purpose is to first build user and movie feature networks 
through deep learning techniques, and then combine the two networks into a fusion network to achieve the prediction of user ratings.
1、The neural network techniques used include: Embedding layer (Embedding) for processing discrete features, Dense layer (Dense) 
for feature transformation and extraction, LSTM for processing sequence information of movie titles, Dropout for preventing overfitting, 
matrix multiplication and weighted average pooling for feature fusion, and Dense layer for rating prediction.
"""

EMBED_DIM = 32
USER_ID_COUNT = 6041
GENDER_COUNT = 2
AGE_COUNT = 7
JOB_COUNT = 21
MOVIE_ID_COUNT = 3953
MOVIE_GENRES_COUNT = 18
MOVIE_TITLE_WORDS_COUNT = 5217
BATCH_SIZE = 256
LSTM_UNIT_NUM = 128

def user_feature_network(user_id,user_gender,user_age,user_job,dropout_rate):

    dropout_rate = DROPOUT_RATE

    #users embed layer
    user_id_embed_layer = Embedding(USER_ID_COUNT,EMBED_DIM,name='user_id_embed')(user_id)
    user_id_embed_layer = Reshape((EMBED_DIM,))(user_id_embed_layer)

    user_gender_embed_layer = Embedding(GENDER_COUNT, EMBED_DIM // 2, name='user_gender_embed')(user_gender)
    user_gender_embed_layer = Reshape((EMBED_DIM // 2,))(user_gender_embed_layer)

    user_age_embed_layer = Embedding(AGE_COUNT,EMBED_DIM // 2,name='user_age_embed')(user_age)
    user_age_embed_layer = Reshape((EMBED_DIM // 2,))(user_age_embed_layer)

    user_job_embed_layer = Embedding(JOB_COUNT, EMBED_DIM // 2, name='user_job_embed')(user_job)
    user_job_embed_layer = Reshape((EMBED_DIM // 2,))(user_job_embed_layer)


    #users fully Connected Layer+dropout

    user_id_fc_layer = Dense(EMBED_DIM,activation='relu',name='user_id_fc')(user_id_embed_layer)
    user_id_fc_layer = Dropout(dropout_rate,name='user_id_dropout')(user_id_fc_layer)

    user_gender_fc_layer = Dense(EMBED_DIM, activation='relu', name='user_gender_fc')(user_gender_embed_layer)
    user_gender_fc_layer = Dropout(dropout_rate, name='user_gender_dropout')(user_gender_fc_layer)

    user_age_fc_layer = Dense(EMBED_DIM, activation='relu', name='user_age_fc')(user_age_embed_layer)
    user_age_fc_layer = Dropout(dropout_rate, name='user_age_dropout')(user_age_fc_layer)

    user_job_fc_layer = Dense(EMBED_DIM, activation='relu', name='user_job_fc')(user_job_embed_layer)
    user_job_fc_layer = Dropout(dropout_rate, name='user_job_dropout')(user_job_fc_layer)

    #user_combine_feature
    user_combine_feature = Concatenate(axis=1,name='user_combine_feature')([user_id_fc_layer,
                                                                            user_gender_fc_layer,
                                                                            user_age_fc_layer,
                                                                            user_job_fc_layer])
    user_combine_fc_layer = Dense(200,activation='relu',name='user_combine_fc_layer')(user_combine_feature)

    return user_combine_fc_layer

def movie_feature_network(movie_id,movie_genres,movie_titles,movie_title_length,dropout_rate):

    dropout_rate = DROPOUT_RATE

    #movie_id
    movie_id_embed_layer = Embedding(MOVIE_ID_COUNT,EMBED_DIM,name='movie_id_embed_1')(movie_id)
    movie_id_embed_layer = Reshape((EMBED_DIM,))(movie_id_embed_layer)
    movie_id_embed_layer = Dense(EMBED_DIM, activation='relu', name='movie_id_embed_2')(movie_id_embed_layer)
    movie_id_embed_layer = Dropout(dropout_rate,name='movie_id_embed')(movie_id_embed_layer)

    #movie_genres
    movie_genres_embed_layer = Dense(EMBED_DIM,use_bias=False,activation='relu',name='movie_genres_embed_layer1')(movie_genres)
    movie_genres_embed_layer = Dropout(dropout_rate,name='movie_genres_embed_layer')(movie_genres_embed_layer)


    #movie_titles
    embedding_layer = Embedding(input_dim=MOVIE_TITLE_WORDS_COUNT, output_dim=EMBED_DIM, name="movie_embedding")
    movie_title_embed_layer = embedding_layer(movie_titles)
    lstm_layer = tf.keras.layers.LSTM(
        LSTM_UNIT_NUM,  # LSTM单元数
        return_sequences=True,  # 如果需要输出所有时间步的输出
        return_state=True,  # 如果需要返回最终的状态（h和c）
        dropout=dropout_rate,  # 使用的dropout率
        recurrent_dropout=dropout_rate  # 对递归连接的dropout率
    )

    lstm_output, state_h, state_c = lstm_layer(movie_title_embed_layer)
    lstm_output_pooled = GlobalAveragePooling1D()(lstm_output)
    movie_title_length_expanded = Dense(LSTM_UNIT_NUM,use_bias=False,activation=None)(movie_title_length)
    lstm_output = Multiply()([lstm_output_pooled,1.0/movie_title_length_expanded])

    #movie_combine_feature
    movie_combine_feature = Concatenate(axis=1,name='movie_combine_feature')([movie_id_embed_layer,
                                                                              movie_genres_embed_layer,
                                                                              lstm_output])
    movie_combine_layer = Dense(200,activation='relu',name='movie_fc_layer')(movie_combine_feature)

    return movie_combine_layer

def full_network(user_input,movie_input,dropout_rate):

    dropout_rate = DROPOUT_RATE

    user_combine_fc_layer = user_feature_network(*user_input,dropout_rate)
    movie_combine_layer = movie_feature_network(*movie_input,dropout_rate)

    #Merge user and movie features
    input_layer = Concatenate(axis=1, name='user_movie_fc')([user_combine_fc_layer,
                                                             movie_combine_layer])

    predicted = Dense(1, name='prediction')(input_layer)

    return user_combine_fc_layer,movie_combine_layer,predicted

def trainable_variable_summaries():
    for variable in tf.trainable_variables():
        name = variable.name.split(':')[0]
        tf.summary.histogram(name,variable)
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean/' + name,mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable -mean)))
        tf.summary.scalar('stddev/' + name,stddev)


