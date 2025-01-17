import logging
import os
import pickle
import tensorflow as tf

from fuke_dataset import Dataset,decompression_feature
from fuke_inference import full_network
import numpy as np
from sklearn.metrics import mean_squared_error
from dropout_rate import DROPOUT_RATE



"""
Step 5: Testing the Deep Learning Model
Load the trained model weights and perform evaluation on the test set   
"""

#一、参数
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO,format=LOG_FORMAT)

BATCH_SIZE = 256
EPOCH = 1
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.97
SHOW_LOG_STEPS = 100
SAVE_MODEL_STEPS = 500
MOVIE_TITLE_WORDS_COUNT = 5217

def model_test(test_x,test_y,model_path):
    # 1、输入层数据定义
    user_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_id')
    user_gender = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_gender')
    user_age = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_age')
    user_job = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_job')

    movie_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='movie_id')
    movie_genres = tf.keras.Input(shape=(18,), dtype=tf.float32, name='movie_genres')
    movie_titles = tf.keras.Input(shape=(15,), dtype=tf.int32, name='movie_titles')
    movie_title_length = tf.keras.Input(shape=(1,), dtype=tf.int32,
                                        name='movie_title_length')
    targets = tf.keras.Input(shape=(1,), dtype=tf.int32, name='targets')

    dropout_rate = DROPOUT_RATE

    user_input = [user_id, user_gender, user_age, user_job]
    movie_input = [movie_id, movie_genres, movie_titles, movie_title_length]
    _, _, predicted = full_network(user_input, movie_input, dropout_rate)

    model = tf.keras.Model(inputs=user_input + movie_input + [targets], outputs=predicted)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_BASE),
                  loss=tf.keras.losses.MeanSquaredError())

    dataset = Dataset(test_x.values, test_y.values)
    batch_per_epoch = (len(test_x) + BATCH_SIZE - 1) // BATCH_SIZE  # 训练批次向上取整

    model = tf.keras.models.load_model(model_path, safe_mode=False)# 加载整个模型
    #model.summary()

    avg_loss = 0
    y_true = []
    y_pred = []

    for batch_i in range(batch_per_epoch):
        if batch_i % 100 == 0:
            print('batch{}'.format( batch_i))
        xs, ys = dataset.next_batch(BATCH_SIZE)
        users, movies = decompression_feature(xs)

        feed = [
            np.array(users.id, dtype=np.int32),
            np.array(users.gender, dtype=np.int32),
            np.array(users.age, dtype=np.int32),
            np.array(users.job, dtype=np.int32),
            np.array(movies.id, dtype=np.int32),
            np.array(movies.genres, dtype=np.float32),
            np.array(movies.titles, dtype=np.int32),
            np.array(movies.title_length, dtype=np.int32),
            np.array(ys, dtype=np.float32)
        ]


        predicted_ratings = model.predict(feed, verbose=0)

        test_loss = model.evaluate(feed,ys,batch_size=BATCH_SIZE, verbose=0)
        avg_loss += test_loss*len(users.id)
        y_true.extend(ys.flatten())
        y_pred.extend(predicted_ratings.flatten())

        show_message = 'Batch{:>4}/{}  test_loss = {:.4f}'.format(batch_i,batch_per_epoch,test_loss)
        logging.info(show_message)

    avg_loss = avg_loss / dataset.size
    logging.info('model test MSE{:.4f}'.format(avg_loss))
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    logging.info(f'model test rmse：{rmse:.4f}')


if __name__ == '__main__':
    with open('data/data.p', 'rb') as data:
        _,test_x,_,test_y = pickle.load(data,encoding='utf-8')
    print(f"test_x columns: {test_x.columns.tolist()}")

    model_path = 'data/model/latest_model.keras'
    model_test(test_x,test_y,model_path)












