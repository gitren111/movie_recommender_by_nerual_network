import logging
import os
import pickle
import tensorflow as tf
from fuke_dataset import Dataset, decompression_feature
from fuke_inference import full_network, trainable_variable_summaries
import numpy as np
from sklearn.metrics import mean_squared_error
from dropout_rate import DROPOUT_RATE

"""
Step 4: Deep Learning Model Training
The main functions include defining placeholders for input data, constructing deep neural networks, defining loss functions, 
performing backpropagation optimization, and saving the model regularly.
"""

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

BATCH_SIZE = 256
EPOCH = 6
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.95
SHOW_LOG_STEPS = 100
SAVE_MODEL_STEPS = 500
MOVIE_TITLE_WORDS_COUNT = 5217
VALIDAION_SPLIT = 0.2


# train model
def train(train_x, train_y, save_dir):  # save_dir模型保存目录
    # Input layer data definition
    user_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_id')
    user_gender = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_gender')
    user_age = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_age')
    user_job = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_job')

    movie_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='movie_id')
    movie_genres = tf.keras.Input(shape=(18,), dtype=tf.float32, name='movie_genres')
    movie_titles = tf.keras.Input(shape=(15,), dtype=tf.int32, name='movie_titles')
    movie_title_length = tf.keras.Input(shape=(1,), dtype=tf.int32,
                                        name='movie_title_length')

    targets = tf.keras.Input(shape=(1,), dtype=tf.int32, name='targets')  # 评分标签列

    dropout_rate = DROPOUT_RATE

    # Path for saving the latest model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    latest_weights_path = os.path.join(save_dir, "latest_model.keras")
    batch_index_path = os.path.join(save_dir, 'batch_index_pkl')
    best_loss_path = os.path.join(save_dir, 'best_loss_pkl')

    #Check whether there is a saved weight file
    if os.path.exists(latest_weights_path):
        print("Discover the optimal weight file and load the optimal weights to continue training...")
        model = tf.keras.models.load_model(latest_weights_path, safe_mode=False)

        #Read the original model training, and the learning rate and number of steps can be set
        arti_lr = 0.001
        arti_iter = 20000

        # Restore the global step of the model
        if hasattr(model.optimizer, 'iteration'):
            initial_iteration = model.optimizer.iterations.numpy()
        else:
            initial_iteration = arti_iter

        # learning rate ExponentialDecay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=arti_lr,
            decay_steps=3100,
            decay_rate=LEARNING_RATE_DECAY,
            staircase=True
        )
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0),
                      loss=tf.keras.losses.MeanSquaredError())

        # Restore the step count of the model to the state when it was saved during the last training
        model.optimizer.iterations.assign(initial_iteration)
        print(f'read global step，training start from{initial_iteration}step')

    else:
        print("No model file found, start training from scratch...")
        user_input = [user_id, user_gender, user_age, user_job]
        movie_input = [movie_id, movie_genres, movie_titles, movie_title_length]
        _, _, predicted = full_network(user_input, movie_input, dropout_rate)
        model = tf.keras.Model(inputs=user_input + movie_input + [targets], outputs=predicted)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE_BASE,
            decay_steps=3100,
            decay_rate=LEARNING_RATE_DECAY,
            staircase=True
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0),
                      loss=tf.keras.losses.MeanSquaredError())
        # model.summary()

    #Split the training set and validation set in proportion
    validation_split_index = int(len(train_y) * (1 - VALIDAION_SPLIT))
    train_x_split = train_x[:validation_split_index]
    train_y_split = train_y[:validation_split_index]
    val_x_split = train_x[validation_split_index:]
    val_y_split = train_y[validation_split_index:]

    dataset = Dataset(train_x_split.values, train_y_split.values)
    val_dataset = Dataset(val_x_split.values, val_y_split.values)

    # Load the saved training state and restore the index position of the Dataset object
    if os.path.exists(batch_index_path):
        with open(batch_index_path, 'rb') as f:
            state = pickle.load(f)
            dataset._current = min(state['current'], len(dataset._index))  # 确保 current 不超过索引范围
            dataset._epoch = state['epoch']
            dataset._index = state['index']
            print(f'Resume training again and continue training from the index {dataset._current} of the {dataset._epoch}th round')

            #Verify the validity of the restored index
            if not all(0 <= i < dataset.size for i in dataset._index):
                print("The restored index is invalid. Reset the index...")
                dataset._index = list(range(dataset.size))
    else:
        print("No index file found. Start training from the beginning...")

    batch_per_epoch = (len(train_y_split) + BATCH_SIZE - 1) // BATCH_SIZE  # 训练批次向上取证
    var_batch_per_epoch = (len(val_y_split) + BATCH_SIZE - 1) // BATCH_SIZE  # 训练批次向上取整
    y_true = []
    y_pred = []

    if os.path.exists(best_loss_path):
        with open(best_loss_path, 'rb') as b:
            best_loss_state = pickle.load(b)
            best_loss = best_loss_state['best_loss']
            print("The optimal loss file is found, and the optimal loss is loaded successfully...")
    else:
        print("No optimal loss file is found, and the best_loss is initialized to a maximum value...")
        best_loss = float('inf')

    #Training Loop
    for epoch_i in range(EPOCH):
        print('Now loop for{}rounds'.format(epoch_i))
        logging.info('Epoch{}/{}'.format(epoch_i + 1, EPOCH))
        for batch_i in range(batch_per_epoch):
            if batch_i % 100 == 0:
                print('{}th round，Loop through{}batches'.format(epoch_i, batch_i))

                if isinstance(model.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    current_lr = model.optimizer.learning_rate(model.optimizer.iterations).numpy()
                else:
                    current_lr = model.optimizer.learning_rate.numpy()
                print(f"batches {batch_i}, global step {model.optimizer.iterations.numpy()},current_lr: {current_lr}")

            #Acquire feature and label data in batches from the dataset
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

            ys = tf.reshape(ys, (-1, 1))
            ys = np.array(ys, dtype=np.float32)

            train_loss = model.train_on_batch(feed, ys)

            if batch_i % SHOW_LOG_STEPS == 0:
                logging.info('Epoch{},Batch{}/{} - Train_loss:{:.4f}'.format(epoch_i + 1,
                                                                             batch_i + 1,
                                                                             batch_per_epoch,
                                                                             train_loss))

            predicted_ratings = model.predict(feed, verbose=0)

            y_true.extend(ys.flatten())
            y_pred.extend(predicted_ratings.flatten())

            #Validation on the validation set and saving the optimal model
            if batch_i == 1500:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                logging.info(f'rmse：{rmse:.4f}')

                val_avg_loss = 0
                for batch_i in range(var_batch_per_epoch):
                    if batch_i % 100 == 0:
                        print('The validation set loops through{}batches'.format(batch_i))
                    xs, ys = val_dataset.next_batch(BATCH_SIZE)
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
                        np.array(ys, dtype=np.float32)  # 目标值
                    ]

                    test_loss = model.evaluate(feed, ys, batch_size=BATCH_SIZE, verbose=0)
                    val_avg_loss += test_loss * len(users.id)

                val_avg_loss = val_avg_loss / val_dataset.size
                logging.info('The best_loss saved by the model is{:.4f}，the MSE loss value of the test is{:.4f}'
                             .format(best_loss, val_avg_loss))

                # Compare the validation loss with the optimal loss
                if val_avg_loss < best_loss:
                    best_loss = val_avg_loss
                    best_loss_state = {'best_loss': best_loss}
                    with open(best_loss_path, 'wb') as b:
                        pickle.dump(best_loss_state, b)
                        print(f'Save the latest best_loss{best_loss}')

                    model.save(latest_weights_path)
                    print(f'The model performs better on the current validation set, and a new optimal model has been saved：{latest_weights_path}')
                    # Save the current model state
                    state = {
                        'current': dataset._current,
                        'epoch': dataset._epoch,
                        'index': dataset._index if all(0 <= i < dataset.size for i in dataset._index) else list(
                            range(dataset.size))
                    }
                    with open(batch_index_path, 'wb') as f:
                        pickle.dump(state, f)
                    print(f'Save the training set: current epoch{dataset._epoch},batch {batch_i}，index {dataset._current}')
                    break
                else:
                    print(f'The current model validation loss is{val_avg_loss:.4f}，not lower than the minimum loss{best_loss:.4f}，model will not be saved')
                break


if __name__ == '__main__':
    with open('data/data.p', 'rb') as data:
        train_x, _, train_y, _ = pickle.load(data, encoding='utf-8')

    train(train_x, train_y, 'data/model/')







