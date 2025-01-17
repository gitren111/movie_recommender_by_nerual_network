# 在MovieLens 1M数据集上使用深度学习进行评分预测

## MovieLen 1M数据及简介
MovieLens 1M数据集包含包含6000个用户在近4000部电影上的100万条评分，也包括电影元数据信息和用户属性信息。下载地址为：   
[http://files.grouplens.org/datasets/movielens/ml-1m.zip](http://files.grouplens.org/datasets/movielens/ml-1m.zip)   
数据集分为三个文件：电影元数据信息（movie.dat）、用户属性信息（users.dat)和用户评分数据（ratings.dat)。

### 电影元数据
电影元数据的格式为：MovieID::Title::Genres。
- Title：电影名（包括发布年份）
- Genres：多种电影题材由是“|”分隔，题材种类有以下18种：
	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western
- MovieID：
### 用户属性信息
用户属性信息的格式为：UserID::Gender::Age::Occupation::Zip-code。

- Gender：“M”表示男，“F”表示女
- Age:年龄值有以下几种：
	*  1:  “小于18岁”
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"
- Occupation：职业有以下几种：
	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"
### 电影评分
电影评分的格式为：UserID::MovieID::Rating::Timestamp   
- UserID: [1-6040]  
- MovieIDs:[1-3952]  
- Ratings:1-5的整数  
- Timestamp：时间戳  
- 每个用户至少有20个评分  

## 数据预处理与网络模型设计
- 对数据集的数据进行初步处理，分别将用户数据和电影数据保存，并将数据拆分为训练集和测试集；  
- 搭建神经网络模型，首先分别搭建用户特征网络和电影特征网络，使用嵌入层处理离散特征，然后使用全连接层进行特征变换和特征提取，并进行Dropout正则化，其中movie_title用LSTM模型来处理电影标题的序列信息，学习单词上下文的关系，并用平均池化方法，排除0填充和标题长度对特征影响，接着将用户特征网络和电影特征网络进行融合，通过全连接层和Relu激活函数学习交互关系，最终将融合的用户-电影特征网络通过只有一个神经元的全连接层，输出评分预测值
### 模型py文件及顺序简介
- pre_process.py:数据下载、解压、初步处理和保存  
- fuke_dataset.py:数据加载和管理，为模型提供批量化的数据  
- fuke_inference.py:神经网络模型  
- fuke_train.py:模型训练  
- fuke_test.py:模型测试  
- fuke_features.py:特征提取  
- fuke_predicted.py:模型预测  

### 数据预处理
MovieLens数据集中，用户特征中UserID、Gender、Age、Job以及电影特征中MovieID都可以认为是类别型数据，通常使用One-Hot编码。
但是MovieID和UserID值得类型比较多，如果使用One-Hot编码，每个值都会被编码成一个维数很高的稀疏向量，作为神经网络输入是计算量很大。
除此之外，采用One-Hot编码，不同属性值的距离都是相等的， 比如“小于18岁”和“35-44”与“56+”与“25-44”的距离平方都是2。
所以在数据预处理阶段，我们不使用One-Hot编码，而仅仅将这些数据编码成数字，用这些数据当作嵌入矩阵的索引。
神经网络的第一层使用嵌入层，嵌入矩阵通过学习得到。

电影题材和电影名比较特殊，他们可以视作多值属性，且长度不等。对于电影题材，因为类型不多，可以直接使用Multi-Hot编码，
在神经网络中通过编码后的向量与嵌入矩阵相乘实现不同长度的输入。对于电影名的处理稍微复杂一点，首先创建word->int的映射字典，
然后使用数字列表编码，并填充为相同的长度，经过一个LSTM网络，并对网络的所有输出求均值(平均池化）得到电影名特征。

- UserID、Occupation、MovieID不变
- Gender字段：需要将‘F’和‘M’转换成0和1
- Age字段：转成7个连续数字0-6
- Genres字段：多值属性，使用Multi-Hot编码，维数为18
- Title字段：创建word->int的映射字典，然后使用数字列表编码，并填充为相同的长度，维数为15

数据预处理的完整代码见项目中的[pre_process.py](pre_process.py)

#### 电影题材的multi-hot编码函数
```
def genres_multi_hot(genre_int_map):
    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split('|')]
        multi_hot = np.zeros(len(genre_int_map))
        multi_hot[genre_int_list] = 1
        return multi_hot
    return helper
```

#### 电影数字列表编码函数
```python
def title_encode(word_int_map):
    def helper(title):
        title_words = [word_int_map[word] for word in title.split()]
        if len(title_words) > 15:
            return np.array(title[:15])
        else:
            title_vector = np.zeros(15)
            title_vector[:len(title_words)] = title_words
            return title_vector

    return helper
```
#### 数据预处理函数
```python
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
```
### 模型设计
![network](../network.png)
本文使用的网络模型如上图所示。网络可以分成两大部分，分别是用户特征网络和电影特征网络，这两个子网络最终通过全连接层输出一个200维的向量，作为用户特征和电影特征。
有了用户特征向量和电影特征向量之后，就可以通过各种方式拟合评分，本文中将两个输入通过只有一个神经元的全连接层，将输出作为评分,
将MSE作为损失函数去优化网络。

#### 用户特征网络
UserID和Age、Gender、Job的处理方式相同，首先将输入作为索引从嵌入矩阵中取出对应的特征向量，其中UserID编码为32维向量，其他特征编码为16维向量。
然后分别在其后添加一个全连接层和一个dropout层，全连接层的神经元个数为32。最后将得到的四个32维的向量拼接到一起形成一个128维的向量，作为全连接层的输入，最后输出一个200维的用户特征向量。
#### 用户特征网络核心代码
```python
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
```

#### 电影特征网络
MovieID和Genres的处理方式与用户特征方式一样。  

Title中通过嵌入层之后编码为[n，15，32]三维稠密矩阵，15体现了时间步的关系，32代表电影名称的每个单词都是32维特征向量来表示，然后通过一层隐层为128个神经元的LSTM，然后对这15个LSTM单元的输出求平均值，并除以电影名称长度（排除0填充和标题长度对特征影响），最终得到一个128维特征向量。
将其与MovieID和Genres输出向量拼接到一起作为全连接层输入，最后得到一个200维向量，作为电影特征向量。

需要注意的是，虽然预处理阶段填充之后的标题长度都是15，但在实际计算时使用Keras LSTM层，学习电影名称基于上下文的信息，并进行平均池化和按序列真实长度进行归一化，实现排除0填充影响和对标题长度对特征影响
。
#### 电影特征网络核心代码
```python
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
```
#### 损失层核心代码
```python
def full_network(user_input,movie_input,dropout_rate):

    dropout_rate = DROPOUT_RATE

    user_combine_fc_layer = user_feature_network(*user_input,dropout_rate)
    movie_combine_layer = movie_feature_network(*movie_input,dropout_rate)

    #Merge user and movie features
    input_layer = Concatenate(axis=1, name='user_movie_fc')([user_combine_fc_layer,
                                                             movie_combine_layer])

    predicted = Dense(1, name='prediction')(input_layer)

    return user_combine_fc_layer,movie_combine_layer,predicted
```

## 模型训练
将数据集按照0.8和0.2比例随机分成了训练集和测试集，在训练集中按照0.8和0.2比例随机分成了训练集和验证集，每1500批次就在验证集里做一次验证，只有在验证损失低于记录的损失时才保存模型，并且对模型训练的位置进行记录，方便下次训练时可以从上次训练基础上继续训练。模型中设置学习率调度器，让学习率随着训练批次增加能够逐渐衰减，能让模型一开始迅速收敛，然后通过学习率衰减学习更加细微的特征变化。


## 实验结果
将数据集按照0.8和0.2的比例随机分成了训练集和测试集，经过5个epoch的训练之后得到最终模型，在测试集上测试结果,MSE在0.84左右。
下面是某次运行的结果
```
2024-12-02 17:50:03,892 - INFO - Batch  773/782   test_loss = 0.878
2024-12-02 17:50:03,954 - INFO - Batch  774/782   test_loss = 0.856
2024-12-02 17:50:04,021 - INFO - Batch  775/782   test_loss = 0.792
2024-12-02 17:50:04,069 - INFO - Batch  776/782   test_loss = 0.794
2024-12-02 17:50:04,150 - INFO - Batch  777/782   test_loss = 0.756
2024-12-02 17:50:04,201 - INFO - Batch  778/782   test_loss = 0.861
2024-12-02 17:50:04,239 - INFO - Batch  779/782   test_loss = 0.753
2024-12-02 17:50:04,282 - INFO - Batch  780/782   test_loss = 0.970
2024-12-02 17:50:04,304 - INFO - Batch  781/782   test_loss = 1.117
2024-12-02 17:50:04,304 - INFO - 模型测试的MSE损失值是0.883
```
## 特征提取与评分预测
由于用户属性信息和电影元数据信息都是静态数据，模型训练好之后可以离线计算用户特征、电影特征，然后存储起来供评分预测和推荐使用。

### 特征提取核心代码
```python
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
    user_features_1 = user_combined_feature_output.predict(combined_features)  
    movie_features_1 = movie_combine_feature_output.predict(combined_features) 

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
```
### 评分以及用户、电影相似度计算
离线存储特征和参数之后，可以直接计算评分而不需要使用Tensorflow去定义网络。除了预测评分之后，也可以通过特征计算最相似的用户和电影。

```python
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
```

输出结果：

```
UserId=1,MovieId=1193,Rating=3.548
UserId=5900,MovieId=3100,Rating=3.555
这些用户与用户[{'UserID': 5900, 'Gender': 'M', 'Age': 25, 'JobID': 7}]最相似
最相似的用户{'UserID': 595, 'Gender': 'M', 'Age': 25, 'JobID': 7},相似度：0.9994
最相似的用户{'UserID': 26, 'Gender': 'M', 'Age': 25, 'JobID': 7},相似度：0.9991
最相似的用户{'UserID': 4487, 'Gender': 'M', 'Age': 25, 'JobID': 7},相似度：0.9990
最相似的用户{'UserID': 3973, 'Gender': 'M', 'Age': 25, 'JobID': 7},相似度：0.9990
最相似的用户{'UserID': 2809, 'Gender': 'M', 'Age': 25, 'JobID': 7},相似度：0.9990
这些电影与电影[{'MovieID': 1401, 'Title': 'Ghosts of Mississippi (1996)', 'Genres': 'Drama'}]最相似
最相似的电影{'MovieID': 2272, 'Title': 'One True Thing (1998)', 'Genres': 'Drama'}，相似度：0.9953
最相似的电影{'MovieID': 1302, 'Title': 'Field of Dreams (1989)', 'Genres': 'Drama'}，相似度：0.9947
最相似的电影{'MovieID': 2329, 'Title': 'American History X (1998)', 'Genres': 'Drama'}，相似度：0.9943
最相似的电影{'MovieID': 1957, 'Title': 'Chariots of Fire (1981)', 'Genres': 'Drama'}，相似度：0.9941
最相似的电影{'MovieID': 2420, 'Title': 'Karate Kid, The (1984)', 'Genres': 'Drama'}，相似度：0.9931
```

## 下一步工作
1. 使用更多的特征，进一步降低MSE
    - 用户属性数据中的Zip-code可以标识用户所处地区，不同地域的人可能有不同的喜好，应该是一个有用处的特征。
    - 时间特征：电影名中的上映时间，不同时代的电影，评分可能略有差异；用户评分时间距电影上映时间可能也会影响评分。
2. 使用得到的特征做电影推荐