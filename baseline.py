import pandas as pd
import numpy as np 
from collections import Counter
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES']='0'

def read_data(file_path: str, if_train):
    trunk = pd.read_csv(file_path, sep='\t')
    feature_name = ['user_ID','message_ID','refresh','time_range','recall_type','app_name','ctr','up_num','down_num','increase','storage_time_to_today',
                        'sample_date','click_nums','message_like_listC','message_like_listD','message_like_listB','message_like_listH','message_like_listA',
                        'messageD1_seq','messageD2_seq','messageD3_seq','messageD4_seq','messageD5_seq','messageC1_seq','messageC2_seq',
                        'messageC3_seq','messageC4_seq','messageC5_seq','mask_seq','realtime_messageC_seq','messageA_seq','messageB_seq','messageC_seq',
                        'messageD_seq','messageE_seq','messageF_seq','messageG_seq','messageH_seq']
    if if_train:
        trunk.columns = feature_name+['label']
    else:
        trunk.columns = feature_name  
    return trunk

# 获取每个序列里面点击元素集合的长度
def seq_value_counts(seq):
    seq = seq.split(',')
    seq_set = set(seq)#len(seq)
    return len(seq_set)
# mask特征转换成点击次数
def get_mask_feature_sum(mask):
    mask = mask.split(',')
    mask = [int(i) for i in mask]
    return sum(mask)
# 构造序列字典
def get_seq_dic(df, seq_name, thresh_hold=100):
    seq_dic = {}
    for j in df[seq_name].values:
        j = j.split(',')
        for i in j:
            if i in seq_dic:
                seq_dic[i]+=1
            else:
                seq_dic[i]=1
    seq_set = [i for i in seq_dic if seq_dic[i]>=thresh_hold]
    return dict(zip(list(seq_set), range(1,len(seq_set)+1)))
# 序列长度为一的进行长尾截断处理
def get_single_value_dic(df, seq_name, thresh_hold=5000):
    seq_dic = {}
    for j in df[seq_name].values:
        if j in seq_dic:
            seq_dic[j]+=1
        else:
            seq_dic[j]=1
    seq_set = [i for i in seq_dic if seq_dic[i]>=thresh_hold]
    return dict(zip(list(seq_set), range(1,len(seq_set)+1)))

# 序列长度为一的进行元素转换
def list_turn(x,x_dic):
    if x in x_dic:
        x = x_dic[x]
    else : x = 0
    return x

# 序列转换成字典里面的元素
def get_convert_data(df, seq_name, seq_dic):
    seq = []
    for i in df[seq_name].values:
        i = i.split(',')
        l = len(i)
        seq_id = map(lambda x: seq_dic[x] if x in seq_dic else 0, i)

        seq.append(list(seq_id))
    return np.array(seq)
# 获取每个序列的字典
def get_seq_list(df):
    message_like_listC_dic = get_seq_dic(df, 'message_like_listC')
    message_like_listD_dic = get_seq_dic(df, 'message_like_listD')
    message_like_listB_dic = get_seq_dic(df, 'message_like_listB')
    message_like_listH_dic = get_seq_dic(df, 'message_like_listH')
    message_like_listA_dic = get_seq_dic(df, 'message_like_listA')

    messageDD_seq_dic = get_seq_dic(df, 'messageDD_seq')
    messageCC_seq_dic = get_seq_dic(df, 'messageCC_seq')

    realtime_messageC_seq_dic = get_seq_dic(df, 'realtime_messageC_seq')

    messageA_seq_dic = get_seq_dic(df, 'messageA_seq')
    messageC_seq_dic = get_seq_dic(df, 'messageC_seq')
    messageD_seq_dic = get_seq_dic(df, 'messageD_seq')
    messageE_seq_dic = get_seq_dic(df, 'messageE_seq')
    messageF_seq_dic = get_seq_dic(df, 'messageF_seq')
    messageG_seq_dic = get_seq_dic(df, 'messageG_seq')
    return [message_like_listC_dic, message_like_listD_dic, message_like_listB_dic,
            message_like_listH_dic, message_like_listA_dic, messageDD_seq_dic, messageCC_seq_dic,
            realtime_messageC_seq_dic, messageA_seq_dic, messageC_seq_dic, messageD_seq_dic,
            messageE_seq_dic, messageF_seq_dic, messageG_seq_dic]

train_file = '/home/ryne/Downloads/data/train/train_data.csv'
test_file = '/home/ryne/Downloads/data/test/test_data.csv'

train_data = read_data(train_file, True)
test_data = read_data(test_file, False)

print("load data finised, train_data shape[0] is {}".format(train_data.shape[0]))

seq_name_list = [
    'message_like_listC','message_like_listD','message_like_listB','message_like_listH','message_like_listA',
    'messageD1_seq','messageD2_seq','messageD3_seq','messageD4_seq','messageD5_seq',
    'messageC1_seq','messageC2_seq','messageC3_seq','messageC4_seq','messageC5_seq',
    'realtime_messageC_seq','messageA_seq','messageC_seq',
    'messageD_seq','messageE_seq','messageF_seq','messageG_seq'
]

deep_name_list = [
    'message_like_listC','message_like_listD','message_like_listB','message_like_listH','message_like_listA',
    'messageDD_seq','messageCC_seq','realtime_messageC_seq','messageA_seq','messageC_seq',
    'messageD_seq','messageE_seq','messageF_seq','messageG_seq'
]
    
## 统计特征预处理
train_data['mask_seq'] = train_data['mask_seq'].apply(lambda x: get_mask_feature_sum(x))
test_data['mask_seq'] = test_data['mask_seq'].apply(lambda x: get_mask_feature_sum(x))

for i in seq_name_list:
    train_data[i+"_value_count"] = train_data[i].apply(lambda x: seq_value_counts(x))
    test_data[i+"_value_count"] = test_data[i].apply(lambda x: seq_value_counts(x))

train_data['ctr'] = train_data['ctr'].apply(lambda x: x if x>=0 else -1)
test_data['ctr'] = test_data['ctr'].apply(lambda x: x if x>=0 else -1)

train_data['message_mask_avg'] = train_data.groupby(['message_ID','sample_date'])['mask_seq'].transform('mean')
train_data['message_ctr_avg'] = train_data.groupby(['message_ID','sample_date'])['ctr'].transform('mean')
train_data['messageDD_seq'] = train_data['messageD1_seq']+","+"-1"+","+train_data['messageD2_seq']+","+"-1"+","+train_data['messageD3_seq']+","+"-1"+","+ train_data['messageD4_seq']+","+"-1"+","+train_data['messageD5_seq']
train_data['messageCC_seq'] = train_data['messageC1_seq']+","+"-1"+","+train_data['messageC2_seq']+","+"-1"+","+train_data['messageC3_seq']+","+"-1"+","+ train_data['messageC4_seq']+","+"-1"+","+train_data['messageC5_seq']

test_data['message_mask_avg'] = test_data.groupby(['message_ID','sample_date'])['mask_seq'].transform('mean')
test_data['message_ctr_avg'] = test_data.groupby(['message_ID','sample_date'])['ctr'].transform('mean')
test_data['messageDD_seq'] = test_data['messageD1_seq']+","+"-1"+","+test_data['messageD2_seq']+","+"-1"+","+test_data['messageD3_seq']+","+"-1"+","+ test_data['messageD4_seq']+","+"-1"+","+test_data['messageD5_seq']
test_data['messageCC_seq'] = test_data['messageC1_seq']+","+"-1"+","+test_data['messageC2_seq']+","+"-1"+","+test_data['messageC3_seq']+","+"-1"+","+ test_data['messageC4_seq']+","+"-1"+","+test_data['messageC5_seq']

messageB_seq_dic = get_single_value_dic(train_data, 'messageB_seq')
#'mask_seq','messageB_seq','messageH_seq'
train_data['messageB_seq'] = train_data['messageB_seq'].apply(lambda x: list_turn(x,messageB_seq_dic))
test_data['messageB_seq'] = test_data['messageB_seq'].apply(lambda x: list_turn(x,messageB_seq_dic))


## 模型结构
## wide dnn
wide_name_list = [i for i in train_data.columns if i not in seq_name_list+
                  ['messageDD_seq','messageCC_seq','label','sample_date']+
                  ['user_ID','message_ID']]

input_shape = len(wide_name_list)
wide_inp = keras.layers.Input(shape=(input_shape,), dtype='float32', name='wide_inp')
dnn_layer = keras.layers.Dense(64, activation = 'relu')(wide_inp)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
dnn_layer = keras.layers.Dense(32, activation = 'relu')(dnn_layer)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
w = keras.layers.Dense(1)(dnn_layer)

## deep 
embedding_size = 128
is_mask = True
input_dim_list = [50,50,20,5,10,29,54,50,8,10,7,100,16,100]
deep_name_list = [
    'message_like_listC','message_like_listD','message_like_listB','message_like_listH','message_like_listA',
    'messageDD_seq','messageCC_seq','realtime_messageC_seq','messageA_seq','messageC_seq',
    'messageD_seq','messageE_seq','messageF_seq','messageG_seq'
]
dic_list = get_seq_list(train_data)
inputs = []
embeddings = []
for i in range(len(input_dim_list)):
    input_word = keras.Input(shape=(input_dim_list[i],), name=deep_name_list[i])
    embedding_word = keras.layers.Embedding(output_dim=embedding_size, input_dim=len(dic_list[i]) + 1, input_length=input_dim_list[i],
                                mask_zero=is_mask)(input_word)
    inputs.append(input_word)
    embeddings.append(embedding_word)

embedding_combine = keras.layers.concatenate(inputs=embeddings, axis=1)

# attention
# inputs = embedding_combine
# head_size=256,
# num_heads=4,
# ff_dim=4,
# dropout=0.25,
# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
#     x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = keras.layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(x, x)
#     x = keras.layers.Dropout(dropout)(x)
#     res = x + inputs

#     # Feed Forward 
#     x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
#     x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = keras.layers.Dropout(dropout)(x)
#     x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

# attention_output = keras.layers.Attention()([embedding_combine, embedding_combine, embedding_combine])
# attention_output = keras.layers.GlobalAveragePooling1D()(attention_output)
# attention_output = keras.layers.Dropout(0.25)(attention_output)
# output = keras.layers.Dense(1, activation='sigmoid')(attention_output)

##dnn
# attention_output = keras.layers.Attention()([embedding_combine, embedding_combine, embedding_combine])
# lstm_output = keras.layers.LSTM(units=256, input_shape=embedding_combine.shape[1:])(embedding_combine)
embedding_output = keras.layers.GlobalAveragePooling1D()(embedding_combine)
dnn_layer = keras.layers.Dense(256, activation = 'relu')(embedding_output)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
dnn_layer = keras.layers.Dense(128, activation = 'relu')(dnn_layer)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
dnn_layer = keras.layers.Dense(64, activation = 'relu')(dnn_layer)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
dnn_layer = keras.layers.Dense(32, activation = 'relu')(dnn_layer)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
dnn_layer = keras.layers.Dense(16, activation = 'relu')(dnn_layer)
dnn_layer = keras.layers.BatchNormalization()(dnn_layer)
output = keras.layers.Dense(1)(dnn_layer)

## CNN
# conv_output = keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=embedding_combine.shape[1:])(embedding_combine)
# cnn_layer = keras.layers.BatchNormalization()(conv_output)
# cnn_layer = keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=cnn_layer.shape[1:])(cnn_layer)
# cnn_layer = keras.layers.BatchNormalization()(cnn_layer)
# cnn_output = keras.layers.GlobalAveragePooling1D()(cnn_layer)
# output = keras.layers.Dense(1,activation='sigmoid')(cnn_output)

input_wide_deep = keras.layers.concatenate([output, w])
# input_wide_deep = keras.layers.concatenate(inputs=[output, w], axis=1)
output_wide_deep = keras.layers.Dense(1, activation='sigmoid', name="output_wide_deep", )(input_wide_deep)

model = keras.Model(inputs=inputs+[wide_inp], outputs=[output_wide_deep])
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=[keras.metrics.AUC()])#'adam'
print(model.summary())

train = train_data[train_data['sample_date']<20220316]
val = train_data[train_data['sample_date']==20220316]
test = test_data

deep_features = deep_name_list

x_train_set = {}
x_val_set = {}
x_test_set = {}
for i in range(len(input_dim_list)):
    x_train_set[deep_features[i]]=get_convert_data(train, deep_features[i], dic_list[i])
    x_val_set[deep_features[i]] = get_convert_data(val, deep_features[i], dic_list[i])
    x_test_set[deep_features[i]] = get_convert_data(test, deep_features[i], dic_list[i])

ss = StandardScaler()
ss.fit(train[wide_name_list].fillna(0))

x_train_set['wide_inp'] = ss.transform(train[wide_name_list].fillna(0).values)
x_val_set['wide_inp'] = ss.transform(val[wide_name_list].fillna(0).values)
x_test_set['wide_inp'] = ss.transform(test[wide_name_list].fillna(0).values)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')
check_point_path = "../model/ctr_deep_model.h5"
model_checkpoint = keras.callbacks.ModelCheckpoint(check_point_path, verbose=1, save_best_only=True, save_weights_only=True)
model.fit(x_train_set, train['label'].values, \
            validation_data=(x_val_set, val['label'].values), \
            epochs=2, batch_size=1024, shuffle=True, \
            callbacks=[early_stopping, model_checkpoint])


test_result = model.predict(x_test_set)
test['result'] = test_result
test[['result']].to_csv('./result.csv',index=False)