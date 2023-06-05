import pandas as pd


def set_display():
    # 显示所有列
    pd.set_option("display.max_columns", None)
    # 显示所有行
    pd.set_option("display.max_rows", None)
    pd.set_option("max_colwidth", 500)


def read_data(file_path: str):
    trunk = pd.read_csv(file_path, sep='\t')

    print("data sample is {}".format(trunk[:2]))
    print("data shape is {}, type is {}".format(trunk.shape, type(trunk)))
    print("用户ID is {}， 资讯 id is {}".format(trunk.iloc[0, 0], trunk.iloc[0, 1]))

    print("主被动刷新 is {}".format(trunk.iloc[0, 2]))
    print("交易时段 is {}".format(trunk.iloc[0, 3]))
    print("资讯召回方式 is {}".format(trunk.iloc[0, 4]))
    print("APP客户端名称 is {}".format(trunk.iloc[0, 5]))

    print("资讯点击率 is {}".format(trunk.iloc[0, 6]))
    print("大盘涨停股票数 is {}".format(trunk.iloc[0, 7]))
    print("大盘跌停股票数 is {}".format(trunk.iloc[0, 8]))
    print("大盘涨跌幅is {}".format(trunk.iloc[0, 9]))

    print("资讯入库距今时长分段 is {}".format(trunk.iloc[0, 10]))
    print("推荐样本日期 is {}".format(trunk.iloc[0, 11]))

    print("用户近30天feed流点击次数 is {}".format(trunk.iloc[0, 12]))

    print("length is {}, 用户偏好资讯特征C列表 is {}".format(len(trunk.iloc[0, 13].split(',')), trunk.iloc[0, 13]))
    print("length is {}, 用户偏好资讯特征D列表 is {}".format(len(trunk.iloc[0, 14].split(',')), trunk.iloc[0, 14]))
    print("length is {}, 用户偏好资讯特征B列表 is {}".format(len(trunk.iloc[0, 15].split(',')), trunk.iloc[0, 15]))
    print("length is {}, 用户偏好资讯特征H列表 is {}".format(len(trunk.iloc[0, 16].split(',')), trunk.iloc[0, 16]))
    print("length is {}, 用户偏好资讯特征A列表 is {}".format(len(trunk.iloc[0, 17].split(',')), trunk.iloc[0, 17]))

    print("length is {}, 用户近三天阅读的第1篇资讯的特征D is {}".format(len(trunk.iloc[0, 18].split(',')), trunk.iloc[0, 18]))
    print("length is {}, 用户近三天阅读的第2篇资讯的特征D is {}".format(len(trunk.iloc[0, 19].split(',')), trunk.iloc[0, 19]))
    print("length is {}, 用户近三天阅读的第3篇资讯的特征D is {}".format(len(trunk.iloc[0, 20].split(',')), trunk.iloc[0, 20]))
    print("length is {}, 用户近三天阅读的第4篇资讯的特征D is {}".format(len(trunk.iloc[0, 21].split(',')), trunk.iloc[0, 21]))
    print("length is {}, 用户近三天阅读的第5篇资讯的特征D is {}".format(len(trunk.iloc[0, 22].split(',')), trunk.iloc[0, 22]))

    print("length is {}, 用户近三天阅读的第1篇资讯的特征C is {}".format(len(trunk.iloc[0, 23].split(',')), trunk.iloc[0, 23]))
    print("length is {}, 用户近三天阅读的第2篇资讯的特征C is {}".format(len(trunk.iloc[0, 24].split(',')), trunk.iloc[0, 24]))
    print("length is {}, 用户近三天阅读的第3篇资讯的特征C is {}".format(len(trunk.iloc[0, 25].split(',')), trunk.iloc[0, 25]))
    print("length is {}, 用户近三天阅读的第4篇资讯的特征C is {}".format(len(trunk.iloc[0, 26].split(',')), trunk.iloc[0, 26]))
    print("length is {}, 用户近三天阅读的第5篇资讯的特征C is {}".format(len(trunk.iloc[0, 27].split(',')), trunk.iloc[0, 27]))

    print("length is {}, 用户近三天阅读资讯mask特征 is {}".format(len(trunk.iloc[0, 28].split(',')), trunk.iloc[0, 28]))
    print("length is {}, 用户近实时点击资讯特征C序列 is {}".format(len(trunk.iloc[0, 29].split(',')), trunk.iloc[0, 29]))

    print("length is {}, 资讯特征A is {}".format(len(trunk.iloc[0, 30].split(',')), trunk.iloc[0, 30]))
    print("资讯特征B is {}".format(trunk.iloc[0, 31]))
    print("length is {}, 资讯特征C is {}".format(len(trunk.iloc[0, 32].split(',')), trunk.iloc[0, 32]))
    print("length is {}, 资讯特征D is {}".format(len(trunk.iloc[0, 33].split(',')), trunk.iloc[0, 33]))
    print("length is {}, 资讯特征E is {}".format(len(trunk.iloc[0, 34].split(',')), trunk.iloc[0, 34]))
    print("length is {}, 资讯特征F is {}".format(len(trunk.iloc[0, 35].split(',')), trunk.iloc[0, 35]))
    print("length is {}, 资讯特征G is {}".format(len(trunk.iloc[0, 36].split(',')), trunk.iloc[0, 36]))
    print("资讯特征H is {}".format(trunk.iloc[0, 37]))

    print("label is {}".format(trunk.iloc[0, 38]))
    return trunk.iloc[:, 0:38], trunk.iloc[:, 38]


if __name__ == '__main__':
    set_display()
    train_file = '/home/ryne/Downloads/data/train/train_data.csv'
    data, label = read_data(train_file)
    print('data type is {}, shape is {}'.format(type(data), data.shape))

    print('label type is {}, shape is {}'.format(type(label), label.shape))
    pass
