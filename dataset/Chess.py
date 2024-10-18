'''
    数据集被删除
'''
# import pandas as pd
# from sklearn import preprocessing
# from torch.utils.data import Dataset

# def load_chess_dataset(data_dir):
#      #读取数据
#     data_dir = data_dir + '/Chess/krkopt.data' # 28055 0.9训练集
#     xy = pd.read_csv(data_dir)
#     #增加表头
#     xy.columns = ['bk_x','bk_y','wk_x','wk_y','ws_x','ws_y','outcome']
#     #数据格式化
#     xy.replace(to_replace={'^a$': 1, '^b$': 2, '^c$': 3, '^d$': 4, '^e$': 5, '^f$': 6, '^g$': 7, '^h$': 8, '^draw$': 1,
#                 "(?!draw)": 0}, regex=True, inplace=True)
#     #数据归一化
#     xy[['bk_x','bk_y','wk_x','wk_y','ws_x','ws_y']] = preprocessing.scale(
#         xy[['bk_x','bk_y','wk_x','wk_y','ws_x','ws_y']])
#     pd.DataFrame(data=xy).to_csv("krkopt_fill.csv")
    
# class Krkoptdataset():
#     def __init__(self, data_dir):
#         #读取数据
#         data_dir = data_dir + '/Chess/krkopt_fill.csv' # 28055 0.9训练集
#         new_xy = pd.read_csv(data_dir)
 
#         self.x_data = new_xy[['bk_x','bk_y','wk_x','wk_y','ws_x','ws_y']]
#         self.y_data = new_xy[['outcome']]
#         self.len = new_xy.shape[0]
 
#     def __getitem__(self, item):
 
#         return self.x_data[item],self.y_data[item]
 
#     def __len__(self):
 
#         return self.len
 
#     def get_data(self):
 
#         return self.x_data, self.y_data
 

# data_dir = "/data2/wyh-dataset/tabular-dataset"
# krkopt = Krkoptdataset(data_dir=data_dir)
# #X_train, X_test, y_train, y_test = train_test_split(krkopt.getX(),krkopt.getY(), train_size=5000, random_state=0)
# print(krkopt.get_data()[1])
