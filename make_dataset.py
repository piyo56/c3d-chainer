import argparse
import os, glob, re
import numpy as np
import random

user_num_regex = re.compile(r'.*?\/user(?P<user_num>[0-9]+)')
def extract_user_num(name):
    match = re.search(user_num_regex, name)
    user_num = int(match.group('user_num'))

    return user_num

def train_test_split(X_data, t_data, train_ratio=0.7):
    X_train, t_train = [], []
    X_test, t_test = [], []

    users_list = list(set([extract_user_num(path) for path in X_data]))
    num_users = len(users_list)
    random.shuffle(users_list)

    users_train = users_list[:int(num_users * train_ratio)]

    for x, y in zip(X_data, t_data):
        user_num = extract_user_num(x)
        if user_num in users_train:
            X_train.append(x)
            t_train.append(y)
        else:
            X_test.append(x)
            t_test.append(y)
    
    assert len(t_train)+len(t_test) == len(t_data)

    return X_train, X_test, t_train, t_test 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='root path of dataset')
    args = parser.parse_args()

    root_path= os.path.abspath(args.dataset)

    video_categories = glob.glob(os.path.join(root_path, "*"))
    X_data = []
    t_data = []
    for category in video_categories:
        categ = int(os.path.basename(category))
        for video_path in glob.glob(os.path.join(category, "*")):
            X_data.append(video_path)
            t_data.append(categ)
    
    # print(len(X_data), len(t_data))
    
    # split train, test
    X_train, X_test, t_train, t_test = train_test_split(X_data, t_data)
    
    # save them
    pwd = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(pwd, "data")
    np.save(os.path.join(save_path, "X_train"), X_train)
    np.save(os.path.join(save_path, "t_train"), t_train)
    np.save(os.path.join(save_path, "X_test"), X_test)
    np.save(os.path.join(save_path, "t_test"), t_test)

if __name__=="__main__":
    main()
