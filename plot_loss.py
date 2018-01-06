import argparse
import os, sys, re, glob
# import matplotlib
# matplotlib.use("pdf")
import matplotlib.pyplot as plt
plt.style.use("bmh")
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path')
    parser.add_argument('result_path')
    args = parser.parse_args()
        
    # read log
    log_path = os.path.abspath(args.log_path)
    with open(log_path, 'r') as f:
        log_json = json.load(f)
    
    epochs = list(range(len(log_json)))
    train_loss = [d['main/loss'] for d in log_json]
    test_loss = [d['validation/main/loss'] for d in log_json]
    train_acc = [d['main/accuracy'] for d in log_json]
    test_acc = [d['validation/main/accuracy'] for d in log_json]

    # plot setting
    result_path = os.path.abspath(args.result_path)
    os.makedirs(result_path, exist_ok=True)
    plt.rcParams['font.family'] ='sans-serif'#使用するフォント
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 12 #フォントの大きさ
    plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

    # plot loss
    plt.plot(epochs, train_loss, 'r-', label='training loss')
    plt.plot(epochs, test_loss, 'b-', label='test loss')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(result_path, "loss.pdf"), bbox_inches="tight", pad_inches=0.0)

    # plot accuracy
    plt.figure()
    plt.plot(epochs, train_acc, 'r-', label='training acc')
    plt.plot(epochs, test_acc, 'b-', label='test acc')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(os.path.join(result_path, "acc.pdf"), bbox_inches="tight", pad_inches=0.0)

if __name__=="__main__":
    main()
