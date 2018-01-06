from __future__ import print_function
import argparse
import os

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
import numpy as np

import models.VGG
import models.C3D
from datasets.MUG import RgbdLabeledMugDataset


def main():
    archs = {
        'vgg3d': models.VGG.VGG3D,
        'c3d': models.C3D.C3D
    }
    optimizers = {
        'sgd': chainer.optimizers.SGD,
        'momentum_sgd': chainer.optimizers.MomentumSGD
    }
    parser = argparse.ArgumentParser(description='Chainer ConvolutionND example:')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--dataset', '-d', default='data',
                         help='dataset path for mug dataset')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='c3d',
                        help='Architecture')
    parser.add_argument('--optimizer', '-z', choices=optimizers.keys(), default='momentum_sgd',
                        help='Optimizer')
    parser.add_argument('--train-data', '-i', default='images',
                        help='Directory of training data')
    parser.add_argument('--test-data', '-t', default='tests',
                        help='Directory of test data')
    parser.add_argument('--mean', '-m', default=None,
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--frames', '-f', type=int, default=6,
                        help='frames for convlution')
    parser.add_argument('--no-random', action='store_true',
                        help='Disable data augmentation')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    print('Using MUG dataset.')
    print('Using {} model.'.format(args.arch))
    print('Disable data augmentation' if args.no_random else 'Enable data augmentation')
    mean = np.load(args.mean) if os.path.isfile(args.mean if args.mean else "") else None

    dataset_path = os.path.abspath(args.dataset)
    X_train_path = os.path.join(dataset_path, "X_train.npy")
    t_train_path = os.path.join(dataset_path, "t_train.npy")
    train_dataset = RgbdLabeledMugDataset(X_train_path, t_train_path)
    X_test_path = os.path.join(dataset_path, "X_test.npy")
    t_test_path = os.path.join(dataset_path, "t_test.npy")
    test_dataset = RgbdLabeledMugDataset(X_test_path, t_test_path)

    model = L.Classifier(archs[args.arch](6))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # optimizer = optimizers[args.optimizer](args.learnrate)
    optimizer = chainer.optimizers.Adam(alpha=1e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_dataset, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 5 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(5, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'))
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}.npz'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    # trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)

if __name__ == '__main__':
    main()
