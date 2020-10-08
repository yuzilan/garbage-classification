import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Directory for input data.', default='./datasets/train_data/')
    parser.add_argument('--train_local', type=str, help='Directory for log data.', default='./logs/')
    parser.add_argument('--num_filters', type=int, help='(ex, --num_filters 32)', default=[32])
    parser.add_argument('--smi_filter_lengths', type=int, help='ex, --smi_filter_lengths 4 6 8', default=[4, 6, 8])
    parser.add_argument('--seq_filter_lengths', type=int, help='ex, --seq_filter_lengths 4 8 12', default=[4, 8, 12])
    parser.add_argument('--max_smi_len', type=int, help='Length of input sequences.', default=100)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs to train.', default=100)

    # params for train
    parser.add_argument('--train_outputs', type=str, help='Path to save training outputs.', default='./datasets/train_outputs/')
    parser.add_argument('--restore_model_path', type=str, help='A history model you have trained, you can load it and continue trainging.', default='')
    parser.add_argument('--batch_size', type=int, help='Must divide evenly into the dataset sizes.', default=8)
    parser.add_argument('--num_classes', type=int, help='Num of classes which your task should classify', default=40)
    parser.add_argument('--input_size', type=int, help='The input image size of the model', default=456)
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.', default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=30)

    # tf.app.flags.DEFINE_string('data_url', '', 'the training data path')
    # # tf.app.flags.DEFINE_string('train_url', '', 'the path to save training outputs')
    # tf.app.flags.DEFINE_integer('keep_weights_file_num', 20,
    #                             'the max num of weights files keeps, if set -1, means infinity')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS
