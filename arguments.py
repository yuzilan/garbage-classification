import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Directory for input data.', default='./datasets/train_data/')
    parser.add_argument('--train_local', type=str, help='Directory for log data.', default='./logs/')
    # parser.add_argument('--num_filters', type=int, help='(ex, --num_filters 32)', default=[32])
    # parser.add_argument('--smi_filter_lengths', type=int, help='ex, --smi_filter_lengths 4 6 8', default=[4, 6, 8])
    # parser.add_argument('--seq_filter_lengths', type=int, help='ex, --seq_filter_lengths 4 8 12', default=[4, 8, 12])
    # parser.add_argument('--max_smi_len', type=int, help='Length of input sequences.', default=100)
    # parser.add_argument('--num_epoch', type=int, help='Number of epochs to train.', default=100)

    # params for train
    parser.add_argument('--train_outputs', type=str, help='Path to save training outputs.', default='./train_outputs/')
    parser.add_argument('--restore_model_path', type=str, help='A history model you have trained, you can load it and continue trainging.', default='')
    parser.add_argument('--batch_size', type=int, help='Must divide evenly into the dataset sizes.', default=4)
    parser.add_argument('--num_classes', type=int, help='Num of classes which your task should classify', default=40)
    parser.add_argument('--input_size', type=int, help='The input image size of the model', default=456)
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.', default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=6)

    # params for save pb
    parser.add_argument('--test_data_url', type=str, help='the test data path on obs', default='./datasets/test_data/')
    parser.add_argument('--keep_weights_file_num', type=int, help='the max num of weights files keeps, if set -1, means infinity', default=20)
    
#     tf.app.flags.DEFINE_string('test_data_url', '', 'the test data path on obs')
#     tf.app.flags.DEFINE_string('deploy_script_path', '',
#                            'a path which contain config.json and customize_service.py, '
#                            'if it is set, these two scripts will be copied to {train_url}/model directory')
# tf.app.flags.DEFINE_string('freeze_weights_file_path', '',
#                            'if it is set, the specified h5 weights file will be converted as a pb model, '
#                            'only valid when {mode}=save_pb')
    
#     tf.app.flags.DEFINE_integer('keep_weights_file_num', 20,
#                             'the max num of weights files keeps, if set -1, means infinity')
    # tf.app.flags.DEFINE_string('data_url', '', 'the training data path')
    # # tf.app.flags.DEFINE_string('train_url', '', 'the path to save training outputs')
    # tf.app.flags.DEFINE_integer('keep_weights_file_num', 20,
    #                             'the max num of weights files keeps, if set -1, means infinity')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS
