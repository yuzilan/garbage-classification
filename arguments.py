import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Directory for input data.', default='./datasets/train_data/')
    parser.add_argument('--train_local', type=str, help='Directory for log data.', default='./logs/')

    # params for train
    parser.add_argument('--batch_size', type=int, help='Must divide evenly into the dataset sizes.', default=4)
    parser.add_argument('--num_classes', type=int, help='Num of classes which your task should classify', default=40)
    parser.add_argument('--input_size', type=int, help='The input image size of the model', default=456)
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.', default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--sample_count', type=int, default=14802)  # 样本总数

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS
