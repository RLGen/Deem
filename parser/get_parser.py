import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Detecting Errors in Dataset Labels")
    parser.add_argument("--data_root", type=str, default="tabular-dataset", help="Path to the dataset root folder")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "mnist", "kmnist", "cifar100", "food101", "clothing-1m",
                                                         "adults", "convertype", "drybean", "news", "imdb","hepmass", "svhn"], default="hepmass", help="Dataset to use")
    parser.add_argument("--noise_type", type=str, choices=["sym_noise", "asymmetric_noise"], default="sym_noise", help="the type of noise")
    parser.add_argument('--noise_rate', type = float, default = 0.2, help = 'corruption rate, should be less than 1')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Training data ratio')
    parser.add_argument('--result_dir', type = str, default = 'result', help = 'dir to save result txt files')
    
    # train model
    parser.add_argument("--model", type=str,choices=["cnn"], default="cnn", help="model for training")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Batch size for testing")
    
    # coteaching default argument
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
    parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--top_bn', action='store_true')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--num_iter_per_epoch', type=int, default=400)
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--K', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.2)
    return parser