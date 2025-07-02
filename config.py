
import argparse

## arguments
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', type=str, default='./data/ml-1m/ratings.dat')
parser.add_argument('--min_rating', type=int, default=0, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')

parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500,  help='Size of val and test set. 500 for ML-1m and 10000 for ML-20m recommended')

# Dataloader
parser.add_argument('--dataloader_random_seed', type=float, default=12345)
parser.add_argument('--batch_size', type=int, default=16)

# NegativeSampler
parser.add_argument('--train_negative_sampler_code', type=str, default='random', help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=12345)
parser.add_argument('--test_negative_sampler_code', type=str, default='random',help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=12345)


# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=0)
parser.add_argument('--device_idx', type=str, default='-1')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')

# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=32, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20, 50, 100], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

# Model
parser.add_argument('--model_init_seed', type=int, default=None)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=256, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=0.15, help='Probability for masking items in the training sequence')

# Experiment
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')

################
args = parser.parse_args([])
