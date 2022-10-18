import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--optim', default='adam', type=str,
                        help='the optimizer')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|selfattention|transformer')
    parser.add_argument('--text_enc_type', default="bigru",
                        help='bigru|bert')
    parser.add_argument('--wemb_type', default=None, type=str,
                        help='word embeding type')  
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=0,
                        help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--reset_start_epoch', action='store_true',
                        help='Whether restart the start epoch when load weights')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=2,
                        help='The number of epochs for warming up the embedding layers')
    parser.add_argument('--input_scale_factor', type=float, default=1,
                        help='The factor for scaling the input image')
    parser.add_argument('--drop', action='store_true',
                        help='Whether Abandoning features')
    parser.add_argument('--obj_drop_rate', type=float, default=0.2,
                        help='probability of droping objects.')
    parser.add_argument('--criterion', default="ContrastiveLoss", type=str,
                        help='ContrastiveLoss|InfoNCELoss.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--temperature', default=0.01, type=float,
                        help='Temperature.')
    parser.add_argument('--coding_type', default=None, type=str,
                        help='word embeding type')  
    parser.add_argument('--pooling_type', default=None, type=str,
                        help='word embeding type')  
    parser.add_argument('--alpha', default=None, type=float,
                        help='inversed temperature of softmax in coding.')
    parser.add_argument('--belta', type=float, default=0.1,
                        help='inversed temperature of softmax in pooling.')
    parser.add_argument('--seed', default=2022, type=int,
                        help='random seed')
    return parser
