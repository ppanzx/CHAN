import logging
import argparse
from lib import evaluation
from lib.modules import set_seeds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',help='coco or f30k')
    parser.add_argument('--result1', default='runs/release_weights/coco_butd_grid_bigru/results_coco.npy')
    parser.add_argument('--result2', default='runs/release_weights/coco_butd_region_bigru/results_coco.npy')
    parser.add_argument('--evaluate_cxc', action='store_true')
    parser.add_argument('--seed', default=2022, type=int, help='random seed')
    opt = parser.parse_args()

    set_seeds(opt.seed)

    if opt.dataset == 'coco':
        if not opt.evaluate_cxc:
            # Evaluate COCO 5-fold 1K
            evaluation.eval_ensemble(results_paths=[opt.result1,opt.result2], fold5=True)
            evaluation.eval_ensemble(results_paths=[opt.result1,opt.result2], fold5=False)
        else:
            # Evaluate COCO-trained models on CxC
            evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
    elif opt.dataset == 'f30k':
        # Evaluate Flickr30K
        evaluation.eval_ensemble(results_paths=[opt.result1,opt.result2], fold5=False)


if __name__ == '__main__':
    main()
