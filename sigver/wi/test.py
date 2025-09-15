import torch
from sigver.featurelearning.data import extract_features
import sigver.featurelearning.models as models
import argparse
from sigver.datasets.util import load_dataset, get_subset, load_extracted_features
import sigver.wi.training as training
import numpy as np
import pickle


def main(args):
    
    sk_for_test=args.sk_for_test if args.sk_for_test != -1 else args.gen_for_test


    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                                  map_location=lambda storage, loc: storage, weights_only=False)
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    base_model = models.available_models['signet']().to(device).eval()

    base_model.load_state_dict(state_dict)

    
    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)

    if args.input_type == "image":
        x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)
        features = extract_features(x, process_fn, args.batch_size, args.input_size)
    else:
        features, y, yforg = load_extracted_features(args.data_path)

    data = (features, y, yforg)
    
    

    

    rng = np.random.RandomState(1234)

    eer_u_list = []
    eer_list = []
    all_results = []
    for fold in range(args.folds):
        
        if args.random_users is None:
            if fold == 0 :
                exp_users = range(*args.exp_users)
                dev_users = range(*args.dev_users)
                
                assert len(
                    set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'
                
                exp_set = get_subset(data, exp_users)
                dev_set = get_subset(data, dev_users)
            
        else:
            print("Random spliting users..")
            users = np.array((range(*args.random_users))) 
            exp_users = rng.choice(users, len(users)//2, replace=False)
            dev_users = users[~np.isin(users,exp_users)]
            
            exp_set = get_subset(data, exp_users)
            dev_set = get_subset(data, dev_users)
        
        
        classifiers, results = training.train_test_all_users(exp_set,
                                                             dev_set,
                                                             svm_type=args.svm_type,
                                                             C=args.svm_c,
                                                             gamma=args.svm_gamma,
                                                             num_gen_train=args.gen_for_train,
                                                             num_gen_ref=args.gen_for_ref,
                                                             num_gen_test=args.gen_for_test,
                                                             num_sk_test=sk_for_test,
                                                             fusion=args.fusion,
                                                             global_threshold=args.thr,
                                                             rng=rng)
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)

    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))   

    if args.save_path is not None:
        print('Saving results to {}'.format(args.save_path))
        with open(args.save_path, 'wb') as f:
            pickle.dump(all_results, f)
    return all_results

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--save-path')
    parser.add_argument('--input-size', nargs=2, default=(150, 220))

    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300))
    parser.add_argument('--dev-users', type=int, nargs=2, default=(5000, 7000))
    parser.add_argument('--random-users', nargs=2, type=int,
                        help="Specify a user interval: the exp and dev sets will be created by randomly assigning half the users to each set."
        )

    parser.add_argument('--gen-for-train', type=int, default=12)
    parser.add_argument('--gen-for-test', type=int, default=10)
    parser.add_argument('--gen-for-ref', type=int, default=12)
    parser.add_argument('--sk-for-test', type=int, default=-1, 
        help="Number of skilled forgeries for testing. If set to -1 (default), uses the same value as '--gen-for-test'")

    parser.add_argument('--svm-type', choices=['rbf', 'linear', 'sgd'], default='rbf')
    parser.add_argument('--svm-c', type=float, default=1)
    parser.add_argument('--svm-gamma', type=float, default=2**-11)
    parser.add_argument('--fusion', help='Fusion type', choices=('max','min','mean','median'), default='max', type=str)
    parser.add_argument('--thr', type=float, default=0)

    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=10)
    
    parser.add_argument('--input-type', type=str, default="image", choices=["features", "image"])

    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_args()
    print(arguments)

    main(arguments)
