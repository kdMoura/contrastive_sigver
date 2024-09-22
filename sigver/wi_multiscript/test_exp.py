import torch
from sigver.featurelearning.data import extract_features
import sigver.featurelearning.models as models
import argparse
from sigver.datasets.util import load_dataset, get_subset
import sigver.wi_multiscript.training as training
import sigver.wi_multiscript.data as data
import numpy as np
import pickle
import os

import sys

def train_test(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    assert len(
        set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                                  map_location=lambda storage, loc: storage)
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    base_model = models.available_models['signet']().to(device).eval()

    base_model.load_state_dict(state_dict)

    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)

    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    features = extract_features(x, process_fn, args.batch_size, args.input_size)

    data = (features, y, yforg)

    exp_set = get_subset(data, exp_users)
    dev_set = get_subset(data, dev_users)


    exclude = [4,6,7,8,9,11]
    k_range = [i for i in range(args.spu_range[0],args.spu_range[1]) if i not in exclude] 
    for spu in k_range: #samples per user (spu)
        rng = np.random.RandomState(1234)

        eer_u_list = []
        eer_list = []
        all_results = []
        
        eer_u_list_rf = []
        eer_list_rf = []
        for fold in range(args.folds):
            
        
            #print('Loading svm from')
            #with open(filename, 'rb') as f:
                #modelo = pickle.load(f)
            
            
            
            classifiers, results = training.train_test_all_users(exp_set,
                                                                 dev_set,
                                                                 svm_type=args.svm_type,
                                                                 C=args.svm_c,
                                                                 gamma=args.svm_gamma,
                                                                 num_gen_train=args.gen_for_train,
                                                                 num_gen_ref=spu,
                                                                 num_gen_test=args.gen_for_test,
                                                                 fusion=args.fusion,
                                                                 global_threshold=args.thr,
                                                                 rng=rng)

            filename = "{}_{}_{}_k{}_svm.sav".format(args.extractor,args.ds,str(args.gen_for_train),fold)
            #filename = "{}_{}_{}_svm.sav".format(args.extractor,args.ds,str(args.gen_for_train))

            print('Saving results to {}'.format(args.save_path))
            with open(os.path.join(args.save_path,filename), 'wb') as f:
                pickle.dump(classifiers, f)
                
            this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
            all_results.append(results)
            eer_u_list.append(this_eer_u)
            eer_list.append(this_eer)
            
            this_eer_u_rf, this_eer_rf = results['all_metrics']['EER_userthresholds_rf'], results['all_metrics']['EER_rf']
            eer_u_list_rf.append(this_eer_u_rf)
            eer_list_rf.append(this_eer_rf)
    
        print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
        print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))   
    
    
        print('EER RF (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list_rf) * 100, np.std(eer_list_rf) * 100))
        print('EER RF (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list_rf) * 100, np.std(eer_u_list_rf) * 100))
        
        k_eer_sf_gt = np.mean(eer_list) * 100
        k_eer_sf_gt_sd = np.std(eer_list) * 100
        k_eer_sf_ut = np.mean(eer_u_list) * 100
        k_eer_sf_ut_sd = np.std(eer_u_list) * 100 
        
        k_eer_rf_gt = np.mean(eer_list_rf) * 100
        k_eer_rf_gt_sd = np.std(eer_list_rf) * 100
        k_eer_rf_ut = np.mean(eer_u_list_rf) * 100
        k_eer_rf_ut_sd = np.std(eer_u_list_rf) * 100
        
        
        
        file = open(args.save_path+".txt", 'a')
        file.write("{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\n".format(
            args.wdwi, 
            args.ds,
            args.extractor,
            spu,
            k_eer_sf_gt,
            k_eer_sf_gt_sd,
            k_eer_sf_ut,
            k_eer_sf_ut_sd,
            k_eer_rf_gt,
            k_eer_rf_gt_sd,
            k_eer_rf_ut,
            k_eer_rf_ut_sd,
            args.ds_train,
            str(vars(args)
            )))
        file.close()


    #if args.save_path is not None:
    #    print('Saving results to {}'.format(args.save_path))
    #    with open(args.save_path, 'wb') as f:
    #        pickle.dump(all_results, f)
            
            
    return all_results

def test_modelo(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    assert len(
        set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                                  map_location=lambda storage, loc: storage)
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    base_model = models.available_models['signet']().to(device).eval()

    base_model.load_state_dict(state_dict)

    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)

    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    features = extract_features(x, process_fn, args.batch_size, args.input_size)

    data = (features, y, yforg)

    exp_set = get_subset(data, exp_users)
    dev_set = get_subset(data, dev_users)


    exclude = [4,6,7,8,9,11]
    k_range = [i for i in range(args.spu_range[0],args.spu_range[1]) if i not in exclude] 
    for spu in k_range: #samples per user (spu)
        rng = np.random.RandomState(1234)

        eer_u_list = []
        eer_list = []
        all_results = []
        
        eer_u_list_rf = []
        eer_list_rf = []
        for fold in range(args.folds):
            
            filename = "{}_{}_k{}_svm.sav".format(args.extractor,args.ds_train,fold)
            print('Loading svm from')
            with open(os.path.join(args.svm_model_path,filename), 'rb') as f:
                modelo = pickle.load(f)
            
            
            
            classifiers, results = training.modelo_test(exp_set,
                                                                 dev_set,
                                                                 svm_type=args.svm_type,
                                                                 C=args.svm_c,
                                                                 gamma=args.svm_gamma,
                                                                 num_gen_train=args.gen_for_train,
                                                                 num_gen_ref=spu,
                                                                 num_gen_test=args.gen_for_test,
                                                                 fusion=args.fusion,
                                                                 global_threshold=args.thr,
                                                                 rng=rng,modelo=modelo)

            #filename = "{}_{}_{}_k{}_svm.sav".format(args.extractor,args.ds,str(args.gen_for_train),fold)
            #filename = "{}_{}_{}_svm.sav".format(args.extractor,args.ds,str(args.gen_for_train))

            #print('Saving results to {}'.format(args.save_path))
            #with open(os.path.join(args.save_path,filename), 'wb') as f:
                #pickle.dump(classifiers, f)
                
            this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
            all_results.append(results)
            eer_u_list.append(this_eer_u)
            eer_list.append(this_eer)
            
            this_eer_u_rf, this_eer_rf = results['all_metrics']['EER_userthresholds_rf'], results['all_metrics']['EER_rf']
            eer_u_list_rf.append(this_eer_u_rf)
            eer_list_rf.append(this_eer_rf)
    
        print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
        print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))   
    
    
        print('EER RF (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list_rf) * 100, np.std(eer_list_rf) * 100))
        print('EER RF (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list_rf) * 100, np.std(eer_u_list_rf) * 100))
        
        k_eer_sf_gt = np.mean(eer_list) * 100
        k_eer_sf_gt_sd = np.std(eer_list) * 100
        k_eer_sf_ut = np.mean(eer_u_list) * 100
        k_eer_sf_ut_sd = np.std(eer_u_list) * 100 
        
        k_eer_rf_gt = np.mean(eer_list_rf) * 100
        k_eer_rf_gt_sd = np.std(eer_list_rf) * 100
        k_eer_rf_ut = np.mean(eer_u_list_rf) * 100
        k_eer_rf_ut_sd = np.std(eer_u_list_rf) * 100
        
        
        
        file = open(args.save_path+".txt", 'a')
        file.write("{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\n".format(
            args.wdwi, 
            args.ds,
            args.extractor,
            spu,
            k_eer_sf_gt,
            k_eer_sf_gt_sd,
            k_eer_sf_ut,
            k_eer_sf_ut_sd,
            k_eer_rf_gt,
            k_eer_rf_gt_sd,
            k_eer_rf_ut,
            k_eer_rf_ut_sd,
            args.ds_train,
            str(vars(args)
            )))
        file.close()


    #if args.save_path is not None:
    #    print('Saving results to {}'.format(args.save_path))
    #    with open(args.save_path, 'wb') as f:
    #        pickle.dump(all_results, f)
            
            
    return all_results


def test_modelo_150(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    assert len(
        set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                                  map_location=lambda storage, loc: storage)
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    base_model = models.available_models['signet']().to(device).eval()

    base_model.load_state_dict(state_dict)

    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)

    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    features = extract_features(x, process_fn, args.batch_size, args.input_size)

    data = (features, y, yforg)

    exp_set = get_subset(data, exp_users)
    dev_set = get_subset(data, dev_users)


    exclude = [4,6,7,8,9,11]
    k_range = [i for i in range(args.spu_range[0],args.spu_range[1]) if i not in exclude] 
    for spu in k_range: #samples per user (spu)
        rng = np.random.RandomState(1234)

        eer_u_list = []
        eer_list = []
        all_results = []
        
        eer_u_list_rf = []
        eer_list_rf = []
        for fold in range(args.folds):
            
            filename = "{}_{}_k{}_svm.sav".format(args.extractor,args.ds_train,fold)
            print('Loading svm from')
            with open(os.path.join(args.svm_model_path,filename), 'rb') as f:
                modelo = pickle.load(f)
            
            
            
            classifiers, results = training.modelo_test(exp_set,
                                                                 dev_set,
                                                                 svm_type=args.svm_type,
                                                                 C=args.svm_c,
                                                                 gamma=args.svm_gamma,
                                                                 num_gen_train=args.gen_for_train,
                                                                 num_gen_ref=spu,
                                                                 num_gen_test=args.gen_for_test,
                                                                 fusion=args.fusion,
                                                                 global_threshold=args.thr,
                                                                 rng=rng,modelo=modelo)

            #filename = "{}_{}_{}_k{}_svm.sav".format(args.extractor,args.ds,str(args.gen_for_train),fold)
            #filename = "{}_{}_{}_svm.sav".format(args.extractor,args.ds,str(args.gen_for_train))

            #print('Saving results to {}'.format(args.save_path))
            #with open(os.path.join(args.save_path,filename), 'wb') as f:
                #pickle.dump(classifiers, f)
                
            this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
            all_results.append(results)
            eer_u_list.append(this_eer_u)
            eer_list.append(this_eer)
            
            this_eer_u_rf, this_eer_rf = results['all_metrics']['EER_userthresholds_rf'], results['all_metrics']['EER_rf']
            eer_u_list_rf.append(this_eer_u_rf)
            eer_list_rf.append(this_eer_rf)
    
        print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
        print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))   
    
    
        print('EER RF (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list_rf) * 100, np.std(eer_list_rf) * 100))
        print('EER RF (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list_rf) * 100, np.std(eer_u_list_rf) * 100))
        
        k_eer_sf_gt = np.mean(eer_list) * 100
        k_eer_sf_gt_sd = np.std(eer_list) * 100
        k_eer_sf_ut = np.mean(eer_u_list) * 100
        k_eer_sf_ut_sd = np.std(eer_u_list) * 100 
        
        k_eer_rf_gt = np.mean(eer_list_rf) * 100
        k_eer_rf_gt_sd = np.std(eer_list_rf) * 100
        k_eer_rf_ut = np.mean(eer_u_list_rf) * 100
        k_eer_rf_ut_sd = np.std(eer_u_list_rf) * 100
        
        
        
        file = open(args.save_path+".txt", 'a')
        file.write("{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\n".format(
            args.wdwi, 
            args.ds,
            args.extractor,
            spu,
            k_eer_sf_gt,
            k_eer_sf_gt_sd,
            k_eer_sf_ut,
            k_eer_sf_ut_sd,
            k_eer_rf_gt,
            k_eer_rf_gt_sd,
            k_eer_rf_ut,
            k_eer_rf_ut_sd,
            args.ds_train,
            str(vars(args)
            )))
        file.close()
        
        
        k_eer_sf_gt = np.mean(eer_list[0:150]) * 100
        k_eer_sf_gt_sd = np.std(eer_list[0:150]) * 100
        k_eer_sf_ut = np.mean(eer_u_list[0:150]) * 100
        k_eer_sf_ut_sd = np.std(eer_u_list[0:150]) * 100 
        
        k_eer_rf_gt = np.mean(eer_list_rf[0:150]) * 100
        k_eer_rf_gt_sd = np.std(eer_list_rf[0:150]) * 100
        k_eer_rf_ut = np.mean(eer_u_list_rf[0:150]) * 100
        k_eer_rf_ut_sd = np.std(eer_u_list_rf[0:150]) * 100
        
        file = open(args.save_path+".txt", 'a')
        file.write("{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\n".format(
            args.wdwi, 
            args.ds+"150",
            args.extractor,
            spu,
            k_eer_sf_gt,
            k_eer_sf_gt_sd,
            k_eer_sf_ut,
            k_eer_sf_ut_sd,
            k_eer_rf_gt,
            k_eer_rf_gt_sd,
            k_eer_rf_ut,
            k_eer_rf_ut_sd,
            args.ds_train,
            str(vars(args)
            )))
        file.close()


    #if args.save_path is not None:
    #    print('Saving results to {}'.format(args.save_path))
    #    with open(args.save_path, 'wb') as f:
    #        pickle.dump(all_results, f)
            
            
    return all_results


def main(args):
    if args.exp_type == 'test_save':
        return test_modelo(args)
    if args.exp_type == 'test_save150':
        return test_modelo_150(args)
    return train_test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--save-path')
    parser.add_argument('--input-size', nargs=2, default=(150, 220))

    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300))
    parser.add_argument('--dev-users', type=int, nargs=2, default=(5000, 7000))

    parser.add_argument('--gen-for-train', type=int, default=12)
    parser.add_argument('--gen-for-test', type=int, default=10)
    parser.add_argument('--gen-for-ref', type=int, default=12)

    parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--svm-c', type=float, default=1)
    parser.add_argument('--svm-gamma', type=float, default=2**-11)
    parser.add_argument('--fusion', help='Fusion type', choices=('max','min','mean','median'), default='max', type=str)
    parser.add_argument('--thr', type=float, default=0)

    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=10)
    
    parser.add_argument('--extractor',required=True)
    parser.add_argument('--wdwi',required=True)
    parser.add_argument('--ds',required=True)
    parser.add_argument('--spu-range', nargs=2, type=int, default=(1, 13))
    parser.add_argument('--svm-model-path')
    parser.add_argument('--ds-train')
    
    parser.add_argument('--exp-type', help='Exp type', choices=('train_test','train_save',"test_save","test_save150"), default='train_test', type=str)

    arguments = parser.parse_args()
    print(arguments) 
  
    main(arguments)
