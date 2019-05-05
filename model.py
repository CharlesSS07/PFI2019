#!/home/shelby/.conda/envs/tensorflowgpu/bin/python

import numpy as np
from GO import GO
from PPI import PPI
from PathManager import PathManager as PM
from ParamHandler import ParamHandler as PH
from KNN import KNN
from Metrics import *

def run(paramsfile='None', show=False):
    
    ph = PH(file=paramsfile)
    ph.load()

    pm = PM(ph)
    go = GO(pm, ph)
    ppi = PPI(pm, ph)

    do_pca = ph.get('PCA', True)

    if ph.get('AE', False):
        if do_pca:
            ppi.pca()
        from AE import AE
        print(ppi.batch_length)
        ae = AE(pm, ph, ppi)
        ae.train()
        ae.save()
        c = ae.compress()
        ae.pca_summary(c, 'latent')
        knn = KNN(pm, ph, go, c)
    elif do_pca:
        ppi.pca()
        c = ppi.data
        knn = KNN(pm, ph, go, c)

    Y = knn.predict()
    met = MetricGrapher(pm, ph, Y, go)
    met.new_metric(Metric.Function('posP'), Metric.Function('posR'), label='PR on positives.')
    met.new_metric(Metric.Function('negP'), Metric.Function('negR'), label='PR on negatives.')
    #met.new_metric(Metric.Function('tpr'), Metric.Function('fpr'), label='ROC')
    met.axies = ['Recall', 'Precision']
    met.initialize_metrics()
    met.make_graphs(show=show)
    ph.save()

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run any part or all of my model.')

    parser.add_argument(
        '-p', '--params',
        default='None', 
        type=str,
        help='What params file to use.'
    )

    parser.add_argument(
        '-s', '--show',
        action='store_const',
        const=True,
        default=False, 
        help='Whether or not to show graphs.'
    )

    args = parser.parse_args()
    print(args)

    run(args.params, args.show)
