#!/home/shelby/.conda/envs/tensorflowgpu/bin/python

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

import numpy as np
from GO import GO
from PPI import PPI
from PathManager import PathManager as PM
from ParamHandler import ParamHandler as PH
from KNN import KNN
from Metrics import Metrics

ph = PH(file=args.params)
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
    ae.pca_summary(c[:,0], 'latent')
    d = (c[:,0]-c[:,0].mean(axis=0))/np.std(c[:,0], axis=0)
    knn = KNN(pm, ph, go, d)
elif do_pca:
    ppi.pca()
    c = ppi.data
    d = (c-c.mean(axis=0))/np.std(c, axis=0)
    knn = KNN(pm, ph, go, d)
a = knn.predict()
met = Metrics(pm, ph, a, go)
met.run(show=args.show)
ph.save()