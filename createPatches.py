'''
@Richard Nam 2016/08/03
This script will create patchs for input into the NN
'''

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from PIL import Image
from subprocess import call
import argparse, os, tarfile


# -- convert full size image to npy
def ToRGB(test):
    im = Image.open(test)
    return np.array(im).transpose(2,0,1)

# -- base on this: http://stackoverflow.com/questions/7096323/python-pil-best-scaling-method-that-preserves-lines
def ReshapeImage(test):
    im = Image.fromarray(test)
    im = im.convert('RGB')
    im_c = im.resize((24,32), 1)
    return np.array(im_c).transpose(2,0,1)

def main(patchpickle, minheight, outdir, datadir):
    call(['tar', '-zxvf', '%spatch2.tar.gz' % datadir])
    call(['mv', 'patch2.pkl', datadir])
    # -- read in pickle file with raw patch values
    df = pd.read_pickle('%spatch2.pkl' % datadir)
    df.columns = ['tag','source_image','path']
    # -- create a flag for patchs under min
    df['median_size_flag'] = [1 if i.shape[0] > minheight else 0 for i in df.source_image.values]
    # -- create a numerical flag for positive/negative
    df['glabel'] = np.where(df.tag=='pos',1,0)
    # -- transform image
    df['transform_image'] = [ReshapeImage(i) for i in df.source_image.values]
    # -- subset for where median_size_flag == 1
    df2 = df[(df.median_size_flag==1) & (df.glabel!=-9)]
    # -- split in train/validation, save to disk
    X_train, X_val, y_train, y_val = train_test_split(np.array([i.tolist() for i in df2.transform_image.values]), 
                                                      df2.glabel.values, test_size=.25, 
                                                      random_state=83, stratify=df2.glabel)
    # -- make new dirs if needed, than save
    if os.path.exists(outdir):
        pass
    else:
        os.makedirs(outdir)
    if outdir[-1] != '/':
        outdir+='/'
    np.save('%sX_train.npy' % outdir, X_train)
    np.save('%sX_val.npy' % outdir, X_val)
    np.save('%sy_train.npy' % outdir, y_train)
    np.save('%sy_val.npy' % outdir, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pickle', dest='pickle', default='patch2.pkl', help='pickle file with raw patch data')
    parser.add_argument('-minheight', dest='minheight', type=int, default=8, help='min height for patch truncation')
    parser.add_argument('-outdir', dest='outdir', default='tensors/', help='directory to save a output tensors')
    parser.add_argument('-datadir', dest='datadir', default='data/', help='directory with data tar.gz file')
    args = parser.parse_args()

    print 'start'
    main(args.pickle, args.minheight, args.outdir, args.datadir)
    print 'saving patch tensors to: %s' % args.datadir
    print 'end'

