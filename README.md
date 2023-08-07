# h2seqrec

## Paper
[Hyperbolic Hypergraphs for Sequential Recommendation](https://arxiv.org/pdf/2108.08134.pdf)<br>
[Data](https://drive.google.com/drive/folders/18Yfn5pIKOGdbcmBllqLx87nFII151sqa?usp=sharing)

## Environment
Recommendation Environment
~~~
python==3.6.12
pytorch==1.9.0
numpy==1.17.5
scikit-learn==0.24.2
scipy==1.5.4 
~~~
Hyperbolic Hypergraph Neural Network Environment
~~~
pytorch==1.0.0
python==3.6.13
~~~


## Run steps
1. Pre-training phase (three contrastive pre-training tasks)<br>
a. Run pre-training 

    `python H2SeqRec-Pretrain/run_pretrain.py`<br>
b. Copy the output pre-training embedding to `HGCN/data/AMT/use_pretrain/` folder


2. Hierarchical Hyperbolic Hypergraph Neural Network<br>
a. Prepare dataset.<br>
Split the dataset into monthly, quarterly and yearly dataset. 
For example, if the dataset contains 12 months, 
the `HGCN/data/month` folder will have 12 csv files, 
the `HGCN/data/quarter` folder will have 4 csv files 
and the `HGCN/data/year` folder will have 1 csv file. 
The format of each line is as follows. 
`node1_id node2_id`.
The nodes within each hyperedge are considered connected with each other. 
The `HGCN/config.py` file is the configuration file.<br>
b. Modify parameters. <br>
Taking the monthly hyperbolic hypergraph embedding learning as an example, in `HGCN/train.py`
change the `pre-training embedding` name in line 195, and modify the `time_list` in line 200.
Supposing there are 12 months, the `time_list` should be `range(1,13)`.<br>
c. Run hyperbolic hypergraph neural network.<br>
`python HGCN/train.py --dataset AMT`<br>
P.S. Run monthly, quarterly, yearly hyperbolic hypergraph neural network seperately.
The steps are the same, but parameters should be modified.

3. User-side Hyperbolic Hypergraph Neural Network<br>
a. Prepare user-side hypergraph dataset.<br>
`python create_user_side_hypergraph.py`<br>
b. Modify parameters as 2-b.<br>
c. Run hyperbolic hypergraph neural network as 2-c.<br>
P.S. Only monthly is needed.

4. Completed Model<br>
a. Prepare needed data.<br>
`python user_side_hypergraphs_new_hyperbolic_simple.py`<br>
b. Run recommendation.<br>
`python run_total.py`
