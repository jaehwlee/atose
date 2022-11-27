#!/bin/bash

# HarmonicCNN
python main.py --dataset "keyword" --isTest False --encoder_type "HC"  --latent 64 --withJE True --gpu "0" 
python main.py --dataset "keyword" --isTest False --encoder_type "HC"  --latent 64 --withJE False --gpu "0" 

python main.py --dataset "mtat" --isTest False --encoder_type "HC"  --latent 128 --withJE False --gpu "0" 
python main.py --dataset "mtat" --isTest False --encoder_type "HC"  --latent 128 --withJE True --gpu "0" 

python main.py --dataset "dcase" --isTest False --encoder_type "HC"  --latent 50 --withJE False --gpu "0"
python main.py --dataset "dcase" --isTest False --encoder_type "HC"  --latent 50 --withJE True --gpu "0" 

# TagSincNet
python main.py --dataset "keyword" --isTest False --encoder_type "MS" --latent 64 --withJE True --gpu "0" 
python main.py --dataset "keyword" --isTest False --encoder_type "MS" --latent 64 --withJE False --gpu "0" 

python main.py --dataset "mtat" --isTest False --encoder_type "MS" --latent 128 --withJE False --gpu "0" 
python main.py --dataset "mtat" --isTest False --encoder_type "MS" --latent 128 --withJE True --gpu "0"
python main.py --dataset "dcase" --isTest False --encoder_type "MS" --latent 50 --withJE False --gpu "0" 
python main.py --dataset "dcase" --isTest False --encoder_type "MS" --latent 50 --withJE True --gpu "0" 

# SampleCNN (basic)
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "basic" --latent 64 --withJE True --gpu "0" 
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "basic" --latent 64 --withJE False --gpu "0"  

python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "basic" --latent 128 --withJE False --gpu "0" 
python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "basic" --latent 128 --withJE True --gpu "0"

python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "basic" --latent 50 --withJE False --gpu "0" 
python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "basic" --latent 50 --withJE True --gpu "0" 

# SampleCNN (+se)
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "se" --latent 64 --withJE True --gpu "0" 
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "se" --latent 64 --withJE False --gpu "0"  

python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "se" --latent 128 --withJE False --gpu "0" 
python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "se" --latent 128 --withJE True --gpu "0"

python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "se" --latent 50 --withJE False --gpu "0" 
python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "se" --latent 50 --withJE True --gpu "0" 

# SampleCNN (+res)
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "res" --latent 64 --withJE True --gpu "0" 
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "res" --latent 64 --withJE False --gpu "0"  

python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "res" --latent 128 --withJE False --gpu "0" 
python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "res" --latent 128 --withJE True --gpu "0"

python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "res" --latent 50 --withJE False --gpu "0" 
python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "res" --latent 50 --withJE True --gpu "0" 

# SampleCNN (+rese)
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "rese" --latent 64 --withJE True --gpu "0" 
python main.py --dataset "keyword" --isTest False --encoder_type "SC" --block "rese" --latent 64 --withJE False --gpu "0"  

python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "rese" --latent 128 --withJE False --gpu "0" 
python main.py --dataset "mtat" --isTest False --encoder_type "SC" --block "rese" --latent 128 --withJE True --gpu "0"

python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "rese" --latent 50 --withJE False --gpu "0" 
python main.py --dataset "dcase" --isTest False --encoder_type "SC" --block "rese" --latent 50 --withJE True --gpu "0" 