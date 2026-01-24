# for dataset in AF CR FM RS SRS2 SWJ UWG
for dataset in FM
do
  python preprocess.py --gpu 7 --dataset $dataset --multi_var --flag train
  python preprocess.py --gpu 7 --dataset $dataset --multi_var --flag val
  python preprocess.py --gpu 7 --dataset $dataset --multi_var --flag test
done
