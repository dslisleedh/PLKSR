max=10
for i in `seq 1 $max`
do
    echo "Run $i"
    CUDA_VISIBLE_DEVICES=0 python $1
done
