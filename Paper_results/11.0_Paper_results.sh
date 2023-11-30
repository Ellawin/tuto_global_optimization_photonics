#!/bin/bash
#SBATCH --job-name=PYM
#SBATCH --output=pymooshnew_%A_%a.out
#SBATCH --error=pymooshnew_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=69
#SBATCH -a 0-7

idx=0
#mv logs*.log poubelle/
tag=$RANDOM
for grad in True
do
for Factor in 0.0   #-1.0   #0.0 0.01 0.1 0.2 1.0 2.0 5.0
do
for Full in False
do
for NoverD in 400
do
for Para in False 
do
for Init in -7.0  #1.25 1.5 1.0  #0.0 1.0 0.5 0.25 0.75 #1.25 1.5 -7. 0. 1.  #1. 2. 3. 4. 5. 6.
do
##for pb in bragg bigbragg minibragg  #ellipsometry 
#for pb in ellipsometry
#for pb in photovoltaics hugephotovoltaics bigphotovoltaics #ellipsometry 
#for pb in photovoltaics hugephotovoltaics bragg bigphotovoltaics minibragg  #ellipsometry 
for pb in hugephotovoltaics bragg photovoltaics bigphotovoltaics minibragg  #ellipsometry 
do

echo playing $idx $SLURMARRAY_TASK_ID
if [ "$SLURM_ARRAY_TASK_ID" = "$idx" ]; then
file=pan${pb}_${tag}_${grad}_Factor${Factor}_${Init}_${Full}_${Para}_${NoverD}.py
cp pymoosh_and_nevergrad.py $file
echo Working on $pb 
sed -i "s/^obj_name = .*/obj_name = \"$pb\"/g" $file
sed -i "s/^Grad = .*/Grad = ${grad}/g" $file
sed -i "s/^Factor = .*/Factor = ${Factor}/g" $file
sed -i "s/^Init = .*/Init = ${Init}/g" $file
sed -i "s/^Full = .*/Full = ${Full}/g" $file
sed -i "s/^Para = .*/Para = ${Para}/g" $file
sed -i "s/^NoverD = .*/NoverD = ${NoverD}/g" $file
#sbatch --partition=scavenge --time=72:00:00  --job-name=PYM --gpus-per-task=0 --cpus-per-task=26 --wrap="
python -W ignore -u $file 2>&1 | tee newlogs_${file}_`date | sed 's/ /_/g'`.log 
#&
#sleep 7
else
echo skip
fi
idx=$(( idx + 1  ))
done
done
done
done
done
done
done

#wait


