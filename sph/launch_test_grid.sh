for i in 1 2 3 4 5 6 7 8
do
for j in .04 .05 .06 .07 .08 .09 .1
do
ompsub -n $i ./sph.x -s $j
done 
done
