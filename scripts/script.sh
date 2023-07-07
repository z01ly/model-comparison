#!/bin/bash

mkdir cutouts_1000
mkdir cutouts_1000_2

for i in ./cutouts/300*.png; do 
# echo $i;
cp $i cutouts_1000
done

for i in ./cutouts/400*.png; do 
cp $i cutouts_1000_2
done
