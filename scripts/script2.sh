#!/bin/bash

for i in ./sdss/cutouts/100*.png; do 
cp $i ./cutouts_1000/cutouts_1000_train
done

for i in ./sdss/cutouts/200*.png; do 
cp $i ./cutouts_1000/cutouts_1000_train
done

for i in ./sdss/cutouts/500*.png; do 
cp $i ./cutouts_1000/cutouts_1000_train
done

for i in ./sdss/cutouts/600*.png; do 
cp $i ./cutouts_1000/cutouts_1000_train
done

for i in ./sdss/cutouts/700*.png; do 
cp $i ./cutouts_1000/cutouts_1000_train
done
