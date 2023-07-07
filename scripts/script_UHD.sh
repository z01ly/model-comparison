#!/bin/bash

mkdir ../UHD
cd ../UHD
mkdir test
cd ..

for i in ./mock_images/faceon_1114/UHD*.png; do 
cp $i ./UHD/test
done

for i in ./mock_images/UHD/*.png; do 
cp $i ./UHD/test
done
