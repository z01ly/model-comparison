#!/bin/bash

mkdir ../n80
cd ../n80
mkdir test
cd ..

for i in ./mock_images/faceon_1114/n80*.png; do 
cp $i ./n80/test
done

for i in ./mock_images/n80_rt_1129/*.png; do 
cp $i ./n80/test
done
