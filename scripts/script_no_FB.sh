#!/bin/bash

mkdir ../no_FB
cd ../no_FB
mkdir test
cd ..

for i in ./mock_images/pynbody_rt_0927/no_FB*.png; do 
cp $i ./no_FB/test
done

