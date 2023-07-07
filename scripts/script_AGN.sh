#!/bin/bash

# bh & ell_bh => AGN

mkdir ../AGN
cd ../AGN
mkdir test
cd ..

for i in ./mock_images/faceon_1114/bh*.png; do 
cp $i ./AGN/test
done

for i in ./mock_images/faceon_1114/ell_bh*.png; do 
cp $i ./AGN/test
done

for i in ./mock_images/pynbody_rt_0927/bh*.png; do 
cp $i ./AGN/test
done

for i in ./mock_images/pynbody_rt_0927/ell_bh*.png; do 
cp $i ./AGN/test
done