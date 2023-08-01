#!/bin/bash

# classic & ell_wobh => NOAGN

mkdir ../NOAGN
cd ../NOAGN
mkdir test
cd ..

for i in ./mock_images/faceon_1114/classic*.png; do 
cp $i ./NOAGN/test
done

for i in ./mock_images/faceon_1114/ell_wobh*.png; do 
cp $i ./NOAGN/test
done

for i in ./mock_images/pynbody_rt_0927/classic*.png; do 
cp $i ./NOAGN/test
done

for i in ./mock_images/pynbody_rt_0927/ell_wobh*.png; do 
cp $i ./NOAGN/test
done