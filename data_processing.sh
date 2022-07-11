#echo 'Downloading DeepFix raw dataset...'
#wget https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip -P data/deepfix_raw_data
#cd data/deepfix_raw_data
#unzip prutor-deepfix-09-12-2017.zip
#mv prutor-deepfix-09-12-2017/* ./
#rm -rf prutor-deepfix-09-12-2017 prutor-deepfix-09-12-2017.zip
#gunzip prutor-deepfix-09-12-2017.db.gz
#mv prutor-deepfix-09-12-2017.db dataset.db
#cd ../..

echo 'Downloading DrRepair dataset...'
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--orig-deepfix.zip -P data_processing


cd data_processing
unzip err-data-compiler--auto-corrupt--orig-deepfix.zip
rm err-data-compiler--auto-corrupt--orig-deepfix.zip
mv err-data-compiler--auto-corrupt--orig-deepfix DrRepair_deepfix

cd DrRepair_deepfix
mv bin0/* ./
mv bin1/* ./
mv bin2/* ./
mv bin3/* ./
mv bin4/* ./
rm -rf bin0 bin1 bin2 bin3 bin4
cd ../..

echo 'DrRepair Data generation...'
python data_processing/DrRepair_deepfix/data_generator.py
