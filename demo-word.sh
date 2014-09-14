make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 7 -negative 1 -hs 1 -sample 1e-3 -threads 1 -binary 1 -save-vocab voc #> out 2>&1
