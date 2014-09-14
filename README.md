word2vec_cbow
=============

this is a high performance cuda porting of cbow model of word2vec

I tested the code on K40 with Intel E5-2650 CPU, its performance is about 15xx K words/thread/sec, it is about 20x faster as cpu version run with 16 threads. 
AFAIK, this is the fastest implementation. 
