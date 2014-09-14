__device__ real reduceInWarp(real f, int idInWarp){
	for(int i = warpSize/2; i > 0; i /= 2){
		f += __shfl_xor(f, i, 32);
	}
	return f;
}

template<long long hs>
void __global__ cbow_kernel(long window, long negative, float alpha, long sentence_length, const int* __restrict__ sen, long layer1_size, volatile float *syn0, volatile float *syn1, const float* __restrict__ expTable, const int* __restrict__ vocab_codelen, const char* __restrict__ vocab_code, const int* __restrict__ vocab_point, const int* __restrict__ table, long table_size, long vocab_size, volatile float *syn1neg){
	extern __shared__ real s[]; //2*(real *)calloc(layer1_size, sizeof(real));

	int numWarpsPerBlock = blockDim.x/warpSize;
	int warpIdInBlock = threadIdx.x / warpSize;
	int warpId = warpIdInBlock + numWarpsPerBlock*blockIdx.x;
	int idInWarp = threadIdx.x % warpSize;

	real *neu1 = s + warpIdInBlock * layer1_size;
	real *neu1e = s + (numWarpsPerBlock+warpIdInBlock) * layer1_size;
	volatile int* cw = (volatile int*)(s + 2*numWarpsPerBlock*layer1_size);

	volatile unsigned long *temp_rand = (volatile unsigned long*)(s+2*numWarpsPerBlock*layer1_size) + numWarpsPerBlock;

	for(int sentence_position = warpId; sentence_position < sentence_length; sentence_position += gridDim.x*numWarpsPerBlock){
		long long word = sen[sentence_position];
		if (word == -1) continue;
		if(0 == idInWarp) {
			temp_rand[warpIdInBlock] = sentence_position;
			temp_rand[warpIdInBlock] = temp_rand[warpIdInBlock] * (unsigned long)25214903917 + 11;
		}
		unsigned long next_random = temp_rand[warpIdInBlock];//rand();
		int b = next_random % window;
		for (int c = idInWarp; c < layer1_size; c += warpSize) neu1[c] = 0;
		for (int c = idInWarp; c < layer1_size; c += warpSize) neu1e[c] = 0;
		// in -> hidden
		cw[warpIdInBlock] = 0;
		__syncthreads();

		for(int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
			int c = sentence_position - window + a;
			if (c < 0) continue;
			if (c >= sentence_length) continue;
			long long last_word = sen[c];
			if (last_word == -1) continue;
			for (int c = idInWarp; c < layer1_size; c += warpSize) neu1[c] += syn0[c + last_word * layer1_size];
			if(idInWarp == 0) cw[warpIdInBlock]++;
		}
		if (cw[warpIdInBlock]) {
			for (int c = idInWarp; c < layer1_size; c += warpSize) neu1[c] /= cw[warpIdInBlock];
			if (hs) for (int d = vocab_codelen[word]; d < vocab_codelen[word+1]; d++) {
				float f = 0;
				int l2 = vocab_point[d] * layer1_size;
				// Propagate hidden -> output
				for (int c = idInWarp; c < layer1_size; c += warpSize) f += neu1[c] * syn1[c + l2];
				f = reduceInWarp(f, idInWarp);
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				// 'g' is the gradient multiplied by the learning rate
				float g = (1 - vocab_code[d] - f) * alpha;
				// Propagate errors output -> hidden
				for (int c = idInWarp; c < layer1_size; c += warpSize) neu1e[c] += g * syn1[c + l2];
				// Learn weights hidden -> output
				for (int c = idInWarp; c < layer1_size; c += warpSize) syn1[c + l2] += g * neu1[c];
			}
			// NEGATIVE SAMPLING
			if (negative > 0) for (int d = 0; d < negative + 1; d++) {
				int target;
				int label;
				if (d == 0) {
					target = word;
					label = 1;
				} else {
					if(0 == idInWarp) {
						temp_rand[warpIdInBlock] = temp_rand[warpIdInBlock] * (unsigned long)25214903917 + 11;
					}
					next_random = temp_rand[warpIdInBlock];
					//			
					target = table[(next_random >> 16) % table_size];
					if (target == 0) target = next_random % (vocab_size - 1) + 1;
					if (target == word) continue;
					label = 0;
				}
				int l2 = target * layer1_size;
				float f = 0;
				float g;
				for (int c = idInWarp; c < layer1_size; c += warpSize) f += neu1[c] * syn1neg[c + l2];
				f = reduceInWarp(f, idInWarp);
				if (f > MAX_EXP) g = (label - 1) * alpha;
				else if (f < -MAX_EXP) g = (label - 0) * alpha;
				else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
				for (int c = idInWarp; c < layer1_size; c += warpSize) neu1e[c] += g * syn1neg[c + l2];
				for (int c = idInWarp; c < layer1_size; c += warpSize) syn1neg[c + l2] += g * neu1[c];
			}
			// hidden -> in
			for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				int c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				long long last_word = sen[c];
				if (last_word == -1) continue;
				for (int c = idInWarp; c < layer1_size; c += warpSize) syn0[c + last_word * layer1_size] += neu1e[c];
			}
		}
	}
}

void cbow_cuda(long window, long negative, float alpha, long sentence_length, int *sen, long layer1_size, float *syn0, long hs, float *syn1, float *expTable, int *vocab_codelen, char *vocab_code, int *vocab_point, int *table, long table_size, long vocab_size, float *syn1neg){
	int blockSize = 256;
	int gridSize = (sentence_length)/(blockSize/32);
	size_t smsize = (blockSize/32)*(2*layer1_size+3)*sizeof(float);
//printf("sm size is %d\n", smsize);
//fflush(stdout);
	cbow_kernel<1><<<gridSize, blockSize, smsize>>>(window, negative, alpha, sentence_length, sen, layer1_size, syn0, syn1, expTable, vocab_codelen, vocab_code, vocab_point, table, table_size, vocab_size, syn1neg);
}
