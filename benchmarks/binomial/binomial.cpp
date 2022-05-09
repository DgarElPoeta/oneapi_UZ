Binomial* binomial = reinterpret_cast<Binomial*>(opts->p_problem);

float* randArray = (float*)binomial->a;
float* refOutput = (float*)binomial->b;

int numSamples = size * 4;// / binomial->steps1;
auto offset_workgroups = offset * 4;// / binomial->steps1;
int numSteps = binomial->steps;

float* stepsArray = (float*)malloc((numSteps + 1) * sizeof(float4));

// Iterate for all samples
for (int bid = 0 + offset_workgroups; bid < numSamples + offset_workgroups; ++bid) {
  float s[4];
  float x[4];
  float vsdt[4];
  float puByr[4];
  float pdByr[4];
  float optionYears[4];

  float inRand[4];

  for (int i = 0; i < 4; ++i) {
    inRand[i] = randArray[bid + i];
    s[i] = (1.0f - inRand[i]) * 5.0f + inRand[i] * 30.f;
    x[i] = (1.0f - inRand[i]) * 1.0f + inRand[i] * 100.f;
    optionYears[i] = (1.0f - inRand[i]) * 0.25f + inRand[i] * 10.f;
    float dt = optionYears[i] * (1.0f / (float)numSteps);
    vsdt[i] = VOLATILITY * sqrtf(dt);
    float rdt = RISKFREE * dt;
    float r = expf(rdt);
    float rInv = 1.0f / r;
    float u = expf(vsdt[i]);
    float d = 1.0f / u;
    float pu = (r - d) / (u - d);
    float pd = 1.0f - pu;
    puByr[i] = pu * rInv;
    pdByr[i] = pd * rInv;
  }
  /**
   * Compute values at expiration date:
   * Call option value at period end is v(t) = s(t) - x
   * If s(t) is greater than x, or zero otherwise...
   * The computation is similar for put options...
   */
  for (int j = 0; j <= numSteps; j++) {
    for (int i = 0; i < 4; ++i) {
      float profit = s[i] * expf(vsdt[i] * (2.0f * j - numSteps)) - x[i];
      stepsArray[j * 4 + i] = profit > 0.0f ? profit : 0.0f;
    }
  }

  /**
   * walk backwards up on the binomial tree of depth numSteps
   * Reduce the price step by step
   */
  for (int j = numSteps; j > 0; --j) {
    for (int k = 0; k <= j - 1; ++k) {
      for (int i = 0; i < 4; ++i) {
        int index_k = k * 4 + i;
        int index_k_1 = (k + 1) * 4 + i;
        stepsArray[index_k] = pdByr[i] * stepsArray[index_k_1] + puByr[i] * stepsArray[index_k];
      }
    }
  }

  // Copy the root to result
  refOutput[bid] = stepsArray[0];
}

free(stepsArray);
