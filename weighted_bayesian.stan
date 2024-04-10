data {
  int<lower = 1> trials; // Number of trials
  array[trials] real <lower = 0, upper = 1> first_rating; // Participant rating
  array[trials] real <lower = 0, upper = 1> other_rating; // Group rating
  array[trials] int <lower = 0, upper = 1> second_rating; // Participant rating number two
}

transformed data {
  array[trials] real l_Source1;
  array[trials] real l_Source2;
  l_Source1 = logit(first_rating);
  l_Source2 = logit(other_rating);
}

parameters {
  real bias;
  // meaningful weights are btw 0.5 and 1 (theory reasons)
  // real<lower = 0.5, upper = 1> w1; 
  // real<lower = 0.5, upper = 1> w2;
  // real w1; 
  // real w2;
  real<lower = 0, upper = 1> weight1;
  real<lower = 0, upper = 1> weight2;
}

transformed parameters {
  // real<lower = 0, upper = 1> weight1;
  // real<lower = 0, upper = 1> weight2;
  // // weight parameters are rescaled to be on a 0-1 scale (0 -> no effects; 1 -> face value)
  // weight1 = (w1 - 0.5) * 2;  
  // weight2 = (w2 - 0.5) * 2;
}

model {
  target += normal_lpdf(bias | 0, 1);
  target += beta_lpdf(weight1 | 1, 1);
  target += beta_lpdf(weight2 | 1, 1);
  target += bernoulli_logit_lpmf(second_rating | bias + weight1 * to_vector(l_Source1) + weight2 * to_vector(l_Source2));
}

generated quantities {
  real bias_prior;
  real weight1_prior;
  real weight2_prior;
  // real w1_prior;
  // real w2_prior;

  bias_prior = normal_rng(0, 1);
  weight1_prior = beta_rng(1,1);
  weight2_prior = beta_rng(1,1);
  // w1_prior = 0.5 + beta_rng(1,1)/2;
  // w2_prior = 0.5 + beta_rng(1,1)/2;
  
}
