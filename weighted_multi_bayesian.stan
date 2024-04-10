functions{
  real normal_lb_rng(real mu, real sigma, real lb) { // normal distribution with a lower bound
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
  int<lower = 1> trials; // Number of trials
  int<lower=0> subjects;  // n of participants
  // array[trials] int <lower = 1, upper = 8> first_rating; // Participant rating
  // array[trials] int <lower = 1, upper = 8> other_rating; // Group rating
  // array[trials] int <lower = 1, upper = 8> second_rating; // Participant rating number two
  array[trials, subjects] real <lower = 0, upper = 1> first_rating; // Participant rating
  array[trials, subjects] real <lower = 0, upper = 1> other_rating; // Group rating
  array[trials, subjects] int <lower = 0, upper = 1> second_rating; // Participant rating number two
}

transformed data {
  array[trials, subjects] real l_Source1;
  array[trials, subjects] real l_Source2;

  
  l_Source1 = logit(first_rating);
  l_Source2 = logit(other_rating);
}

parameters {
  real biasM;
  real<lower = 0> biasSD;
  array[subjects] real z_bias;
  
  real<lower = 0, upper = 1> weight1M;
  real<lower = 0> weight1SD;
  real<lower = 0, upper = 1> weight2M;
  real<lower = 0> weight2SD;
  array[subjects] real z_weight1;
  array[subjects] real z_weight2;
}

transformed parameters {
  vector[subjects] bias;
  vector[subjects] weight1;
  vector[subjects] weight2;
  bias = biasM + (biasSD * to_vector(z_bias));
  weight1 = weight1M + (weight1SD * to_vector(z_weight1));
  weight2 = weight2M + (weight2SD * to_vector(z_weight2));
}

model {
  target +=  normal_lpdf(biasM | 0, 1);
  target +=  normal_lpdf(biasSD | 0, 1) - normal_lccdf(0 | 0, 1);
  target +=  beta_lpdf(weight1M | 2, 1);
  target +=  normal_lpdf(weight1SD | 0, 1) - normal_lccdf(0 | 0, 1);
  target +=  beta_lpdf(weight2M | 1, 1);
  target +=  normal_lpdf(weight2SD | 0, 1) - normal_lccdf(0 | 0, 1);
  
  target += std_normal_lpdf(to_vector(z_bias));
  target += std_normal_lpdf(to_vector(z_weight1));
  target += std_normal_lpdf(to_vector(z_weight2));
  
  for (s in 1:subjects){
    target +=  bernoulli_logit_lpmf(second_rating[,s] |  bias[s] +  weight1[s] * to_vector(l_Source1[,s]) +  weight2[s] * to_vector(l_Source2[,s]));
  }
}

generated quantities {
  real biasM_prior;
  real biasSD_prior;
  real weight1M_prior;
  real weight1SD_prior;
  real weight2M_prior;
  real weight2SD_prior;
  // real z_bias_prior;
  // array[trials] real log_lik;
  // 
  biasM_prior = normal_rng(0, 1);
  biasSD_prior = normal_lb_rng(0, 1, 0);
  weight1M_prior = beta_rng(1, 1);
  weight1SD_prior = normal_lb_rng(0, 1, 0);
  weight2M_prior = beta_rng(1, 1);
  weight2SD_prior = normal_lb_rng(0, 1, 0);
}
