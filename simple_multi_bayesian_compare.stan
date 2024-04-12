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
  array[trials, subjects] real <lower = 0, upper = 1> first_rating; // Participant rating
  array[trials, subjects] real <lower = 0, upper = 1> other_rating; // Group rating
  array[trials, subjects] int <lower = 0, upper = 1> second_rating; // Participant rating number two
  
  int<lower=0> subjects_test;  // n of participants
  array[trials, subjects_test] real <lower = 0, upper = 1> first_rating_test; // Participant rating
  array[trials, subjects_test] real <lower = 0, upper = 1> other_rating_test; // Group rating
  array[trials, subjects_test] int <lower = 0, upper = 1> second_rating_test; // Participant rating number two 
}

transformed data {
  array[trials, subjects] real l_Source1;
  array[trials, subjects] real l_Source2;
  array[trials, subjects_test] real l_Source1_test;
  array[trials, subjects_test] real l_Source2_test;
  
  l_Source1 = logit(first_rating);
  l_Source2 = logit(other_rating);
  l_Source1_test = logit(first_rating_test);
  l_Source2_test = logit(other_rating_test);
}

parameters {
  real biasM;
  real<lower = 0> biasSD;
  array[subjects] real z_bias;
}

transformed parameters {
  vector[subjects] biasC;
  vector[subjects] bias;
  biasC = biasSD * to_vector(z_bias);
  bias = biasM + biasC;
}

model {
  target +=  normal_lpdf(biasM | 0, 1);
  target +=  normal_lpdf(biasSD | 0, 1) - normal_lccdf(0 | 0, 1);
  
  target += std_normal_lpdf(to_vector(z_bias));
  
  for (s in 1:subjects){
    target +=  bernoulli_logit_lpmf(second_rating[,s] |  bias[s] + 0.5 * to_vector(l_Source1[,s]) + 0.5 * to_vector(l_Source2[,s]));
  }
}

generated quantities {
  real biasM_prior;
  real biasSD_prior;
  array[trials,subjects] real log_lik;
  array[trials,subjects_test] real log_lik_test;

  biasM_prior = normal_rng(0, 1);
  biasSD_prior = normal_lb_rng(0, 1, 0);
  
  for(n in 1:trials){
    for (s in 1:subjects){
      log_lik[n,s] = bernoulli_logit_lpmf(second_rating[n,s] | bias[s] + 0.5 * l_Source1[n,s] + 0.5 * l_Source2[n,s]);
    }
  }
  
  for(n in 1:trials){
    for (s in 1:subjects_test){
      log_lik_test[n,s] = bernoulli_logit_lpmf(second_rating_test[n,s] | biasM + 0.5 * l_Source1_test[n,s] + 0.5 * l_Source2_test[n,s]);
    }
  }
}
