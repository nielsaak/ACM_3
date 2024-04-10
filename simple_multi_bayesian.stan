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
  // vector<lower = 0, upper = 1>[trials] first_transform;
  // vector<lower = 0, upper = 1>[trials] other_transform;
  // vector<lower = 0, upper = 1>[trials] second_transform;
  // vector[trials] l_Source1;
  // vector[trials] l_Source2;
  // vector[trials] l_Output;
  // array[trials] int <lower = 0, upper = 1> Output;
  
  array[trials, subjects] real l_Source1;
  array[trials, subjects] real l_Source2;
  
  // first_transform = (to_vector(first_rating)+0.5)/10;
  // other_transform = (to_vector(other_rating)+0.5)/10;
  // second_transform = (to_vector(second_rating)+0.5)/10;

  // l_Source1 = logit(first_transform);
  // l_Source2 = logit(other_transform);
  // l_Output = logit(second_transform);
  
  l_Source1 = logit(first_rating);
  l_Source2 = logit(other_rating);
  // l_Output = logit(second_rating);

  // Output = bernoulli_rng(l_Output);

  // for (t in 1:trials){
  //   first_transform[t] = (first_rating+0.5)/10
  //   other_transform[t] = (first_rating+0.5)/10
  //   second_transform[t] = (first_rating+0.5)/10
  //   
  //   l_Source1[t] = logit(first_transform[t])
  //   l_Source2[t] = logit(other_transform[t])
  //   l_Output[t] = logit(second_transform[t])
  //   
  //   Output[t] = bernoulli_rng(l_Output[t]);
  // }

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
  // real z_bias_prior;
  // array[trials] real log_lik;
  // 
  biasM_prior = normal_rng(0, 1);
  biasSD_prior = normal_lb_rng(0, 1, 0);
  // z_bias_prior = normal_rng(0, 1);
}
