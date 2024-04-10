data {
  int<lower = 1> trials; // Number of trials
  array[trials] int <lower = 0, upper = 8> first_rating; // Participant rating
  array[trials] int <lower = 0, upper = 8> other_rating; // Group rating
  array[trials] int <lower = 0, upper = 8> second_rating; // Participant rating number two
}

transformed data {

}

parameters {
  real<lower = 0> bias;
}

transformed parameters {
  array[trials] real lp;
  array[trials] real <lower=0> mu;
  
  for (i in 1:trials) {
    // Linear predictor
    // It seems to make the most sense if lp is between 0 and 2
    lp[i] = bias + first_rating[i] + other_rating[i];
    
    // Mean
    mu[i] = exp(lp[i]);
    }
}

model {
  target +=  normal_lpdf(bias | 0, 2);
  
  target +=  poisson_lpmf(second_rating | mu);
}

generated quantities {
  // real bias_prior;
  // // array[trials] real log_lik;
  // 
  // bias_prior = normal_rng(0, 2);
  // 
  // for (n in 1:trials){  
  //   log_lik[n] = bernoulli_logit_lpmf(second_rating[n] | bias + l_Source1[n] +  l_Source2[n]);
  //   
  // }
  
}
