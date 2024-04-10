data {
  int<lower = 1> trials; // Number of trials
  // array[trials] int <lower = 1, upper = 8> first_rating; // Participant rating
  // array[trials] int <lower = 1, upper = 8> other_rating; // Group rating
  // array[trials] int <lower = 1, upper = 8> second_rating; // Participant rating number two
  array[trials] real <lower = 0, upper = 1> first_rating; // Participant rating
  array[trials] real <lower = 0, upper = 1> other_rating; // Group rating
  array[trials] int <lower = 0, upper = 1> second_rating; // Participant rating number two
}

transformed data {
  // vector<lower = 0, upper = 1>[trials] first_transform;
  // vector<lower = 0, upper = 1>[trials] other_transform;
  // vector<lower = 0, upper = 1>[trials] second_transform;
  // vector[trials] l_Source1;
  // vector[trials] l_Source2;
  // vector[trials] l_Output;
  // array[trials] int <lower = 0, upper = 1> Output;
  
  array[trials] real l_Source1;
  array[trials] real l_Source2;
  
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
  real bias;
}

transformed parameters {
  
}

model {
  target +=  normal_lpdf(bias | 0, 1);
  target +=  bernoulli_logit_lpmf(second_rating | bias + 0.5 * to_vector(l_Source1) + 0.5 * to_vector(l_Source2));
  // target +=  bernoulli_logit_lpmf(second_rating | bias + to_vector(l_Source1) + to_vector(l_Source2));
}

generated quantities {
  real bias_prior;
  // array[trials] real log_lik;
  
  bias_prior = normal_rng(0, 1);
  
  // for (n in 1:trials){  
  //   log_lik[n] = bernoulli_logit_lpmf(second_rating[n] | bias + l_Source1[n] +  l_Source2[n]);
  //   
  // }
  
}
