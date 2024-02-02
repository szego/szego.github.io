functions {
    // Antonio R. Vargas
    // https://mathstat.dal.ca/~antoniov/oregon_grad_rates.html
    
    real binomial_rsupp_lpmf(int k, int n, real mu) {
        if (n > 1) {
            // nothing suppressed
            return binomial_logit_lpmf(k | n, mu);
        } else if (n == 1) {
            // cohort suppressed, but not graduates
            vector[10] lp;
            for (j in 0:9)
                lp[j+1] = binomial_logit_lpmf(k | k + j, mu);
            return log(0.1) + log_sum_exp(lp);
        } else {
            // both cohort and graduates suppressed
            vector[19] lp = [0,
                             log(19) + mu,
                             2*log(3) + log(19) + 2*mu,
                             log(3) + log(17) + log(19) + 3*mu,
                             2*log(2) + log(3) + log(17) + log(19) + 4*mu,
                             2*log(2) + 2*log(3) + log(17) + log(19) + 5*mu,
                             2*log(2) + log(3) + log(7) + log(17) + log(19) + 6*mu,
                             2*log(2) + log(3) + log(13) + log(17) + log(19) + 7*mu,
                             log(2) + 2*log(3) + log(13) + log(17) + log(19) + 8*mu,
                             log(2) + log(11) + log(13) + log(17) + log(19) + 9*mu,
                             log(2) + 2*log(3) + log(13) + log(17) + log(19) + 10*mu,
                             2*log(2) + log(3) + log(13) + log(17) + log(19) + 11*mu,
                             2*log(2) + log(3) + log(7) + log(17) + log(19) + 12*mu,
                             2*log(2) + 2*log(3) + log(17) + log(19) + 13*mu,
                             2*log(2) + log(3) + log(17) + log(19) + 14*mu,
                             log(3) + log(17) + log(19) + 15*mu,
                             2*log(3) + log(19) + 16*mu,
                             log(19) + 17*mu,
                             18*mu]';
            return log(0.1) - 18*log_sum_exp(0, mu) + log_sum_exp(lp);
        }
    }
}
data {
    // training data
    int<lower=1> N;
    int<lower=1> N_race_ethn_id;
    int graduates[N];
    int cohort[N];
    int female[N];
    int eng_learn[N];
    int disability[N];
    int homeless[N];
    int econ_disad[N];
    int race_ethn_id[N];
    
    // complete data
    int<lower=1> M;
    int c_graduates[M];
    int c_cohort[M];
    int c_female[M];
    int c_eng_learn[M];
    int c_disability[M];
    int c_homeless[M];
    int c_econ_disad[M];
    int c_race_ethn_id[M];
}
parameters {
    // intercept
    real a;
    
    // population effects
    real b_f;
    real b_el;
    real b_d;
    real b_h;
    real b_ed;
    real b_el_d;
    real b_el_h;
    real b_el_ed;
    real b_d_h;
    real b_d_ed;
    real b_h_ed;
    real b_el_d_h;
    real b_el_d_ed;
    real b_el_h_ed;
    real b_d_h_ed;
    real b_el_d_h_ed;
    
    vector[N_race_ethn_id] a_re;  // race-ethnicity effects
    real<lower=0> sigma;          // std dev of race-ethnicity effects
}
transformed parameters {
    vector[N] mu;
    
    // linear model
    for(i in 1:N)
        mu[i] = a + a_re[race_ethn_id[i]] + b_f*female[i] + b_el*eng_learn[i] +
                b_d*disability[i] + b_h*homeless[i] + b_ed*econ_disad[i] +
                b_el_d*eng_learn[i]*disability[i] + b_el_h*eng_learn[i]*homeless[i] +
                b_el_ed*eng_learn[i]*econ_disad[i] + b_d_h*disability[i]*homeless[i] +
                b_d_ed*disability[i]*econ_disad[i] + b_h_ed*homeless[i]*econ_disad[i] +
                b_el_d_h*eng_learn[i]*disability[i]*homeless[i] +
                b_el_d_ed*eng_learn[i]*disability[i]*econ_disad[i] +
                b_el_h_ed*eng_learn[i]*homeless[i]*econ_disad[i] +
                b_d_h_ed*disability[i]*homeless[i]*econ_disad[i] +
                b_el_d_h_ed*eng_learn[i]*disability[i]*homeless[i]*econ_disad[i];
}
model {
    // fixed priors for the intercept and population effects
    a ~ normal(0, 4);
    b_f ~ normal(0, 2);
    b_el ~ normal(0, 2);
    b_d ~ normal(0, 2);
    b_h ~ normal(0, 2);
    b_ed ~ normal(0, 2);
    b_el_d ~ normal(0, 2);
    b_el_h ~ normal(0, 2);
    b_el_ed ~ normal(0, 2);
    b_d_h ~ normal(0, 2);
    b_d_ed ~ normal(0, 2);
    b_h_ed ~ normal(0, 2);
    b_el_d_h ~ normal(0, 2);
    b_el_d_ed ~ normal(0, 2);
    b_el_h_ed ~ normal(0, 2);
    b_d_h_ed ~ normal(0, 2);
    b_el_d_h_ed ~ normal(0, 2);
    
    // priors for the hierarchical structure on race-ethnicity
    sigma ~ gamma(1.5, 1);
    a_re ~ normal(0, sigma);
    
    for (i in 1:N)
        target += binomial_rsupp_lpmf(graduates[i] | cohort[i], mu[i]);
}
generated quantities {
    // Compute the log-likelihood for the complete dataset
    vector[M] log_lik;
    vector[M] mu_c;
    for (i in 1:M) {
        mu_c[i] = a + a_re[c_race_ethn_id[i]] + b_f*c_female[i] + b_el*c_eng_learn[i] + 
                b_d*c_disability[i] + b_h*c_homeless[i] + b_ed*c_econ_disad[i] +
                b_el_d*c_eng_learn[i]*c_disability[i] + b_el_h*c_eng_learn[i]*c_homeless[i] +
                b_el_ed*c_eng_learn[i]*c_econ_disad[i] + b_d_h*c_disability[i]*c_homeless[i] +
                b_d_ed*c_disability[i]*c_econ_disad[i] + b_h_ed*c_homeless[i]*c_econ_disad[i] +
                b_el_d_h*c_eng_learn[i]*c_disability[i]*c_homeless[i] +
                b_el_d_ed*c_eng_learn[i]*c_disability[i]*c_econ_disad[i] +
                b_el_h_ed*c_eng_learn[i]*c_homeless[i]*c_econ_disad[i] +
                b_d_h_ed*c_disability[i]*c_homeless[i]*c_econ_disad[i] +
                b_el_d_h_ed*c_eng_learn[i]*c_disability[i]*c_homeless[i]*c_econ_disad[i];
        log_lik[i] = binomial_rsupp_lpmf(c_graduates[i] | c_cohort[i], mu_c[i]);
    }
}
