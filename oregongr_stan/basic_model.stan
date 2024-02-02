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
    int graduates[N];
    int cohort[N];
    
    // complete data
    int<lower=1> M;
    int c_graduates[M];
    int c_cohort[M];
}
parameters {
    real mu;  // logit graduation rate
}
model {
    int n_missing = 0;
    
    for (i in 1:N)
        if (cohort[i] >= 1)
            target += binomial_rsupp_lpmf(graduates[i] | cohort[i], mu);
        else
            n_missing += 1;
    
    // instead of calling this for each missing row, call it
    // once and multiply by the number of missing rows
    target += n_missing * binomial_rsupp_lpmf(0 | 0, mu);
}
generated quantities {
    // Compute the log-likelihood for the complete dataset
    vector[M] log_lik;
    real bc = binomial_rsupp_lpmf(0 | 0, mu);  // computed once and re-used
    for (i in 1:M)
        if (c_cohort[i] >= 1)
            log_lik[i] = binomial_rsupp_lpmf(c_graduates[i] | c_cohort[i], mu);
        else
            log_lik[i] = bc;
}
