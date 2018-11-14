Bayesian Modeling with R and Stan
========================================================
author: Sean Raleigh
date: November 14, 2018
autosize: true

R Users Group, Salt Lake City, UT

<style>
.small-code pre code {
  font-size: 1em;
}
</style>

========================================================

<div align="center">
<img src="./images/Westminster_Logo_primary_night.jpg" alt="Westminster College" width = 90% height = 90%>
</div>


========================================================

<div align="center">
<img src="./images/QUARC_logo_text.jpg" alt="QUARC" width = 70% height = 70%>
</div>


Preliminaries
========================================================

Load necessary packages:

```r
library(tidyverse)
library(triangle) # triangular distribution
library(rstan)
```

Tell Stan not to recompile code that has already been compiled:


```r
rstan_options(auto_write = TRUE)
```


Bayes's Theorem
========================================================

$$Pr(A, B)$$


Bayes's Theorem
========================================================
transition: none

$$Pr(A, B) = Pr(A) Pr(B \mid A)$$


Bayes's Theorem
========================================================
transition: none

$$
\begin{align*}
Pr(A, B)    &= Pr(A) Pr(B \mid A) \\
            &= Pr(B) Pr(A \mid B)
\end{align*}
$$


Bayes's Theorem
========================================================
transition: none

$$Pr(A \mid B) = \frac{Pr(A) Pr(B \mid A)}{Pr(B)}$$


Bayesian Data Analysis
========================================================

$$Pr(\theta \mid X) = \frac{Pr(\theta) Pr(X \mid \theta)}{Pr(X)}$$


Bayesian Data Analysis
========================================================
transition: none

$$Pr(\theta \mid X) = \frac{Pr(\theta) Pr(X \mid \theta)}{\displaystyle{\int_{\theta} Pr(\theta) Pr(X \mid \theta) \, d\theta}}$$


Bayesian Data Analysis
========================================================
transition: none

$$Pr(\theta \mid X) \propto Pr(\theta) Pr(X \mid \theta)$$


Bayesian Data Analysis
========================================================
transition: none

$$
\begin{align*}
Pr(\theta \mid X)   &\propto Pr(\theta) Pr(X \mid \theta) \\
posterior           &\propto prior \times likelihood
\end{align*}
$$


Frequentist vs Bayesian
========================================================

| Frequentist   | Bayesian  |
|---------------|-----------|
| Probability is "long-run frequency" | Probability is "degree of certainty" |
| $Pr(X \mid \theta)$ is a sampling distribution<br/>(function of $X$ with $\theta$ fixed) | $Pr(X \mid \theta)$ is a likelihood<br/>(function of $\theta$ with $X$ fixed) |
| No prior  | Prior |
| P-values (NHST) | Full probability model available for summary/decisions  |
| Confidence intervals  | Credible intervals    |
| Violates the "likelihood principle":<br/>&emsp;Sampling intention matters<br/>&emsp;Corrections for multiple testing<br/>&emsp;Adjustment for planned/post hoc testing | Respects the "likelihood principle":<br/>&emsp;Sampling intention is  irrelevant<br/>&emsp;No corrections for multiple testing<br/>&emsp;No adjustment for planned/post hoc testing |
| Objective?    | Subjective?   |



Binomial example
========================================================




In 18 trials, we observe 12 successes.

The likelihood function is expressed as follows:

$$p(X = 12 \mid \theta) \propto \theta^{12} (1 - \theta)^{6}$$


Binomial example
========================================================
transition: none



![plot of chunk unnamed-chunk-5](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-5-1.png)


Binomial example
========================================================
transition: none

Assume a uniform prior:

$$p(\theta) = 1.$$

Then the posterior is

$$p(\theta \mid X) \propto p(\theta) p(X \mid \theta) = p(X \mid \theta).$$


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-6](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-6-1.png)


Binomial example
========================================================
transition: none

Suppose we now choose a prior that is relatively far from the data, say, a normal distribution centered at 0.3 with standard deviation 0.1:

$$\theta \sim N(0.3, 0.1).$$


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-7](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-7-1.png)


Binomial example
========================================================
transition: none

What about a prior that is close to the data? Something like

$$\theta \sim N(0.7, 0.1).$$


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-8](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-8-1.png)


Binomial example
========================================================
transition: none

What about a triangular prior?


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-9](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-9-1.png)


Binomial example
========================================================
transition: none

Instead of 12 successes in 18 trials, suppose we only observe 2 successes in 3 trials.


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-10](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-10-1.png)


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-11](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-11-1.png)


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-12](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-12-1.png)


Binomial example
========================================================
transition: none

![plot of chunk unnamed-chunk-13](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-13-1.png)


Stan
========================================================

RStan requires data in a list:


```r
N <- 18  # Define the sample size
y <- c(rep(1, 12), rep(0, 6))  # 12 S, 6 F
stan_data <- list(N = N, y = y)
stan_data
```

```
$N
[1] 18

$y
 [1] 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
```


Stan
========================================================
transition: none


```r
bin_unif_model <-
'
data {
    int<lower = 0> N;
    int<lower = 0, upper = 1> y[N];
}
parameters {
    real<lower = 0, upper = 1> theta;
}
model {
    theta ~ uniform(0, 1);  // prior
    y ~ bernoulli(theta);   // likelihood
}
'
```


Stan
========================================================
transition: none


```r
bin_unif <- stan_model(model_code =
                           bin_unif_model)
```


Stan
========================================================
transition: none

Now we sample from the model using our data.


```r
set.seed(42)
fit_bin_unif <- sampling(bin_unif,
                         data = stan_data)
```

```

SAMPLING FOR MODEL 'faacd2dc663724f818fddd6389ea2950' NOW (CHAIN 1).
Chain 1: 
Chain 1: Gradient evaluation took 0 seconds
Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
Chain 1: Adjust your expectations accordingly!
Chain 1: 
Chain 1: 
Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 1: 
Chain 1:  Elapsed Time: 0.012 seconds (Warm-up)
Chain 1:                0.01 seconds (Sampling)
Chain 1:                0.022 seconds (Total)
Chain 1: 

SAMPLING FOR MODEL 'faacd2dc663724f818fddd6389ea2950' NOW (CHAIN 2).
Chain 2: 
Chain 2: Gradient evaluation took 0 seconds
Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
Chain 2: Adjust your expectations accordingly!
Chain 2: 
Chain 2: 
Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 2: 
Chain 2:  Elapsed Time: 0.011 seconds (Warm-up)
Chain 2:                0.01 seconds (Sampling)
Chain 2:                0.021 seconds (Total)
Chain 2: 

SAMPLING FOR MODEL 'faacd2dc663724f818fddd6389ea2950' NOW (CHAIN 3).
Chain 3: 
Chain 3: Gradient evaluation took 0 seconds
Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
Chain 3: Adjust your expectations accordingly!
Chain 3: 
Chain 3: 
Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 3: 
Chain 3:  Elapsed Time: 0.011 seconds (Warm-up)
Chain 3:                0.011 seconds (Sampling)
Chain 3:                0.022 seconds (Total)
Chain 3: 

SAMPLING FOR MODEL 'faacd2dc663724f818fddd6389ea2950' NOW (CHAIN 4).
Chain 4: 
Chain 4: Gradient evaluation took 0 seconds
Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
Chain 4: Adjust your expectations accordingly!
Chain 4: 
Chain 4: 
Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 4: 
Chain 4:  Elapsed Time: 0.012 seconds (Warm-up)
Chain 4:                0.01 seconds (Sampling)
Chain 4:                0.022 seconds (Total)
Chain 4: 
```


========================================================
transition: none

<div align="center">
<img src="./images/3_Chainz.png" alt="3 Chainz", width = 90% height = 90%>
</div>


Stan
========================================================
transition: none
class: small-code


```r
fit_bin_unif
```

```
Inference for Stan model: faacd2dc663724f818fddd6389ea2950.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

        mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
theta   0.64    0.00 0.10   0.43   0.57   0.65   0.72   0.83  1449    1
lp__  -13.46    0.02 0.72 -15.52 -13.63 -13.20 -13.00 -12.95  1769    1

Samples were drawn using NUTS(diag_e) at Tue Nov 13 16:49:12 2018.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```


Stan
========================================================
transition: none


```r
stan_dens(fit_bin_unif) + xlim(0,1) +
    slide_theme_1
```

![plot of chunk unnamed-chunk-19](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-19-1.png)


Stan
========================================================
transition: none


```r
bin_norm1_model <-
'
data {
    int<lower = 0> N;
    int<lower = 0, upper = 1> y[N];
}
parameters {
    real<lower = 0, upper = 1> theta;
}
model {
    theta ~ normal(0.3, 0.1);   // prior
    y ~ bernoulli(theta);       // likelihood
}
'
```


Stan
========================================================
transition: none


```r
bin_norm1 <- stan_model(model_code =
                            bin_norm1_model)
```


Stan
========================================================
transition: none


```r
set.seed(42)
fit_bin_norm1 <- sampling(bin_norm1,
                          data = stan_data)
```


Stan
========================================================
transition: none
class: small-code


```r
fit_bin_norm1
```

```
Inference for Stan model: b12fb38c7b0da2c8c3a4ac151466273e.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

        mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
theta   0.46    0.00 0.07   0.32   0.41   0.46   0.51   0.61  1221    1
lp__  -16.21    0.02 0.74 -18.26 -16.39 -15.92 -15.74 -15.69  1613    1

Samples were drawn using NUTS(diag_e) at Tue Nov 13 16:50:08 2018.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```


Stan
========================================================
transition: none


```r
stan_dens(fit_bin_norm1) + xlim(0,1) +
    slide_theme_1
```

![plot of chunk unnamed-chunk-24](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-24-1.png)


Real data
========================================================

Patient data:

* `individual_level_data.csv`
   - `program_id`
   - `score_admit`
   - `score_discharge`


```r
patients <- read_csv("./data/individual_level_data.csv")
```

26 programs.

Scores can be from -16 to 240. (Higher scores indicate more dysfunction.)


Real data
========================================================
transition: none


```r
patients
```

```
# A tibble: 1,613 x 3
   program_id score_admit score_discharge
        <int>       <int>           <int>
 1          1          15              -7
 2          1          57              15
 3          1          59             120
 4          1          36              20
 5          1          71              53
 6          1          60              67
 7          1          36               9
 8          1         142             119
 9          2         136              68
10          2          48              80
# ... with 1,603 more rows
```


Real data
========================================================
transition: none

Program data:

* `program_level_data.csv`
   - `program_id`
   - `program_type`
   - `n`


```r
programs <- read_csv("./data/program_level_data.csv")
```

Two types of programs ("A" and "B").


Real data
========================================================
transition: none


```r
programs
```

```
# A tibble: 26 x 3
   program_id program_type     n
        <int> <chr>        <int>
 1          1 A                8
 2          2 B               52
 3          3 A               21
 4          4 A               26
 5          5 A               19
 6          6 A                7
 7          7 B               15
 8          8 B              359
 9          9 B              116
10         10 A              117
# ... with 16 more rows
```


Real data
========================================================
transition: none


```r
N <- NROW(patients)
K <- NROW(programs)
score_discharge <- patients$score_discharge
program <- patients$program_id
score_admit <- patients$score_admit -
    mean(patients$score_admit) # Center score
program_type <-
    ifelse(programs$program_type == "A", 0, 1)

score_data <-
    list(N = N,
         K = K,
         score_discharge = score_discharge,
         program = program,
         score_admit = score_admit,
         program_type = program_type)
```


Real data
========================================================
transition: none


```r
score_model_data <-
'
data {
    int<lower = 0> N; // Sample size
    int<lower = 0> K; // Group size
    int<lower = 1, upper = K> program[N];

    vector[N] score_discharge;
    vector[N] score_admit;
    vector<lower = 0, upper = 1>[K] program_type;
}
'
```


Real data
========================================================
transition: none


```r
score_model_params <-
'
parameters {
    // hyperparameters
    vector[2] gamma_a;
    vector[2] gamma_b;
    real<lower = 0> sigma_a;
    real<lower = 0> sigma_b;

    // Intercepts
    vector[K] a;

    // Slopes
    vector[K] b;
    real<lower = 0> sigma_score;
}
'
```



========================================================
transition: none

<div align="center">
<img src="./images/Hierarchical_Xzibit.jpg" alt="Xzibit" width = 90% height = 90%>
</div>


Real data
========================================================
transition: none
class: small-code


```r
score_model_model <-
'
model {
    // hyperpriors
    gamma_a[1] ~ normal(112, 64);   // Group A intercepts
    gamma_a[2] ~ normal(0, 64);     // Group B intercept diffs
    gamma_b[1] ~ normal(0, 2);      // Group A slopes
    gamma_b[2] ~ normal(0, 1);      // Group B slope diffs
    sigma_a ~ normal(0, 50) T[0, ];
    sigma_b ~ normal(0, 50) T[0, ];

    // priors
    a ~ normal(gamma_a[1] + gamma_a[2] * program_type, sigma_a);
    b ~ normal(gamma_b[1] + gamma_b[2] * program_type, sigma_b);
    sigma_score ~ normal(0, 50) T[0, ];

    // likelihood
    for (i in 1:N) {
        score_discharge[i] ~ normal(a[program[i]] +
                                    b[program[i]] * score_admit[i],
                                    sigma_score);
    }
}
'
```


Real data
========================================================
transition: none


```r
score_model <- paste(score_model_data,
                     score_model_params,
                     score_model_model)
```


Real data
========================================================
transition: none


```r
score_stan <- stan_model(model_code =
                             score_model)
```


Real data
========================================================
transition: none


```r
set.seed(42)
score_fit <- sampling(score_stan,
                      data = score_data)
```


Real data
========================================================
transition: none
class: small-code


```r
score_fit
```

```
Inference for Stan model: 37688f5a342a03590db902aa949085ac.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

                mean se_mean   sd     2.5%      25%      50%      75%
gamma_a[1]     30.57    0.07 3.38    23.98    28.30    30.58    32.77
gamma_a[2]      4.31    0.10 5.42    -6.44     0.76     4.33     7.92
gamma_b[1]      0.26    0.00 0.05     0.17     0.23     0.26     0.29
gamma_b[2]      0.13    0.00 0.07     0.00     0.08     0.13     0.17
sigma_a        11.74    0.04 2.31     7.91    10.10    11.54    13.11
sigma_b         0.11    0.00 0.04     0.04     0.08     0.10     0.13
a[1]           44.29    0.11 7.64    29.22    39.12    44.26    49.49
a[2]           32.22    0.05 3.48    25.31    29.89    32.18    34.61
a[3]           25.63    0.06 5.05    15.64    22.26    25.73    28.99
a[4]           28.32    0.06 4.67    19.04    25.15    28.25    31.48
a[5]           27.25    0.09 5.48    16.87    23.49    27.31    30.93
a[6]           27.65    0.11 7.51    13.03    22.47    27.65    32.88
a[7]           46.60    0.08 5.91    35.17    42.55    46.55    50.56
a[8]           30.54    0.02 1.32    28.05    29.65    30.54    31.47
a[9]           39.90    0.03 2.38    35.28    38.28    39.89    41.55
a[10]          27.95    0.03 2.41    23.26    26.31    27.96    29.57
a[11]          47.67    0.03 1.85    44.02    46.45    47.70    48.90
a[12]          23.92    0.05 3.98    16.11    21.18    23.91    26.64
a[13]          27.51    0.10 7.25    13.08    22.57    27.62    32.31
a[14]          35.19    0.10 6.56    22.27    30.95    35.12    39.56
a[15]          23.94    0.10 6.83    10.46    19.35    23.99    28.69
a[16]          31.23    0.03 1.74    27.80    30.06    31.20    32.42
a[17]          31.06    0.10 6.12    19.24    26.87    31.12    35.28
a[18]          47.21    0.04 2.99    41.43    45.10    47.19    49.26
a[19]          54.00    0.05 2.72    48.71    52.10    53.96    55.94
a[20]          18.63    0.10 6.92     4.71    13.81    18.66    23.50
a[21]          40.63    0.11 8.30    24.49    35.10    40.43    46.11
a[22]          23.21    0.06 4.53    14.32    20.17    23.22    26.24
a[23]          32.64    0.04 2.71    27.43    30.80    32.59    34.48
a[24]          12.18    0.11 6.41    -0.14     7.79    12.22    16.52
a[25]          26.20    0.04 2.65    21.07    24.36    26.19    28.03
a[26]          25.85    0.12 8.58     8.95    20.32    25.72    31.56
b[1]            0.36    0.00 0.12     0.15     0.28     0.35     0.43
b[2]            0.33    0.00 0.08     0.17     0.28     0.33     0.38
b[3]            0.28    0.00 0.11     0.07     0.21     0.28     0.35
b[4]            0.26    0.00 0.09     0.09     0.20     0.26     0.32
b[5]            0.22    0.00 0.09     0.04     0.16     0.22     0.28
b[6]            0.23    0.00 0.10     0.02     0.17     0.24     0.30
b[7]            0.48    0.00 0.11     0.29     0.40     0.47     0.55
b[8]            0.30    0.00 0.04     0.22     0.27     0.30     0.33
b[9]            0.39    0.00 0.06     0.27     0.35     0.38     0.42
b[10]           0.22    0.00 0.05     0.12     0.19     0.22     0.26
b[11]           0.42    0.00 0.05     0.33     0.38     0.42     0.45
b[12]           0.24    0.00 0.08     0.10     0.20     0.25     0.29
b[13]           0.39    0.00 0.11     0.18     0.32     0.39     0.46
b[14]           0.28    0.00 0.11     0.06     0.21     0.28     0.35
b[15]           0.25    0.00 0.10     0.07     0.19     0.25     0.31
b[16]           0.29    0.00 0.05     0.17     0.25     0.29     0.32
b[17]           0.28    0.00 0.11     0.07     0.21     0.27     0.34
b[18]           0.51    0.00 0.09     0.35     0.45     0.51     0.57
b[19]           0.37    0.00 0.08     0.23     0.31     0.36     0.42
b[20]           0.21    0.00 0.09     0.02     0.15     0.21     0.27
b[21]           0.26    0.00 0.11     0.03     0.19     0.26     0.32
b[22]           0.24    0.00 0.09     0.06     0.18     0.24     0.29
b[23]           0.24    0.00 0.07     0.11     0.20     0.24     0.28
b[24]           0.33    0.00 0.11     0.09     0.27     0.34     0.40
b[25]           0.16    0.00 0.06     0.03     0.12     0.16     0.20
b[26]           0.25    0.00 0.12     0.01     0.18     0.25     0.32
sigma_score    25.52    0.01 0.46    24.61    25.21    25.51    25.83
lp__        -6054.74    0.39 8.21 -6070.75 -6060.03 -6054.94 -6049.61
               97.5% n_eff Rhat
gamma_a[1]     37.29  2610 1.00
gamma_a[2]     14.66  2961 1.00
gamma_b[1]      0.35  1324 1.00
gamma_b[2]      0.26  1635 1.00
sigma_a        16.83  2743 1.00
sigma_b         0.19   435 1.01
a[1]           59.00  4569 1.00
a[2]           38.97  5079 1.00
a[3]           35.63  6103 1.00
a[4]           37.45  5467 1.00
a[5]           38.01  3846 1.00
a[6]           41.74  4996 1.00
a[7]           58.36  5483 1.00
a[8]           33.08  5334 1.00
a[9]           44.55  5177 1.00
a[10]          32.60  5189 1.00
a[11]          51.18  4149 1.00
a[12]          31.61  5368 1.00
a[13]          41.68  5328 1.00
a[14]          48.38  4591 1.00
a[15]          36.86  4856 1.00
a[16]          34.68  4817 1.00
a[17]          42.96  3436 1.00
a[18]          52.91  5242 1.00
a[19]          59.32  3451 1.00
a[20]          32.21  4888 1.00
a[21]          57.09  5221 1.00
a[22]          32.04  5141 1.00
a[23]          38.02  3715 1.00
a[24]          24.56  3578 1.00
a[25]          31.37  5273 1.00
a[26]          42.66  4797 1.00
b[1]            0.64  1666 1.00
b[2]            0.48  3562 1.00
b[3]            0.50  2804 1.00
b[4]            0.44  3428 1.00
b[5]            0.39  3803 1.00
b[6]            0.43  3298 1.00
b[7]            0.71  2082 1.00
b[8]            0.37  3448 1.00
b[9]            0.50  4448 1.00
b[10]           0.33  4315 1.00
b[11]           0.51  3742 1.00
b[12]           0.39  2979 1.00
b[13]           0.62  3951 1.00
b[14]           0.51  2518 1.00
b[15]           0.44  3712 1.00
b[16]           0.39  3900 1.00
b[17]           0.50  2109 1.00
b[18]           0.69  1219 1.00
b[19]           0.52  1364 1.00
b[20]           0.39  3145 1.00
b[21]           0.51  2831 1.00
b[22]           0.40  3965 1.00
b[23]           0.37  3601 1.00
b[24]           0.54  3654 1.00
b[25]           0.28  2074 1.00
b[26]           0.48  3465 1.00
sigma_score    26.46  5714 1.00
lp__        -6037.55   441 1.00

Samples were drawn using NUTS(diag_e) at Tue Nov 13 16:52:27 2018.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```


Real data
========================================================
transition: none
class: small-code

46 is the clinical cutoff.


```r
score_summary <- as_tibble(summary(score_fit)$summary,
                           rownames = "param")
score_summary %>%
    filter(str_detect(.$param, pattern = "^a"))
```

```
# A tibble: 26 x 11
   param  mean se_mean    sd `2.5%` `25%` `50%` `75%` `97.5%` n_eff  Rhat
   <chr> <dbl>   <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>   <dbl> <dbl> <dbl>
 1 a[1]   44.3  0.113   7.64   29.2  39.1  44.3  49.5    59.0 4569. 1.000
 2 a[2]   32.2  0.0489  3.48   25.3  29.9  32.2  34.6    39.0 5079. 1.00 
 3 a[3]   25.6  0.0647  5.05   15.6  22.3  25.7  29.0    35.6 6103. 0.999
 4 a[4]   28.3  0.0632  4.67   19.0  25.1  28.2  31.5    37.5 5467. 0.999
 5 a[5]   27.2  0.0884  5.48   16.9  23.5  27.3  30.9    38.0 3846. 1.000
 6 a[6]   27.7  0.106   7.51   13.0  22.5  27.7  32.9    41.7 4996. 1.000
 7 a[7]   46.6  0.0799  5.91   35.2  42.5  46.5  50.6    58.4 5483. 1.000
 8 a[8]   30.5  0.0180  1.32   28.0  29.6  30.5  31.5    33.1 5334. 1.000
 9 a[9]   39.9  0.0330  2.38   35.3  38.3  39.9  41.5    44.5 5177. 0.999
10 a[10]  27.9  0.0334  2.41   23.3  26.3  28.0  29.6    32.6 5189. 1.000
# ... with 16 more rows
```


Real data
========================================================
transition: none
class: small-code

* Slopes are not expected to be negative.
    - More dysfunction at admission will predict more at discharge.
* But they should be less than 1.
    - Patients should not leave more dysfunctional than when they entered.


```r
score_summary %>%
    filter(str_detect(.$param, pattern = "^b"))
```

```
# A tibble: 26 x 11
   param  mean  se_mean     sd `2.5%` `25%` `50%` `75%` `97.5%` n_eff  Rhat
   <chr> <dbl>    <dbl>  <dbl>  <dbl> <dbl> <dbl> <dbl>   <dbl> <dbl> <dbl>
 1 b[1]  0.361 0.00301  0.123  0.153  0.277 0.346 0.429   0.642 1666. 1.00 
 2 b[2]  0.329 0.00129  0.0772 0.172  0.279 0.332 0.380   0.482 3562. 1.00 
 3 b[3]  0.278 0.00205  0.109  0.0711 0.207 0.276 0.346   0.504 2804. 1.00 
 4 b[4]  0.263 0.00155  0.0906 0.0872 0.204 0.262 0.319   0.441 3428. 1.000
 5 b[5]  0.222 0.00143  0.0882 0.0443 0.165 0.223 0.282   0.390 3803. 1.00 
 6 b[6]  0.234 0.00180  0.103  0.0172 0.170 0.236 0.300   0.435 3298. 1.00 
 7 b[7]  0.477 0.00236  0.108  0.286  0.403 0.469 0.546   0.710 2082. 1.00 
 8 b[8]  0.300 0.000669 0.0393 0.223  0.274 0.300 0.326   0.375 3448. 1.00 
 9 b[9]  0.385 0.000911 0.0608 0.267  0.346 0.385 0.425   0.504 4448. 1.000
10 b[10] 0.223 0.000826 0.0542 0.118  0.188 0.224 0.259   0.330 4315. 1.000
# ... with 16 more rows
```


Real data
========================================================
transition: none

Crosses are mean discharge scores per program. Hierarchical modeling "partially pools" toward the overall means.



![plot of chunk unnamed-chunk-40](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-40-1.png)


Real data
========================================================
transition: none

![plot of chunk unnamed-chunk-41](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-41-1.png)


Real data
========================================================
transition: none

Blue = "No pooling", Green = "Complete pooling"

Red = "Partial pooling (from multilevel model)"

![plot of chunk unnamed-chunk-42](Bayesian-Modeling-with-R-and-Stan-figure/unnamed-chunk-42-1.png)


Thank you!
========================================================
type:section

[sraleigh@westminstercollege.edu](mailto:sraleigh@westminstercollege.edu)

[github.com/VectorPosse](https://github.com/VectorPosse)

[rpubs.com/VectorPosse](http://rpubs.com/VectorPosse)
