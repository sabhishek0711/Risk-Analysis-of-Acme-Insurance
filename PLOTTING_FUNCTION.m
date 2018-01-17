%% CONVERGENCE PLOTS

yrs = [1000;10000;20000;50000;70000];

var_99 = [102566.6143
334363.402
342518.0289
330731.5631
335574.6754
];

var_99_SC = [3414.634146
16186.82999
17824.27528
17524.9654
17824.27528
];

var_99_NC = [34442.49356
54837.11083
55255.05949
54825.75783
54980.88909
];

AAL = [184214.525
236026.6008
248812.001
241398.8992
241133.3296
];
AAL_SC = [534.6318375
652.5831181
697.0802088
680.7138252
679.9869101
];
AAL_NC = [1276.734647
1936.600005
1938.940206
1834.738078
1830.51975
];

tvar_90 = [1037146.957
2152162.137
2190775.996
2005370.733
2017075.793
];
tvar_90_SC = [1138.211382
5395.609996
5941.425094
5841.655133
5941.425094
];
tvar_90_NC = [11622.00753
19718.65772
20026.39852
19865.91829
19909.91484
];

figure;
plot(yrs,var_99,yrs,var_99_SC,yrs,var_99_NC);
grid on;
xlabel('No. of years of simulation');
ylabel('Value at Risk at p = 0.99');
legend('Portfolio Level','Location Level-SC','Location Level-NC');
title('Convergence plot on the basis of VAR');

figure;
semilogy(yrs,tvar_90,yrs,tvar_90_SC,yrs,tvar_90_NC);
grid on;
xlabel('No. of years of simulation');
ylabel('Value at Risk at p = 0.90');
legend('Portfolio Level','Location Level-SC','Location Level-NC');
title('Convergence plot on the basis of TVAR');

figure;
semilogy(yrs,AAL,yrs,AAL_SC,yrs,AAL_NC);
grid on;
xlabel('No. of years of simulation');
ylabel('Average annual Loss($)');
legend('Portfolio Level','Location Level-SC','Location Level-NC');
title('Convergence plot on the basis of AAL');

%% Sensitivity Study

% Effect of Initial Capital on Probability of Ruin
I_C = [1.5 2 5 10]*10^6;
p_ruin = [0.0018 0.00172 0.0013 0.00066];

figure;
plot (I_C,p_ruin);
grid on;
xlabel('Initial Capital ($)');
ylabel('Probability of Ruin');
title('Effect of Initial Capital on Probability of Ruin');

premium = [1 1.5 1.7 1.9]*10^3;
p_ruin = [0.00366 0.0018 0.00116 0.0007];

figure;
plot(premium,p_ruin);
grid on;
xlabel('Premium ($)');
ylabel('Probability of Ruin');
title('Effect of Premium on Probability of Ruin');

deduct = [0.01 0.03 0.05 0.1]*10^6;
p_ruin = [0.0018 0.00054 0.00036 0.00008];
AAL = [2.41 1.7497 1.3618 .81657]*10^5;
AAL_NC = [4.7703 3.4007 2.6305 1.3719]*10^4;
AAL_SC = [1.9536 1.4211 1.1072 .66404]*10^5;

figure;
plot(deduct,p_ruin);
grid on;
xlabel('Deductable ($)');
ylabel('Probability of Ruin');
title('Effect of Deductable on Probability of Ruin');

figure;
plot(deduct,AAL);
grid on;
xlabel('Deductable ($)');
ylabel('Average Annual Loss for entire portfolio($)');
title('Effect of Deductable on AAL of entire portfolio');

figure;
plot(deduct,AAL_SC);
grid on;
xlabel('Deductable ($)');
ylabel('Average Annual Loss for Southern California($)');
title('Effect of Deductable on AAL of SC');

figure;
plot(deduct,AAL_NC);
grid on;
xlabel('Deductable ($)');
ylabel('Average Annual Loss for Northern California ($)');
title('Effect of Deductable on AAL of NC');

% Probability of ruin wrt number of years
p_ruin = [0.09
0.009
0.0045
0.0018
0.001285714
];

yrs = [1000
10000
20000
50000
70000
];

figure;
plot(yrs,p_ruin);
grid on;
xlabel('Years of simulations');
ylabel('Probability of Ruin');

event_no = [444 909 1801 4464 6264];

figure;
plot(yrs,event_no);
grid on;
xlabel('Years of simulations');
ylabel('Total number of events during the simulation period');




    
