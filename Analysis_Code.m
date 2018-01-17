%% ------RISK ANALSYS FOR ACME INSURANCE------
%% ------ ABHISHEK SARKAR ------------
%% ------ CEE - 209 Course Project ------------
   
clear all; close all;

load fault.mat;
load location.mat;

tic;

% Initial Parameters
loc_no = length(location(:,1));
Mag(:,2) = fault(:,2);
Mag(:,1) = 5;           
n_flt = 16;
a = fault(:,1);
Num_div = 10;           
yrs = 1000;           
R = 6371;              

% Calculation of fault length and coordinate tranformation
for i = 1:n_flt
    M(i,:) = linspace(Mag(i,1),Mag(i,2),Num_div);
    Nm(i,:) = 10.^(a(i) - M(i,:));
    N(i,:) = abs(diff(Nm(i,:)));
    fault_pos(i,:) = [R*cos(fault(i,4)*pi/180)*cos(fault(i,3)*pi/180) R*cos(fault(i,4)*pi/180)*sin(fault(i,3)*pi/180) R*sin(fault(i,4)*pi/180)...
                      R*cos(fault(i,6)*pi/180)*cos(fault(i,5)*pi/180) R*cos(fault(i,6)*pi/180)*sin(fault(i,5)*pi/180) R*sin(fault(i,6)*pi/180)];

    
     pos_xy(i,:) =  [fault_pos(i,1) fault_pos(i,2) fault_pos(i,4) fault_pos(i,5)];
     fault_length(i) = sqrt((pos_xy(i,3)-pos_xy(i,1))^2 + (pos_xy(i,4)-pos_xy(i,2))^2);
end

% Coordinate tranformation for location points
for i=1:loc_no
     place_loc(i,:) = [R*cos(location(i,1)*pi/180)*cos(location(i,2)*pi/180) R*cos(location(i,1)*pi/180)*sin(location(i,2)*pi/180) R*sin(location(i,1)*pi/180)];
     loc_xy(i,:) = [place_loc(i,1) place_loc(i,2)];
end

% Determining location of each point w.r.t faults
for i = 1:loc_no
    
    pt = loc_xy(i,:);
    
    for j=1:n_flt
        v1 = [pos_xy(j,1) pos_xy(j,2)];
        v2 = [pos_xy(j,3) pos_xy(j,4)];
        
        [projv2,projv1]=point_to_line(pt,v1,v2);
        
        if projv1> fault_length(j) || projv2> fault_length(j)
            fault_pnt_pos(i,j) = 1;                          % 1-for out; 0- for in
            
        else
            fault_pnt_pos(i,j) = 0; 
        end

    end
end

%Declaring V, Deductable, Loss Matrix
V = 10^6;
D = 0.01*V;
Loss = zeros(loc_no,yrs);
ro = 0.2;
rng(1);
counter = 1;

% Main Simulation for Loss Calculation
for i=1:yrs
    u = lhsdesign(n_flt,1);
    for j=1:n_flt   % number of faults
               
        m(j) = u(j)*(Mag(j,2) - Mag(j,1)) + Mag(j,1);
        [c index] = min(abs(M(j,:)-m(j)));
        
         if m(j) > M(j,index)
             n(i,j) = poissrnd(N(j,index));
         else
             n(i,j) = poissrnd(N(j,index-1));
         end
         
         SRL(j) = exp(-3.55 + 0.74*m(j));     
                         
         for p=1:n(i,j)
             z = norminv(rand());
            for  k = 1:loc_no
                
                  y = norminv(rand());
                  X = sqrt(ro)*z + sqrt(1-ro)*y;
                  quantile = normcdf(X);
                  
                  pt = loc_xy(k,:);
             
                 % Shortest distance calculation
                 d(j) = shortest_distance(pt,fault_length(j),fault_pnt_pos(k,j),pos_xy(j,:),SRL(j));               % a vector   1 TO BE MODIFIED
                 R = sqrt(d(j)^2 + 4.91^2);
                 mu_Sa(j) =exp(0.212 + 0.831*(m(j)-6) - 0.120*(m(j)-6)^2 - 0.867*log(R) - 0.487*log(500/1954));
                 sigma_Sa = 0.538;
                 Sa = logninv(rand(),mu_Sa(j),sigma_Sa);
                 Sa = exp(mu_Sa(j) + 2*sigma_Sa);        
                 if Sa ~=0
                     mu_L = V*normcdf(log(Sa/3)/1.5);
                     sigma_L = min(mu_L*(1/(sqrt(mu_L/V)) - 1), 4*mu_L);
                     sigma_lnL = sqrt(log((sigma_L/mu_L)^2 + 1));
                     mu_lnL = log(mu_L) - 0.5*(sigma_lnL)^2;
                 else
                     sigma_lnL=0;
                     mu_lnL=0;
                 end
                 loss = logninv(quantile,mu_lnL,sigma_lnL);
                 loss = min(V-D,loss-D);
                 loss = max(0,loss-D);
                 Loss(k,i) = Loss(k,i) + loss;
    
            end
            Loss_per_event(counter) = sum(Loss(:,i));
            counter = counter + 1;               % No of events
         end
    end 
end

 
% Calculation of profit of a company.
Prem_per_yr = 1500;

Premium = length(location(:,1))*Prem_per_yr;           
PremiumNCal = length(location(287:312,1))*Prem_per_yr; 
PremiumSCal = length(location(1:287,1))*Prem_per_yr;   

Total_loss = sum(Loss);
NCal_loss = sum(Loss(287:312,:));
SCal_loss = sum(Loss(1:287,:));

Initial_capital =1.5*10^6;
Capital(1) = Initial_capital + Premium - Total_loss(1);

for i=2:yrs
    Capital(i) = Capital(i-1) + Premium - Total_loss(i);
end

% Finding the value at risk at p = 0.75, 0.9, 0.95 and 0.99
total_loss_sort = sort(Total_loss);
SCal_loss_sort = sort(SCal_loss);
NCal_loss_sort = sort(NCal_loss);
F_loss = (1:yrs)/yrs;

% Plotting PDF, CDF and Capital vs time graph
figure;
plot(M',Nm');
grid on;
xlabel('Magnitude (Mw)');
ylabel('Rate of Occurance (M>m)')
title('Decreasing exponential for rate of occurance');

figure;
stairs(total_loss_sort/10^6,F_loss);
grid on;
xlabel('Loss($ Millions)');
ylabel('F(s)');
title('CDF of Loss per year');

figure;
hist(Total_loss,30);
grid on;
xlabel('Loss($)');
ylabel('Frequency');
title('Distribution of Loss in a year');

figure;
plot(1:yrs,Capital);
hold on
plot(1:yrs,zeros(1,yrs),'k');
grid on
xlabel('years');
ylabel('Capital($)');
ruin_p = length(Capital(Capital<0))/yrs;
title({'Variation of Companys Capital over the years';strcat('Ruin Probability =',num2str(ruin_p))});



% Value at Risk, TVAR and AAL Calculations;
p = [0.5 0.75 0.9 0.95 0.99];
for i=1:length(p)
    [c index] = min(abs(F_loss-p(i)));
    if p(i) > F_loss(index)
                VAR(i) = total_loss_sort(index); 
                VAR_SC(i) = SCal_loss_sort(index);
                VAR_NC(i) = NCal_loss_sort(index);
    else
                VAR(i) = total_loss_sort(index-1); 
                VAR_SC(i) = SCal_loss_sort(index-1);
                VAR_NC(i) = NCal_loss_sort(index-1);
    end
end

A = cumsum(VAR(length(p):-1:1));
A_SC = cumsum(VAR_SC(length(p):-1:1));
A_NC = cumsum(VAR_NC(length(p):-1:1));
A = A(length(p):-1:1);
A_SC = A_SC(length(p):-1:1);
A_NC = A_NC(length(p):-1:1);
TVAR = A./(length(p):-1:1);
TVAR_SC = A_SC./(length(p):-1:1);
TVAR_NC = A_NC./(length(p):-1:1);

AAL = mean(Total_loss);
AAL_SC = mean(SCal_loss);
AAL_NC = mean(NCal_loss);

% Calculating Minumum Capital Required and expected Premium.

dividend = 0.02;
sigma_loss = std(Total_loss);
sigma_loss_SC = std(SCal_loss);
sigma_loss_NC = std(NCal_loss);
U = sigma_loss*sqrt(-log(ruin_p)/(2*dividend));
U_SC = sigma_loss_SC*sqrt(-log(ruin_p)/(2*dividend));
U_NC = sigma_loss_NC*sqrt(-log(ruin_p)/(2*dividend));
E_prem = (AAL + sigma_loss*sqrt(-2*dividend*log(ruin_p)))/312;
E_prem_SC = (AAL_SC + sigma_loss_SC*sqrt(-2*dividend*log(ruin_p)))/287;
E_prem_NC = (AAL_NC + sigma_loss_NC*sqrt(-2*dividend*log(ruin_p)))/26;

% Calculation of Expected Premium per location
for i = 1: loc_no
    
    L = Loss(i,:);
    mu_L_loc = mean(L);
    sigma_L_loc = std(L);
    E_prem_loc(i) =  mu_L_loc + R*(sigma_L_loc^2);
end

toc;



























