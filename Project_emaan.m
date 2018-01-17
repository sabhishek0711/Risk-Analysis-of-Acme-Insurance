function Project()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    clc;
    close all;
    
    faults = loadFaults();
    sites = loadLocations();
    
    % Simulation constants
    Value = 10^6;           % !! NOTE !! Must also change in getRandLoss !!
    DSouthCal = 0.075*Value;% !! NOTE !! Must also change in getRandLoss !!
    DNorthCal = 0.18*Value; % !! NOTE !! Must also change in getRandLoss !!
    premium = 2/1000;
    premiumTot = length(sites)*0.002*(Value*(1-(286/312)*DSouthCal-(26/312)*DSouthCal));
    
    % Plot various test figures
    % plotTestFigures(faults,sites);
    plotCapitalOptimization(faults,sites,premiumTot)

    % Select number of years to analyze
    Years = 50000;
    
    % Simulate the annual portfolio performance N times
    % @losses: A vector of losses (One for each simulation)
    [losses, sitelosses] = simulatePortfolio(faults, sites, Years);
    
    % Calculate VAR, TVAR for losses
    printVAR(losses);
    
    % Calculate the annual probability of ruin
    net = premium + initialCapital - losses;
    AAR = sum(net<0)/length(net);
    fprintf('Annual probability of ruin %.5f',AAR);
    
    % Calculate mean losses
    AAL = mean(losses);
    fprintf('Annual Average Loss %.1f',AAL);
    
    SiteSum = sum(sitelosses,2)/Years;
    
    figure(100)
    bar(SiteSum)
    title('AAL per Site')
    xlabel('Site #')
    ylabel('Loss ($)')
    
    % Plot CDF for annual losses
    %plotLossCDF(losses)
    %title('Annual Loss CDF');
    
    % Plot CDF of net profit;
    %plotLossCDF(net);
    %title('Net profit CDF');
    
    % Plot predicted protfolio cashflow
    % plotPortfolioCashflow(losses,premium,initialCapital);
    % title('Predicted portfolio cashflow');
    
    % Save all figures to the images directory
    saveAllFigures('images');
end


function plotLossCDF(losses)
    % Plot the severity CDF
    % @simulations: The cell array of data to plot
    losses = sort(losses);
    probability = (1:length(losses))/length(losses);
    figure();
    plot(losses,probability);
    xlabel('Annual Loss, S');
    ylabel('P(S)');
end


function plotCapitalOptimization(faults,sites,premium)
    % Plot the required AAP, given the initial capital.
    
    capital = (1:3:15)*10^6;
    APR = zeros(1,length(capital));
    
    parfor i=1:length(capital)
        nyears = capital(i)/2000; % More simulations for smaller ARP
        [losses,~] = simulatePortfolio(faults, sites, nyears);
        net = premium + capital(i) - losses - 0.03*capital(i);
        APR(i) = sum(net<0)/length(net);
    end
    
    figure();
    plot(capital,APR);
    title('Annual Probability of Ruin');
    xlabel('Inital Capital ($)');
    ylabel('Annual Probability of Ruin');
end





function plotPortfolioCashflow(losses,premium,initialCapital)
    % Plot anticipated cashflow over nyears
    nyears = length(losses);
    net = zeros(1,nyears);
    net(1) = initialCapital;
    
    for i=1:nyears
        fprintf('\n------ Year %i -----\n',i)
        net(i+1) = net(i) + premium - losses(i);
    end
    
    % Add a year 0 (with initial capital)
    time = 0:nyears;
    
    % Plot the gross company value over nyears
    figure(); hold on;
    plot(time,net/10^6,'b');
    title('Net profit to date');
    xlabel('time (years)' );
    ylabel('money ($million)');
end



function [losses,sitelosses] = simulatePortfolio(faults, sites, nyears)
    % Simulate the annual portfolio performance N times
    % @losses: A vector of losses (One for each simulation)
    losses = zeros(1,nyears);
    sitelosses = zeros(length(sites),nyears);
    parfor i = 1:nyears
        [losses(i),sitelosses(:,i)] = simulateYear(faults,sites);
    end
end


function [yearlyLoss, siteLoss] = simulateYear(faults,sites)
    % Simulates a single year of earthquake losses
    % Return the yearlyLoss as a single dollar value
    yearlyLoss = 0;
    siteLoss = zeros(1,length(sites));

    for i=1:length(faults)
        fault = faults{i};
        fault.a;
        
        magnitudes = linspace(5,fault.mmax,10);
        rates = getRate(fault.a,magnitudes);
        
        % Generate enough poisson variables to last for this year
        % We would normally do this inside the loop, but it is
        % faster to use vectorization
        N = poissrnd(rates);
        
        % For each magnitude
        for m = 1:length(magnitudes)
            magnitude = magnitudes(m);
            nevents = N(m);
            
            % For each event (earthquake)
            for n=1:nevents
                % X is the total damages across all sites, for the given
                % event, of magnitude M occuring on the given fault.
                losses = getEventLoss(fault,magnitude,sites);
                %fprintf('Magnitude %.2f, losses, %.1f\n',magnitude,losses);
                yearlyLoss = yearlyLoss + sum(losses);
                siteLoss = siteLoss + losses;
            end
        end
    end
end
   


function loss = getEventLoss(fault,magnitude,sites)
    % Return the total loss (as a single value) caused by an event on
    % the fault, with the mangitude specified as well as site specific loss
    
    % Randomly place the rupture, and calculate the rupture-site distances
    distances = getRandomDistance(fault,magnitude,sites);

    % Calculate the SA and SA+sigma at each site
    % SA will be a vector containing the acceleration at each site
    [SA,SAmuSig] = getSA(magnitude,distances);
    
    % Randomly generate a loss at each site, given the spectral acceleration 
    rho = 0.2;
    loss = getRandLoss(SA, rho);
    %losses = getRandLoss(SAmuSig, rho)
end
        
            
  
function distances = getRandomDistance(fault,magnitude,sites)
    % Return a vector of site-source distances
    % Places the rupture at a random position and calculates the distance
    SRL = 10^(-3.55+0.74*magnitude);
    distances = zeros(length(sites),1);
    for i = 1:length(sites)
        site = sites{i};
        % Randomly place the rupture on the fault
        rupLocation = rand()*(fault.length-SRL);
        
        % Use similar triangles to calculate the lat,long position of the rupture
        % Assumes that lat,long behavior linearly (small angle approximation)
        ratioEndi = rupLocation/fault.length;
        ratioEndj = (rupLocation+SRL)/fault.length;

        EndiLat = (fault.lat2-fault.lat1)*ratioEndi+fault.lat1;
        EndjLat = (fault.lat2-fault.lat1)*ratioEndj+fault.lat1;
        EndiLong = (fault.long2-fault.long1)*ratioEndi+fault.long1;
        EndjLong = (fault.long2-fault.long1)*ratioEndj+fault.long1;
                
        % Gradient of the fault line, latitude
        lambda = (fault.lat2-fault.lat1)/(fault.long2-fault.long1);
        
        % Position of the perpendicular point on the fault
        X0 = (site.long/lambda+site.lat-fault.lat1+lambda*fault.long1)/(lambda+1/lambda);
        Y0 =-1/lambda*(X0-site.long)+site.lat;
        
        % if X0 is between EndiLong and EndjLong
        if (X0>min(EndiLong,EndjLong) && X0<max(EndiLong,EndjLong))
           %plot([site.long X0],[site.lat Y0],'r');
           distances(i) = lldistkm([site.lat,site.long],[Y0,X0]);  %km
        else
           dist1 = lldistkm([site.lat,site.long],[EndiLat,EndiLong]);
           dist2 = lldistkm([site.lat,site.long],[EndjLat,EndjLong]);
           distances(i) = min(dist1,dist2); %km
        end
        %figure(); hold on;
        %plot([fault.long1,fault.long2],[fault.lat1,fault.lat2],'r-o');
        %plot([EndiLong,EndjLong],[EndiLat,EndjLat],'g-o');
        %plot(X0,Y0,'*b',site.long,site.lat,'*g')
    end
end



function [SAmu,SAmuSig]=getSA(M,D)
    % Get spectral acceleration given magnitude and distance
    % @M is the magnitude of the rupture
    % @D can be a single distance or a vector of distances
    R = sqrt(D.^2 + 4.91^2);
    CC = -0.487*log(500/1954);
    X =(0.212 + 0.831*(M-6) - 0.12*(M-6).^2 - 0.867*log(R) + CC);
    SAmu = exp(X);
    SAmuSig = exp(X + 0.538);
end



function RandLoss = getRandLoss(SAmu,rho)
    % Return a random loss, given the mean [expected] spectral acceleration
    % @SAmu can be a vector or a single spectral acceleration
    % Losses will be correlated by the correlation factor rho
    
    V = 10^6;
    DSouthCal = 0.075*V;
    DNorthCal = 0.18*V;
    
    LossMu = V*normcdf(log(SAmu/3)/1.5, 0, 1);
    LossSig = min(LossMu.*(1./sqrt(LossMu/V)-1), 4*LossMu);
    stdLnX = sqrt(log(LossSig.^2 ./ LossMu.^2 + 1));
    meanLnX = log(LossMu)-1/2*stdLnX.^2;
    
    % RandLoss = lognrnd(meanLnX,stdLnX);
    % Generate random Y for correlated random variables
    Z = normrnd(0,1); 
    Y = normrnd(0,1,1,length(SAmu));
    RandLoss = zeros(1,length(SAmu));
    
    % Iterate through all of our Y values and find a corrosponding
    % lognormal loss
    for i=1:1:length(Y)
        X = sqrt(rho)*Z + sqrt(1-rho)*Y(i);
        rank = normcdf(X, 0, 1);
        loss = logninv(rank,meanLnX(i),stdLnX(i));
        % Limit loss to above deductable and below total value minus deductable
        
        if i<=286
            D = DSouthCal;
        else
            D = DNorthCal;
        end
                     
        loss = min(V-D, loss-D);
        loss = max(0, loss);

        % Spectral accelerations of zero give strange results
        if SAmu(i)==0
            loss=0;
        end
        RandLoss(i) = loss;
    end
    % We can vectorize the code above for efficiency
    X = sqrt(rho)*Z + sqrt(1-rho)*Y;
    rank = normcdf(X, 0, 1);
    RandLoss = logninv(rank,meanLnX',stdLnX');
    RandLoss = min(V-D, RandLoss-D);
    RandLoss = max(0, RandLoss);
end



function rates=getRate(FaultA,magnitudes)
    % Returns a vector of the rates of the range magnitudes in question.
    int = magnitudes(2)-magnitudes(1);
    magplus = magnitudes+int/2;
    magminus = magnitudes-int/2;
    rates = 10.^(FaultA-magminus)-10.^(FaultA-magplus);
end




function locations=loadLocations()
    % Load faults from the xlsx file
    % Locations are stored in a cell array
    % Locations have zip,lat,long fields
    filename = 'Spec/Exposure.xlsx';
    data = xlsread(filename);
    locations = cell(1,length(data));
    for i=1:length(locations)
        loc.zip = data(i,2);
        loc.lat = data(i,3);
        loc.long = data(i,4);
        locations{i} = loc; 
    end
end




function faults=loadFaults()
    % Load faults from the xlsx file
    % Faults are stored in a cell array
    % Faults have rate,mmax,long1,lat1,long2,lat2 fields
    filename = 'Spec/Fit Source Characterization.xlsx';
    data = xlsread(filename);
    faults = cell(1,length(data));
    for i=1:length(faults)
        fault.a = data(i,1);
        fault.mmax = data(i,2);
        fault.long1 = data(i,3);
        fault.lat1 = data(i,4);
        fault.long2 = data(i,5);
        fault.lat2 = data(i,6);
        % Calculate the surface length of the fault
        fault.length = lldistkm([fault.lat1,fault.long1], [fault.lat2,fault.long2]);
        faults{i} = fault; 
    end
end



function printVAR(losses)
    % Print a table TAVR and VAR values for a given N and X 
    fprintf('VAR(0.5)   VAR(0.75),  VAR(0.9), VAR(0.95),  VAR(0.99)');
    fprintf('  |  TVAR(0.5)  TVAR(0.75), TVAR(0.9), TVAR(0.9),  TVAR(0.99)\n');

    fprintf('%i  ',   length(losses));
    fprintf('%7.1f  ',prctile(losses,50));
    fprintf('%7.1f  ',prctile(losses,75));
    fprintf('%7.1f  ',prctile(losses,90));
    fprintf('%7.1f  ',prctile(losses,95));
    fprintf('%7.1f  ',prctile(losses,99));
    fprintf(' |  ');
    fprintf('%7.1f  ',tvar(losses,50));
    fprintf('%7.1f  ',tvar(losses,75));
    fprintf('%7.1f  ',tvar(losses,90));
    fprintf('%7.1f  ',tvar(losses,95));
    fprintf('%7.1f  ',tvar(losses,99));
    fprintf('\n');
end



function tvar = tvar(data,percentile)
    % Calculate the TVAR for a given percentile
    quantile = prctile(data,percentile);
    tailvalues = data(data>quantile);
    tvar = mean(tailvalues);
end
    


function saveAllFigures(dirname)
    % Save all figures to the images directory
    figlist=findobj('type','figure');
    for i=1:numel(figlist)
        %num2str(figlist(i))
        filename = fullfile(dirname,['figure', num2str(i), '.png']);
        saveas(figlist(i),filename);
    end
end



function [d1km, d2km]=lldistkm(latlon1,latlon2)
    % format: [d1km d2km]=lldistkm(latlon1,latlon2)
    % Distance:
    % d1km: distance in km based on Haversine formula
    % (Haversine: http://en.wikipedia.org/wiki/Haversine_formula)
    % d2km: distance in km based on Pythagoras’ theorem
    % (see: http://en.wikipedia.org/wiki/Pythagorean_theorem)
    % After:
    % http://www.movable-type.co.uk/scripts/latlong.html
    %
    % --Inputs:
    %   latlon1: latlon of origin point [lat lon]
    %   latlon2: latlon of destination point [lat lon]
    %
    % --Outputs:
    %   d1km: distance calculated by Haversine formula
    %   d2km: distance calculated based on Pythagoran theorem
    %
    % --Example 1, short distance:
    %   latlon1=[-43 172];
    %   latlon2=[-44  171];
    %   [d1km d2km]=distance(latlon1,latlon2)
    %   d1km =
    %           137.365669065197 (km)
    %   d2km =
    %           137.368179013869 (km)
    %   %d1km approximately equal to d2km
    %
    % --Example 2, longer distance:
    %   latlon1=[-43 172];
    %   latlon2=[20  -108];
    %   [d1km d2km]=distance(latlon1,latlon2)
    %   d1km =
    %           10734.8931427602 (km)
    %   d2km =
    %           31303.4535270825 (km)
    %   d1km is significantly different from d2km (d2km is not able to work
    %   for longer distances).
    %
    % First version: 15 Jan 2012
    % Updated: 17 June 2012
    %--------------------------------------------------------------------------

    radius=6371;
    lat1=latlon1(1)*pi/180;
    lat2=latlon2(1)*pi/180;
    lon1=latlon1(2)*pi/180;
    lon2=latlon2(2)*pi/180;
    deltaLat=lat2-lat1;
    deltaLon=lon2-lon1;
    a=sin((deltaLat)/2)^2 + cos(lat1)*cos(lat2) * sin(deltaLon/2)^2;
    c=2*atan2(sqrt(a),sqrt(1-a));
    d1km=radius*c;    %Haversine distance

    x=deltaLon*cos((lat1+lat2)/2);
    y=deltaLat;
    d2km=radius*sqrt(x*x + y*y); %Pythagoran distance
    drawnow();
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% All code below this point is for testing purposes only %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function plotTestFigures(faults,sites)
    % Plot the method figures
    plotRateCurve(faults);

    % Plot the location Map
    figure(); hold on;
    title('California Faults and Sites')
    for i=1:length(faults)
        f=faults{i};
        plot([f.lat1,f.lat2]',[f.long1,f.long2]');
    end
    for i=1:length(sites)
        site=sites{i};
        scatter(site.lat, site.long)
    end
    
    
    % Plot mean loss given SA
    figure();
    SA = linspace(0,15,100);
    meanLosses = zeros(1,length(SA));
    for i=1:length(SA)
        accelerations = SA(i)*ones(1000,1);
        meanLosses(i) = mean(getRandLoss(accelerations,0.2));
    end
    plot(SA, meanLosses);
    title('Mean loss given SA');
    xlabel('Spectral Acceleration');
    ylabel('Expected Loss ($)');
    
    
    
    % Plot distribution of losses given different SA values
    figure(); hold on;
    SA = [1,2,5,10,15];
    for SAmu=SA
       accelerations = SAmu*ones(1,5*10^5)';
       losses = getRandLoss(accelerations,0.2);
       buckets = linspace(0,10^6,100);
       h = histc(losses, buckets);
       plot(buckets,h)
       axis([-inf inf 0 10^5]);
    end
    legend({'SA=1','SA=2','SA=5','SA=10','SA=15'});
    title('Net loss for different spectral accelerations');
    xlabel('Loss ($)');
    ylabel('Frequency');
       
     
    
    % plot spectral accleration, for different magnitudes
    figure(); hold on;
    for magnitude = [5,6,7,8]
        distances = linspace(0,50,100);
        SA = getSA(magnitude,distances);
        plot(distances,SA);
    end
    title('Spectral Acceleration given distance')
    xlabel('Distance from rupture');
    ylabel('Spectral Acceleration');
    legend({'Magnitude 5','Magnitude 6','Magnitude 7','Magnitude 8'});
    drawnow();
end



function plotRateCurve(faults)
    % Plot Gutenberg-Richter curve for the first fault in faults
    figure(); hold on;
    for i=[1,2,7,15]
        fault = faults{i};
        magnitudes = linspace(5,fault.mmax,10);
        rates = 10.^(fault.a - 1*magnitudes);
        plot([magnitudes magnitudes(end)] ,[rates 0]);
    end
    
    for i=[1,2,7,15]
        fault = faults{i};
        magnitudes = linspace(5,fault.mmax,10);
        rates = getRate(fault.a,magnitudes);
        plot([magnitudes magnitudes(end)] ,[rates 0]);
    end
    
    xlabel('Magnitude');
    ylabel('Frequency (M<m)');
    legend({'Hayward','Calaveras South','San Andreas','San Jose'});
end

