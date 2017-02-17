function Allcodes

% Closed-form solutions: CC, PC, CP, PP

function price = Calloncall(S0,X1,X2,T1,T2,r,sigma)
    I=fsolve(@(x) blsprice(x,X2,r,T2-T1,sigma)-X1,S0);
    y1=(log(S0./I)+(r+sigma.^2./2).*T1)./(sigma.*sqrt(T1));
    y2=y1-sigma.*sqrt(T1);
    z1=(log(S0./X2)+(r+sigma.^2./2).*T2)./(sigma.*sqrt(T2));
    z2=z1-sigma.*sqrt(T2);
    rho=sqrt(T1./T2);
        price=S0.*mvncdf([z1,y1],[0,0],[1,rho;rho,1])-X2.*exp(-r.*T2).*...
        mvncdf([z2,y2],[0,0],[1,rho;rho,1])-X1.*exp(-r.*T1).*normcdf(y2,0,1);
end

function price = Putoncall(S0,X1,X2,T1,T2,r,sigma)
    I=fsolve(@(x) blsprice(x,X2,r,T2-T1,sigma)-X1,S0);
    y1=(log(S0./I)+(r+sigma.^2./2).*T1)./(sigma.*sqrt(T1));
    y2=y1-sigma.*sqrt(T1);
    z1=(log(S0./X2)+(r+sigma.^2./2).*T2)./(sigma.*sqrt(T2));
    z2=z1-sigma.*sqrt(T2);
    rho=sqrt(T1./T2);
        price=-S0.*mvncdf([z1,-y1],[0,0],[1,-rho;-rho,1])+X2.*exp(-r.*T2).*...
        mvncdf([z2,-y2],[0,0],[1,-rho;-rho,1])+X1.*exp(-r.*T1).*normcdf(-y2,0,1);
end

function price = Callonput(S0,X1,X2,T1,T2,r,sigma)
    I=fsolve(@(x) blsprice(x,X2,r,T2-T1,sigma)+X2.*exp(-r.*(T2-T1))-x-X1,S0);
    y1=(log(S0./I)+(r+sigma.^2./2).*T1)./(sigma.*sqrt(T1));
    y2=y1-sigma.*sqrt(T1);
    z1=(log(S0./X2)+(r+sigma.^2./2).*T2)./(sigma.*sqrt(T2));
    z2=z1-sigma.*sqrt(T2);
    rho=sqrt(T1./T2);
        price=-S0.*mvncdf([-z1,-y1],[0,0],[1,rho;rho,1])+X2.*exp(-r.*T2).*...
        mvncdf([-z2,-y2],[0,0],[1,rho;rho,1])-X1.*exp(-r.*T1).*normcdf(-y2,0,1);
end


function price = Putonput(S0,X1,X2,T1,T2,r,sigma)
    I=fsolve(@(x) blsprice(x,X2,r,T2-T1,sigma)+X2.*exp(-r.*(T2-T1))-x-X1,S0);
    y1=(log(S0./I)+(r+sigma.^2./2).*T1)./(sigma.*sqrt(T1));
    y2=y1-sigma.*sqrt(T1);
    z1=(log(S0./X2)+(r+sigma.^2./2).*T2)./(sigma.*sqrt(T2));
    z2=z1-sigma.*sqrt(T2);
    rho=sqrt(T1./T2);
        price=S0.*mvncdf([-z1,y1],[0,0],[1,-rho;-rho,1])-X2.*exp(-r.*T2).*...
        mvncdf([-z2,y2],[0,0],[1,-rho;-rho,1])+X1.*exp(-r.*T1).*normcdf(y2,0,1);
end

% Calculation of Closed_form

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma = 0.15; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma);
CC = Calloncall(S0, Call, X2, T1, T2, r, sigma)
PC = Putoncall(S0, Call, X2, T1, T2, r, sigma)
PP = Putonput(S0, Put, X2, T1, T2, r, sigma)
CP = Callonput(S0, Put, X2, T1, T2, r, sigma)



% Simple Monte Carlo: CC & PC, CP & PP

function [Callc, Putc, CI , elapsedTime, cc] = CCMC ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, seed)
tic;
rng(seed);
nu = r - sigma1 .^ 2 ./ 2; nuT1 = nu .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1);
ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, N));

[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
cc = exp(-r .* T1) .* max(c - X1, 0);
pc = exp(-r .* T1) .* max(X1 - c, 0);
Callc = mean(cc);
Putc = mean(pc);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(cc);
CI = MUCI;
elapsedTime = toc;
end

% Calculation of Simple MC: CC & PC, Client 5, 1, 2, 3, 4

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, Putc, CI , elapsedTime, cc] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, Putc, CI , elapsedTime] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, Putc, CI , elapsedTime, cc] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, Putc, CI , elapsedTime] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, Putc, CI , elapsedTime, cc] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, Putc, CI , elapsedTime] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, Putc, CI , elapsedTime, cc] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, Putc, CI , elapsedTime] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, Putc, CI , elapsedTime, cc] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, Putc, CI , elapsedTime] = CCMC(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)


function [Callp, Putp, CI , elapsedTime, cp] = CPMC ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, seed)
tic;
rng(seed);
nu = r - sigma1 .^ 2 ./ 2; nuT1 = nu .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1);
ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1,N));

[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
cp = exp(-r.*T1).* max(p - X1, 0);
pp = exp(-r.*T1).* max(X1 - p, 0);
Callp = mean(cp);
Putp = mean(pp);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(cp);
CI = MUCI;
elapsedTime = toc;
end

% Calculation of Simple MC: CP & PP, Client 5, 1, 2, 3, 4

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, Putp, CI , elapsedTime, cp] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, Putp, CI , elapsedTime] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, Putp, CI , elapsedTime, cp] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, Putp, CI , elapsedTime] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, Putp, CI , elapsedTime, cp] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, Putp, CI , elapsedTime] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, Putp, CI , elapsedTime, cp] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, Putp, CI , elapsedTime] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, Putp, CI , elapsedTime, cp] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor (std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, Putp, CI , elapsedTime] = CPMC(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)


% Smart Lattice

function price = CCLattice (S0, X1, X2, r, T1, T2, sigma1, sigma2, N)
deltaT = T1 ./ N; 
u = exp(sigma1 .* sqrt(deltaT)); d =1 ./ u;
p = (exp(r .* deltaT) - d) ./ (u - d);
discount = exp(-r .* deltaT);
p_u = discount .* p; p_d = discount .* (1-p);

SVals = zeros(2 .* N + 1, 1);
SVals(1) = S0 .* d .^ N;
CVals(1) = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i = 2:2*N+1
    SVals(i) = u .* SVals(i-1);
    CVals(i) = blsprice(SVals(i), X2, r, T2-T1, sigma2);
end

CCVals = zeros(2 .* N + 1, 1);
for i = 1:2:2*N+1 
    CCVals(i) = max(CVals(i) - X1, 0);
end

for tau = 1:N 
    for i = (tau+1):2:(2*N+1-tau) 
        CCVals(i) = p_u .* CCVals(i+1) + p_d .* CCVals(i-1);
    end
end
price = CCVals(N+1);
end

function price = PCLattice (S0, X1, X2, r, T1, T2, sigma1, sigma2, N)
deltaT = T1 ./ N;
u = exp(sigma1 .* sqrt(deltaT)); d =1 ./ u;
p = (exp(r .* deltaT) - d) ./ (u - d);
discount = exp(-r .* deltaT);
p_u = discount .* p; p_d = discount .* (1-p);

SVals = zeros(2 .* N + 1, 1);
SVals(1) = S0 .* d .^ N;
CVals(1) = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i = 2:2*N+1
    SVals(i) = u .* SVals(i-1);
    CVals(i) = blsprice(SVals(i), X2, r, T2-T1, sigma2);
end

PCVals = zeros(2 .* N + 1, 1);
for i = 1:2:2*N+1 
    PCVals(i) = max(X1 - CVals(i), 0);
end

for tau = 1:N 
    for i = (tau+1):2:(2*N+1-tau) 
        PCVals(i) = p_u .* PCVals(i+1) + p_d .* PCVals(i-1);
    end
end
price = PCVals(N+1);
end

function price = CPLattice (S0, X1, X2, r, T1, T2, sigma1, sigma2, N)
deltaT = T1 ./ N;
u = exp(sigma1 .* sqrt(deltaT)); d =1 ./ u;
p = (exp(r .* deltaT) - d) ./ (u - d);
discount = exp(-r .* deltaT);
p_u = discount .* p;
p_d = discount .* (1-p);
 
SVals = zeros(2 .* N + 1, 1);
SVals(1) = S0 .* d .^ N;
[CVals(1), PVals(1)] = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i = 2:2*N+1
    SVals(i) = u .* SVals(i-1);
    [CVals(i), PVals(i)] = blsprice(SVals(i), X2, r, T2 - T1, sigma2);
    exp(-r .* (T2-T1));
end
 
CPVals = zeros(2 .* N + 1, 1);
for i = 1:2:2*N+1 
    CPVals(i) = max(PVals(i) - X1, 0);
end

for tau = 1:N 
    for i = (tau+1):2:(2*N+1-tau) 
        CPVals(i) = p_u .* CPVals(i+1) + p_d .* CPVals(i-1);
    end
end
price = CPVals(N+1);
end

function price = PPLattice(S0, X1, X2, r, T1, T2, sigma1, sigma2, N)
deltaT = T1 ./ N;
u = exp(sigma1 .* sqrt(deltaT)); d =1 ./ u;
p = (exp(r .* deltaT) - d) ./ (u - d);
discount = exp(-r .* deltaT);
p_u = discount .* p; p_d = discount .* (1-p);

SVals = zeros(2 .* N + 1, 1);
SVals(1) = S0 .* d .^ N;
[CVals(1), PVals(1)] = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i = 2:2*N+1
    SVals(i) = u .* SVals(i-1);
    [CVals(i), PVals(i)] = blsprice(SVals(i), X2, r, T2 - T1, sigma2);
    exp(-r .* (T2-T1));
end

PPVals = zeros(2 .* N + 1, 1);
for i = 1:2:2*N+1 
    PPVals(i) = max(X1 - PVals(i), 0);
end

for tau = 1:N 
    for i = (tau+1):2:(2*N+1-tau) 
        PPVals(i) = p_u .* PPVals(i+1) + p_d .* PPVals(i-1);
    end
end
price = PPVals(N+1);
end

% Smart Lattice Calculation, Client 5, 1, 2, 3, 4

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365;
CCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
PCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
CPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)
PPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365;
CCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
PCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
CPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)
PPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365;
CCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
PCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
CPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)
PPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365;
CCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
PCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
CPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)
PPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365;
CCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
PCLattice(S0, Call, X2, r, T1, T2, sigma1, sigma2, N)
CPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)
PPLattice(S0, Put, X2, r, T1, T2, sigma1, sigma2, N)


% Implicit FDM

function price = CCImpl(S0, X1, X2, r, T1, T2, sigma1, sigma2, Smax, dS, dt)
M=round(Smax/dS); dS=Smax/M; N=round(T1/dt); dt=T1/N;
matval=zeros(M+1,N+1);
vetS=linspace(0,Smax,M+1)'; veti=0:M; vetj=0:N;

vetC=blsprice(vetS, X2, r, T2-T1, sigma2); matval(:,N+1)=max(vetC-X1,0);
matval(1,:)=0; matval(M+1,:)=(blsprice(Smax, X2, r, T2-T1, sigma2) - X1).*exp(-r*dt*(N-vetj));

a=0.5*(r*dt*veti-sigma1^2*dt*veti.^2);
b=1+sigma1^2*dt*(veti.^2)+r*dt;
c=-0.5*(r*dt*veti+sigma1^2*dt*veti.^2);
coeff=diag(a(3:M),-1)+diag(b(2:M))+diag(c(2:M-1),+1);
[L U]=lu(coeff);

aux=zeros(M-1,1);
for j=N:-1:1
aux(1)=-a(2)*matval(1,j); 
aux(M-1) = - c(M) * matval(M+1,j);
    matval(2:M,j)=U\(L\(matval(2:M,j+1)+aux));
end

price=interp1(vetS,matval(:,1),S0);
end

function price = PCImpl(S0, X1, X2, r, T1, T2, sigma1, sigma2, Smax, dS, dt)
M=round(Smax/dS); dS=Smax/M; N=round(T1/dt); dt=T1/N;
matval=zeros(M+1,N+1);
vetS=linspace(0,Smax,M+1)'; veti=0:M; vetj=0:N;

vetC=blsprice(vetS, X2, r, T2-T1, sigma2); matval(:,N+1)=max(X1-vetC,0);
matval(1,:)=X1*exp(-r*dt*(N-vetj)); matval(M+1,:)=0;

a=0.5*(r*dt*veti-sigma1^2*dt*veti.^2);
b=1+sigma1^2*dt*(veti.^2)+r*dt;
c=-0.5*(r*dt*veti+sigma1^2*dt*veti.^2);
coeff=diag(a(3:M),-1)+diag(b(2:M))+diag(c(2:M-1),+1);
[L U]=lu(coeff);

aux=zeros(M-1,1);
for j=N:-1:1
aux(1)=-a(2)*matval(1,j); 
aux(M-1) = - c(M) * matval(M+1,j);
    matval(2:M,j)=U\(L\(matval(2:M,j+1)+aux));
end

price=interp1(vetS,matval(:,1),S0);
end

function price = CPImpl(S0, X1, X2, r, T1, T2, sigma1, sigma2, Smax, dS, dt)
M=round(Smax/dS); dS=Smax/M; N=round(T1/dt); dt=T1/N;
matval=zeros(M+1,N+1);
vetS=linspace(0,Smax,M+1)'; veti=0:M; vetj=0:N;

[C, vetP]=blsprice(vetS, X2, r, T2-T1, sigma2); 
matval(:,N+1)=max(vetP-X1,0);
[Cm, vetPm] = blsprice(Smax, X2, r, T2-T1, sigma2);
matval(1,:)=0; matval(M+1,:)=(vetPm - X1).*exp(-r*dt*(N-vetj));

a=0.5*(r*dt*veti-sigma1^2*dt*veti.^2);
b=1+sigma1^2*dt*(veti.^2)+r*dt;
c=-0.5*(r*dt*veti+sigma1^2*dt*veti.^2);
coeff=diag(a(3:M),-1)+diag(b(2:M))+diag(c(2:M-1),+1);
[L U]=lu(coeff);

aux=zeros(M-1,1);
for j=N:-1:1
aux(1)=-a(2)*matval(1,j); 
aux(M-1) = - c(M) * matval(M+1,j);
    matval(2:M,j)=U\(L\(matval(2:M,j+1)+aux));
end

price=interp1(vetS,matval(:,1),S0);
end

function price = PPImpl(S0, X1, X2, r, T1, T2, sigma1, sigma2, Smax, dS, dt)
M=round(Smax/dS); dS=Smax/M; N=round(T1/dt); dt=T1/N;
matval=zeros(M+1,N+1);
vetS=linspace(0,Smax,M+1)'; veti=0:M; vetj=0:N;

[C, vetP]=blsprice(vetS, X2, r, T2-T1, sigma2); 
matval(:,N+1)=max(X1-vetP,0);
matval(1,:)=X1*exp(-r*dt*(N-vetj)); matval(M+1,:)=0;

a=0.5*(r*dt*veti-sigma1^2*dt*veti.^2);
b=1+sigma1^2*dt*(veti.^2)+r*dt;
c=-0.5*(r*dt*veti+sigma1^2*dt*veti.^2);
coeff=diag(a(3:M),-1)+diag(b(2:M))+diag(c(2:M-1),+1);
[L U]=lu(coeff);

aux=zeros(M-1,1);
for j=N:-1:1
aux(1)=-a(2)*matval(1,j); 
aux(M-1) = - c(M) * matval(M+1,j);
    matval(2:M,j)=U\(L\(matval(2:M,j+1)+aux));
end

price=interp1(vetS,matval(:,1),S0);
end

% FDM Calculation: Client 5, 1, 2, 3, 4

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2);
CCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
CPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2);
CCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
CPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2);
CCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
CPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2);
CCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
CPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2);
CCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PCImpl(S0, Call, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
CPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)
PPImpl(S0, Put, X2, r, T1, T2, sigma1, sigma2, 200, 0.5, 1/365)


% Antithetic Variables: CC, PC, CP, PP

function [Callc, CI , elapsedTime, cc] = CCMCanti ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, seed)
tic;    rng(seed);
nu = r - sigma1 .^ 2 ./ 2; nuT1 = nu .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1); 
a1 = randn(1, N);   ST11 = S0 .* exp(nuT1 + sigmasqrtT1 .* a1); 
ST12 = S0 .* exp(nuT1 + sigmasqrtT1 .* (-a1));

c1 = blsprice(ST11, X2, r, T2-T1, sigma2); 
c2 = blsprice(ST12, X2, r, T2-T1, sigma2);
cc1 = exp(-r .* T1) .* max(c1 - X1, 0); 
cc2 = exp(-r .* T1) .* max(c2 - X1, 0);
cc = (cc1 + cc2) ./ 2; Callc = mean(cc); 

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(cc);
CI = MUCI;
elapsedTime = toc;
end

function [Putc, CI , elapsedTime, pc] = PCMCanti ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, seed)
tic;    rng(seed);
nu = r - sigma1 .^ 2 ./ 2; nuT1 = nu .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1); 
a1 = randn(1, N);   ST11 = S0 .* exp(nuT1 + sigmasqrtT1 .* a1); 
ST12 = S0 .* exp(nuT1 + sigmasqrtT1 .* (-a1));

c1 = blsprice(ST11, X2, r, T2-T1, sigma2); 
c2 = blsprice(ST12, X2, r, T2-T1, sigma2);
pc1 = exp(-r .* T1) .* max(X1 - c1, 0); 
pc2 = exp(-r .* T1) .* max(X1 - c2, 0);
pc = (pc1 + pc2) ./ 2; Putc = mean(pc);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(pc);
CI = MUCI;
elapsedTime = toc;
end

function [Callp, CI , elapsedTime, cp] = CPMCanti ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, seed)
tic;
rng(seed);
nu = r - sigma1 .^ 2 ./ 2; nuT1 = nu .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1); a1 = randn(1, N);
ST11 = S0 .* exp(nuT1 + sigmasqrtT1 .* a1); ST12 = S0 .* exp(nuT1 + sigmasqrtT1 .* (-a1));

p1 = blsprice(ST11, X2, r, T2-T1, sigma2) + X2 .* exp(-r.*(T2-T1)) - ST11;
p2 = blsprice(ST12, X2, r, T2-T1, sigma2) + X2 .* exp(-r.*(T2-T1)) - ST12;
cp1 = exp(-r.*T1).* max(p1 - X1, 0); cp2 = exp(-r.*T1).* max(p2 - X1, 0);
cp = (cp1 + cp2) ./ 2; Callp = mean(cp); 

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(cp);
CI = MUCI;
elapsedTime = toc;
end

function [Putp, CI , elapsedTime, pp] = PPMCanti ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, seed)
tic;
rng(seed);
nu = r - sigma1 .^ 2 ./ 2; nuT1 = nu .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1); a1 = randn(1, N);
ST11 = S0 .* exp(nuT1 + sigmasqrtT1 .* a1); ST12 = S0 .* exp(nuT1 + sigmasqrtT1 .* (-a1));

p1 = blsprice(ST11, X2, r, T2-T1, sigma2) + X2 .* exp(-r.*(T2-T1)) - ST11;
p2 = blsprice(ST12, X2, r, T2-T1, sigma2) + X2 .* exp(-r.*(T2-T1)) - ST12;
pp1 = exp(-r.*T1).* max(X1 - p1, 0); pp2 = exp(-r.*T1).* max(X1 - p2, 0);
pp = (pp1 + pp2) ./ 2; Putp = mean(pp);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(pp);
CI = MUCI;
elapsedTime = toc;
end


% Antithetic Variables Calculation: CC, PC, CP, PP; Client 5, 1, 2, 3, 4
% CC
S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N = 10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

% PC

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCanti ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCanti(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, seed)

% CP

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, CI , elapsedTime, cp] = CPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, CI , elapsedTime, cp] = CPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, CI , elapsedTime, cp] = CPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, CI , elapsedTime, cp] = CPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Callp, CI , elapsedTime, cp] = CPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

% PP

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putp, CI , elapsedTime, pp] = PPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putp, CI , elapsedTime, pp] = PPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putp, CI , elapsedTime, pp] = PPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putp, CI , elapsedTime, pp] = PPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; seed = 777;
[Putp, CI , elapsedTime, pp] = PPMCanti ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCanti(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, seed)


% Control Variates: CC, PC, CP, PP

function [Callc, CI , elapsedTime, cc] = CCMCCV...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)
tic;
rng(seed);
nu1 = r - sigma1 .^ 2 ./ 2; nuT1 = nu1 .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1);
ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, NTrials));

Euc = max(ST1-X2, 0);
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
cc = exp(-r .* T1) .* max(c - X1, 0); 
VarCov1 = cov(Euc, cc); c_coeff1= -VarCov1(1,2)./var(Euc);

ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, N));
Euc = max(ST1-X2, 0); EEuc = exp(r .* T1) .* blsprice(S0, X2, r, T1, sigma1);
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
cc = exp(-r .* T1) .* max(c - X1, 0) + c_coeff1 .* (Euc - EEuc);
Callc = mean(cc);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(cc);
CI = MUCI;
elapsedTime = toc;
end

function [Putc, CI , elapsedTime, pc] = PCMCCV...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)
tic;
rng(seed);
nu1 = r - sigma1 .^ 2 ./ 2; nuT1 = nu1 .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1);
ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, NTrials));

Euc = max(ST1-X2, 0);
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
pc = exp(-r .* T1) .* max(X1 - c, 0);
VarCov2 = cov(Euc, pc); c_coeff2= -VarCov2(1,2)./var(Euc);

ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, N));
Euc = max(ST1-X2, 0); EEuc = exp(r .* T1) .* blsprice(S0, X2, r, T1, sigma1);
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
pc = exp(-r .* T1) .* max(X1 - c, 0) + c_coeff2 .* (Euc - EEuc);
Putc = mean(pc);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(pc);
CI = MUCI;
elapsedTime = toc;
end

function [Callp, CI , elapsedTime, cp] = CPMCCV ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)
tic;
rng(seed);
nu1 = r - sigma1 .^ 2 ./ 2; nuT1 = nu1 .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1);
ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, NTrials));

Eup = max(X2-ST1, 0);
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
cp = exp(-r .* T1).* max(p - X1, 0); 
VarCov1 = cov(Eup, cp); c_coeff1 = -VarCov1(1,2) ./ var(Eup);

ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, N));
Eup = max(X2-ST1, 0); [c2, p2] = blsprice(S0, X2, r, T1, sigma1);
EEup = exp(r .* T1) .* p2;
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
cp = exp(-r .* T1).* max(p - X1, 0) + c_coeff1 .* (Eup - EEup);
Callp = mean(cp); 

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(cp);
CI = MUCI;
elapsedTime = toc;
end

function [Putp, CI , elapsedTime, pp] = PPMCCV ...
    (S0, X1, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)
tic;
rng(seed);
nu1 = r - sigma1 .^ 2 ./ 2; nuT1 = nu1 .* T1; sigmasqrtT1 = sigma1 .* sqrt(T1);
ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, NTrials));

Eup = max(X2-ST1, 0);
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
pp = exp(-r .* T1).* max(X1 - p, 0);
VarCov2 = cov(Eup, pp); c_coeff2 = -VarCov2(1,2) ./ var(Eup);

ST1 = S0 .* exp(nuT1 + sigmasqrtT1 .* randn(1, N));
Eup = max(X2-ST1, 0); [c2, p2] = blsprice(S0, X2, r, T1, sigma1);
EEup = exp(r .* T1) .* p2;
[c, p] = blsprice(ST1, X2, r, T2-T1, sigma2);
pp = exp(-r .* T1).* max(X1 - p, 0) + c_coeff2 .* (Eup - EEup);
Putp = mean(pp);

[MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(pp);
CI = MUCI;
elapsedTime = toc;
end


% Control Variates Calculation: CC, PC, CP, PP; Client 5, 1, 2, 3, 4
% CC
S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Callc, CI , elapsedTime, cc] = CCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cc) .^ 2))
[Callc, CI , elapsedTime] = CCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

% PC

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
[Putc, CI , elapsedTime, pc] = PCMCCV ...
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pc) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pc) .^ 2))
[Putc, CI , elapsedTime] = PCMCCV... 
(S0, Call, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

% CP

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Callp, CI , elapsedTime, cp] = CPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Callp, CI , elapsedTime, cp] = CPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Callp, CI , elapsedTime, cp] = CPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Callp, CI , elapsedTime, cp] = CPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Callp, CI , elapsedTime, cp] = CPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(cp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(cp) .^ 2))
[Callp, CI , elapsedTime] = CPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

% PP

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Putp, CI , elapsedTime, pp] = PPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Putp, CI , elapsedTime, pp] = PPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Putp, CI , elapsedTime, pp] = PPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Putp, CI , elapsedTime, pp] = PPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =10000; NTrials = 10000; seed = 777;
 [Putp, CI , elapsedTime, pp] = PPMCCV ...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed);
N = floor(std(pp) .^ 2 .* 1.96 .^ 2 .* 1002001 ./ (mean(pp) .^ 2))
[Putp, CI , elapsedTime] = PPMCCV...
(S0, Put, X2, T1, T2, r, sigma1, sigma2, N, NTrials, seed)


% Trinomial Lattice: CC, PC, CP, PP

function price = CCTrinomial(S0, X1, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
deltaT = T1 ./ N; nu = r - 0.5 .* sigma1 .^ 2; discount = exp(-r.*deltaT);
p_u = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2+nu*deltaT/deltaX);
p_d = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2-nu*deltaT/deltaX);
p_m = discount - p_u - p_d;
exp_dX = exp(+deltaX);

SVals = zeros(2*N+1,1);
SVals(1) = S0*exp(-N*deltaX);
CVals(1) = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i=2:2*N+1
    SVals(i)=exp_dX*SVals(i-1);
    CVals(i) = blsprice(SVals(i), X2, r, T2-T1, sigma2);
end
CCVals = zeros(2 .* N + 1, 1);
k=mod(N,2)+1; 
for i=1:2*N+1 
    CCVals(i,k)=max(CVals(i)-X1,0);
end

for j=N-1:-1:0
    know=mod(j,2)+1; 
    knext=mod(j+1,2)+1; 
    for i=N+1-j:N+1+j
        CCVals(i,know)=p_d*CCVals(i-1,knext)+ ...
                        p_m*CCVals(i,knext)+ ...
                        p_u*CCVals(i+1,knext);
    end
end
price=CCVals(N+1,1); 
end

function price = PCTrinomial(S0, X1, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
deltaT = T1 ./ N; nu = r - 0.5 .* sigma1 .^ 2; discount = exp(-r.*deltaT);
p_u = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2+nu*deltaT/deltaX);
p_d = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2-nu*deltaT/deltaX);
p_m = discount - p_u - p_d;
exp_dX = exp(+deltaX);

SVals = zeros(2*N+1,1);
SVals(1) = S0*exp(-N*deltaX);
CVals(1) = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i=2:2*N+1
    SVals(i)=exp_dX*SVals(i-1);
    CVals(i) = blsprice(SVals(i), X2, r, T2-T1, sigma2);
end
PCVals = zeros(2 .* N + 1, 1);
k=mod(N,2)+1; 
for i=1:2*N+1 
    PCVals(i,k)=max(X1-CVals(i),0);
end

for j=N-1:-1:0
    know=mod(j,2)+1; 
knext=mod(j+1,2)+1;    
for i=N+1-j:N+1+j
        PCVals(i,know)=p_d*PCVals(i-1,knext)+ ...
                        p_m*PCVals(i,knext)+ ...
                        p_u*PCVals(i+1,knext);
    end
end
price=PCVals(N+1,1); 
end

function price = CPTrinomial(S0, X1, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
deltaT = T1 ./ N; nu = r - 0.5 .* sigma1 .^ 2; discount = exp(-r.*deltaT);
p_u = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2+nu*deltaT/deltaX);
p_d = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2-nu*deltaT/deltaX);
p_m = discount - p_u - p_d;
exp_dX = exp(+deltaX);

SVals = zeros(2*N+1,1);
SVals(1) = S0*exp(-N*deltaX);
[CVals(1), PVals(1)] = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i=2:2*N+1
    SVals(i)=exp_dX*SVals(i-1);
    [CVals(i), PVals(i)] = blsprice(SVals(i), X2, r, T2 - T1, sigma2);
end
CPVals = zeros(2 .* N + 1, 1);
k=mod(N,2)+1; 
for i=1:2*N+1 
    CPVals(i,k)=max(PVals(i)-X1,0);
end

for j=N-1:-1:0
    know=mod(j,2)+1; 
    knext=mod(j+1,2)+1; 
    for i=N+1-j:N+1+j
        CPVals(i,know)=p_d*CPVals(i-1,knext)+ ...
                        p_m*CPVals(i,knext)+ ...
                        p_u*CPVals(i+1,knext);
    end
end
price=CPVals(N+1,1); 
end

function price = PPTrinomial(S0, X1, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
deltaT = T1 ./ N; nu = r - 0.5 .* sigma1 .^ 2; discount = exp(-r.*deltaT);
p_u = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2+nu*deltaT/deltaX);
p_d = discount*0.5*((sigma1^2*deltaT+nu^2*deltaT^2)/deltaX^2-nu*deltaT/deltaX);
p_m = discount - p_u - p_d;
exp_dX = exp(+deltaX);

SVals = zeros(2*N+1,1);
SVals(1) = S0*exp(-N*deltaX);
[CVals(1), PVals(1)] = blsprice(SVals(1), X2, r, T2 - T1, sigma2);
for i=2:2*N+1
    SVals(i)=exp_dX*SVals(i-1);
    [CVals(i), PVals(i)] = blsprice(SVals(i), X2, r, T2 - T1, sigma2);
end
PPVals = zeros(2 .* N + 1, 1);
k=mod(N,2)+1; 
for i=1:2*N+1 
    PPVals(i,k)=max(X1-PVals(i),0);
end

for j=N-1:-1:0
    know=mod(j,2)+1; 
    knext=mod(j+1,2)+1; 
    for i=N+1-j:N+1+j
        PPVals(i,know)=p_d*PPVals(i-1,knext)+ ...
                        p_m*PPVals(i,knext)+ ...
                        p_u*PPVals(i+1,knext);
    end
end
price=PPVals(N+1,1); 
end


% Trinomial Lattice Calculation: reasonable deltaX & Client 5, 1, 2, 3, 4

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.15; X2 = S0; 
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365; deltaT = T1 ./ N;
deltaX = sqrt((deltaT - deltaT .^ 2 ./ 2) .* sigma1 .^ 2 + r .* deltaT .^ 2)
CCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
CPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.15; sigma2 = 0.30; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365; deltaT = T1 ./ N;
deltaX = sqrt((deltaT - deltaT .^ 2 ./ 2) .* sigma1 .^ 2 + r .* deltaT .^ 2)
CCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
CPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.2; sigma2 = 0.18; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365; deltaT = T1 ./ N;
deltaX = sqrt((deltaT - deltaT .^ 2 ./ 2) .* sigma1 .^ 2 + r .* deltaT .^ 2)
CCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
CPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.18; sigma2 = 0.12; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365; deltaT = T1 ./ N;
deltaX = sqrt((deltaT - deltaT .^ 2 ./ 2) .* sigma1 .^ 2 + r .* deltaT .^ 2)
CCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
CPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)

S0 = 50; r = 0.025; T1 = 1; T2 = 2; sigma1 = 0.35; sigma2 = 0.1; X2 = S0;
[Call, Put] = blsprice(S0, X2, r, T2-T1, sigma2); N =365; deltaT = T1 ./ N;
deltaX = sqrt((deltaT - deltaT .^ 2 ./ 2) .* sigma1 .^ 2 + r .* deltaT .^ 2)
CCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PCTrinomial(S0, Call, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
CPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)
PPTrinomial(S0, Put, X2, r, T1, T2, sigma1, sigma2, N, deltaX)


% Risk-return Analysis --- Max Sharpe Method
% Client 5, Use closed_form, Stock price MC

S0 = 50; T1 = 1; T2 = 2; r = 0.025;
mu1 = r + 0.030; mu2 = r - 0.050; sigma1 = 0.15; sigma2 = 0.15; seed = 777; N = 100000; 
nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
rng(seed); e = randn(N,2);
ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));
[blsprice(S0, S0, r, T2-T1, sigma2), S0]	

    % Max Sharpe

[X, fval] = fminunc(@(X) std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* Calloncall(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r) ./ mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* Calloncall(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0])	
    1/fval

[X, fval] = fminunc(@(X) std(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* Callonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r) ./ mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(( blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* Callonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0])  
    1/fval
    
        % Mean R, Mean ER, Std ER, VaR (Callonput as an example)
    mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* Callonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1)

    mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* Callonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r)

    std(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* Callonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r)

    prctile(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) - Callonput(S0, X(1), X(2), T1, T2, r, sigma1), 5)

[X, fval] = fminunc(@(X) std(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* Putoncall(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r) ./ mean(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* Putoncall(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0])	
    1/fval

[X, fval] = fminunc(@(X) std(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* Putonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r) ./ mean(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* Putonput(S0, X(1), X(2), T1, T2, r, sigma1)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0])  
    1/fval

    % Stock Sharpe
mean(ST1T2(:,2)/S0-2*r-1) ./ std(ST1T2(:,2)/S0-2*r-1)

    % Call(at T1) & Put(at T1) Sharpe
mean(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2) ./ std(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2)

mean(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1)) - 2 * r - 2) ./ std(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1))  - 2 * r - 2)


% Client 1,USe Lattice, Stock price MC

S0 = 50; T1 = 1; T2 = 2; r = 0.025;
mu1 = r + 0.03; mu2 = r + 0.005; sigma1 = 0.15; sigma2 = 0.30; seed = 777; N = 100000; 
nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
rng(seed); e = randn(N,2);
ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));

    % Max Sharpe

[X, fval] = fmincon(@(X) std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
    
    % Mean R, Mean ER, Std ER, VaR (CC as an example)
    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1)

    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)

    std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) 

    prctile(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) - CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)

[X, fval] = fmincon(@(X) std(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(( blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])  
    1/fval
         
        % VaR
    prctile(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) - CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)
[X, fval] = fmincon(@(X) std(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
        
        % VaR
    prctile(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) - PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
[X, fval] = fmincon(@(X) std(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])
    1/fval
       
        % VaR
    prctile(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) - PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)

    % Stock Sharpe
mean(ST1T2(:,2)/S0-2*r-1) ./ std(ST1T2(:,2)/S0-2*r-1)

    % Call(at T1) & Put(at T1) Sharpe
mean(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2) ./ std(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2)

mean(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1)) - 2 * r - 2) ./ std(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1))  - 2 * r - 2)


% Client 2 Stock price MC

S0 = 50; T1 = 1; T2 = 2; r = 0.025;
mu1 = r - 0.03; mu2 = r - 0.01; sigma1 = 0.2; sigma2 = 0.18; seed = 777; N = 100000; 
nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
rng(seed); e = randn(N,2);
ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));

    % Max Sharpe

[X, fval] = fmincon(@(X) std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
    
        % Mean R, Mean ER, Std ER, VaR (CC as an example)
    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1)

    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)

    std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) 

    prctile(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) - CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)

[X, fval] = fmincon(@(X) std(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(( blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])  
    1/fval
        % VaR
    prctile(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) - CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)
    
[X, fval] = fmincon(@(X) std(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
        % VaR
    prctile(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) - PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
[X, fval] = fmincon(@(X) std(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])
    1/fval
        % VaR
    prctile(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) - PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
    % Stock Sharpe
mean(ST1T2(:,2)/S0-2*r-1) ./ std(ST1T2(:,2)/S0-2*r-1)

    % Call(at T1) & Put(at T1) Sharpe
mean(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2) ./ std(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2)

mean(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1)) - 2 * r - 2) ./ std(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1))  - 2 * r - 2)


% Client 3 Stock price MC

S0 = 50; T1 = 1; T2 = 2; r = 0.025;
mu1 = r - 0.03; mu2 = r + 0.03; sigma1 = 0.18; sigma2 = 0.12; seed = 777; N = 100000;
nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
rng(seed); e = randn(N,2);
ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));

    % Max Sharpe

[X, fval] = fmincon(@(X) std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
    
        % Mean R, Mean ER, Std ER, VaR (CC as an example)
    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1)

    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)

    std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) 

    prctile(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) - CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)

[X, fval] = fmincon(@(X) std(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(( blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])  
    1/fval
        % VaR
    prctile(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) - CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)
    
[X, fval] = fmincon(@(X) std(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
        % VaR
    prctile(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) - PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
[X, fval] = fmincon(@(X) std(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])
    1/fval
        % VaR
    prctile(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) - PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
    % Stock Sharpe
mean(ST1T2(:,2)/S0-2*r-1) ./ std(ST1T2(:,2)/S0-2*r-1)

    % Call(at T1) & Put(at T1) Sharpe
mean(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2) ./ std(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2)

mean(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1)) - 2 * r - 2) ./ std(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1))  - 2 * r - 2)


% Client 4 Stock price MC

S0 = 50; T1 = 1; T2 = 2; r = 0.025;
mu1 = r + 0.02; mu2 = r + 0.02; sigma1 = 0.35; sigma2 = 0.10; seed = 777; N = 100000;
nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
rng(seed); e = randn(N,2);
ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));

    % Max Sharpe

[X, fval] = fmincon(@(X) std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
    
        % Mean R, Mean ER, Std ER, VaR (CC as an example)
    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1)

    mean(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)

    std(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) .* CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) 

    prctile(((max(ST1T2(:,2) - X(2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)-X(1)) - CCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)
    
[X, fval] = fmincon(@(X) std(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./((blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(( blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) .* CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])  
    1/fval
        % VaR
    prctile(((max(X(2)-ST1T2(:,2), 0) - X(1)) .* max(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1), 0)) ./(blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) + X(2) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X(1)) - CPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365), 5)
    
[X, fval] = fmincon(@(X) std(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) .* PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2), S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * blsprice(S0, S0, r, T2-T1, sigma2); 1.25 * S0; -0.75 * blsprice(S0, S0, r, T2-T1, sigma2); -0.75 * S0])	
    1/fval
        % VaR
    prctile(((-max(ST1T2(:,2) - X(2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2)+X(1)) - PCLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
[X, fval] = fmincon(@(X) std(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ./ mean(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./((-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) .* PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r), [blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0, S0], [1, 0; 0, 1; -1, 0; 0, -1], [1.25 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); 1.25 * S0; -0.75 * (blsprice(S0, S0, r, T2-T1, sigma2) + S0 .* exp(-r .*(T2-T1)) - S0); -0.75 * S0])
    1/fval
        % VaR
    prctile(((-max(X(2)-ST1T2(:,2), 0) + X(1)) .* max(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1), 0)) ./(-blsprice(ST1T2(:, 1), X(2), r, T2-T1, sigma2) - X(2) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X(1)) - PPLattice(S0, X(1), X(2), r, T1, T2, sigma1, sigma2, 365),5)
    
    % Stock Sharpe
mean(ST1T2(:,2)/S0-2*r-1) ./ std(ST1T2(:,2)/S0-2*r-1)

    % Call(at T1) & Put(at T1) Sharpe
mean(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2) ./ std(2 .* max(ST1T2(:,2)-S0, 0) ./ blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) - 2 * r - 2)

mean(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1)) - 2 * r - 2) ./ std(2 .* max(S0-ST1T2(:,2), 0) ./ (blsprice(ST1T2(:,1), S0, r, T2-T1, sigma2) + S0 .* exp(-r .* (T2-T1)) - ST1T2(:,1))  - 2 * r - 2)


% Risk-return Analysis --- Sensitivity Analysis
%Sensitivity Matrix

function [VaR, Sharpe, X2, X1C] = sensitivityCC(S0, T1, T2, r, mu1, mu2, sigma1, sigma2, seed, N)
    nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
    nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
    sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
    rng(seed); e = randn(N,2);
    ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
    ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));
    
    X2 = linspace(0.75 * S0, 1.25 * S0, 11);
    X1C = linspace(0.75 * blsprice(S0, S0, r, T2-T1, sigma2)...
        , 1.25 * blsprice(S0, S0, r, T2-T1, sigma2), 11);
    X1P = linspace(0.75 * (blsprice(S0, S0, r, T2-T1, sigma2)+...
        S0 .* exp(-r .*(T2-T1)) - S0), 1.25 * (blsprice(S0, S0, r, T2-T1, sigma2)...
        + S0 .* exp(-r .*(T2-T1)) - S0), 11);
    
    for i = 1:11
        for j = 1:11
        VaR(i, j) = prctile(((max(ST1T2(:,2) - X2(i), 0) - X1C(j))...
            .* max(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)-X1C(j), 0))...
            ./(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)- X1C(j))...
            - CCLattice(S0, X1C(j), X2(i), r, T1, T2, sigma1, sigma2, 365), 5);
        Sharpe(i, j) = mean(((max(ST1T2(:,2) - X2(i), 0) - X1C(j)) .* ...
            max(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)-X1C(j), 0))...
            ./((blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)-X1C(j))...
            .* CCLattice(S0, X1C(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)...
            ./ std(((max(ST1T2(:,2) - X2(i), 0) - X1C(j)) .* ...
            max(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)-X1C(j), 0))...
            ./((blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)-X1C(j))...
            .* CCLattice(S0, X1C(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r);
        end
    end
end

function [VaR, Sharpe, X2, X1P] = sensitivityCP(S0, T1, T2, r, mu1, mu2, sigma1, sigma2, seed, N)
    nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
    nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
    sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
    rng(seed); e = randn(N,2);
    ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
    ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));
    
    X2 = linspace(0.75 * S0, 1.25 * S0, 11);
    X1C = linspace(0.75 * blsprice(S0, S0, r, T2-T1, sigma2)...
        , 1.25 * blsprice(S0, S0, r, T2-T1, sigma2), 11);
    X1P = linspace(0.75 * (blsprice(S0, S0, r, T2-T1, sigma2)+...
        S0 .* exp(-r .*(T2-T1)) - S0), 1.25 * (blsprice(S0, S0, r, T2-T1, sigma2)...
        + S0 .* exp(-r .*(T2-T1)) - S0), 11);
    
    for i = 1:11
        for j = 1:11
        VaR(i, j) = prctile(((max(X2(i)-ST1T2(:,2), 0) - X1P(j)) .*...
            max(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) + X2(i) .* ...
            exp(-r .*(T2-T1)) - ST1T2(:, 1)-X1P(j), 0)) ./...
            (blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) + X2(i) .* ...
            exp(-r .*(T2-T1)) - ST1T2(:, 1)-X1P(j)) - ...
            CPLattice(S0, X1P(j), X2(i), r, T1, T2, sigma1, sigma2, 365), 5);
        Sharpe(i, j) = mean(((max(X2(i)-ST1T2(:,2), 0) - X1P(j)) .* ...
            max(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) + ...
            X2(i) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X1P(j), 0)) ./...
            (( blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) + X2(i) .*...
            exp(-r .*(T2-T1)) - ST1T2(:, 1)-X1P(j)) .* ...
            CPLattice(S0, X1P(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r) ...
            ./ std(((max(X2(i)-ST1T2(:,2), 0) - X1P(j)) .* ...
            max(blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) + ...
            X2(i) .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X1P(j), 0)) ./...
            ((blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) + X2(i) ...
            .* exp(-r .*(T2-T1)) - ST1T2(:, 1)-X1P(j)) .* ...
            CPLattice(S0, X1P(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r);
        end
    end
end

function [VaR, Sharpe, X2, X1C] = sensitivityPC(S0, T1, T2, r, mu1, mu2, sigma1, sigma2, seed, N)
   nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
    nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
    sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
    rng(seed); e = randn(N,2);
    ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
    ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));
    
    X2 = linspace(0.75 * S0, 1.25 * S0, 11);
    X1C = linspace(0.75 * blsprice(S0, S0, r, T2-T1, sigma2)...
        , 1.25 * blsprice(S0, S0, r, T2-T1, sigma2), 11);
    X1P = linspace(0.75 * (blsprice(S0, S0, r, T2-T1, sigma2)+...
        S0 .* exp(-r .*(T2-T1)) - S0), 1.25 * (blsprice(S0, S0, r, T2-T1, sigma2)...
        + S0 .* exp(-r .*(T2-T1)) - S0), 11);
    
    for i = 1:11
        for j = 1:11
        VaR(i, j) = prctile(((-max(ST1T2(:,2) - X2(i), 0) + X1C(j)) .* ...
            max(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)+X1C(j), 0)) ...
            ./(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)+X1C(j)) - ...
            PCLattice(S0, X1C(j), X2(i), r, T1, T2, sigma1, sigma2, 365),5);
        Sharpe(i, j) = mean(((-max(ST1T2(:,2) - X2(i), 0) + X1C(j))...
            .* max(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)+X1C(j), 0))...
            ./((-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)+X1C(j)) .* ...
            PCLattice(S0, X1C(j),X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)...
            ./ std(((-max(ST1T2(:,2) - X2(i), 0) + X1C(j)) .* ...
            max(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)+X1C(j), 0))...
            ./((-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2)+X1C(j)) .*...
            PCLattice(S0, X1C(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r);
        end
    end
end

function [VaR, Sharpe, X2, X1P] = sensitivityPP(S0, T1, T2, r, mu1, mu2, sigma1, sigma2, seed, N)
    nu1 = mu1 - sigma1 .^ 2 ./ 2; nu1T1 = nu1 .* T1; sigma1sqrtT1 = sigma1 .* sqrt(T1);
    nu2 = mu2  - sigma2 .^ 2 ./ 2; nu2T2minusT1 = nu2 .* (T2-T1);
    sigma2sqrtT2minusT1 = sigma2 .* sqrt(T2-T1);
    rng(seed); e = randn(N,2);
    ST1T2 = S0 * exp(nu1T1 + sigma1sqrtT1 .* e(:,1));
    ST1T2(:,2) = ST1T2(:,1) .* exp(nu2T2minusT1 + sigma2sqrtT2minusT1 .* e(:,2));
    
    X2 = linspace(0.75 * S0, 1.25 * S0, 11);
    X1C = linspace(0.75 * blsprice(S0, S0, r, T2-T1, sigma2)...
        , 1.25 * blsprice(S0, S0, r, T2-T1, sigma2), 11);
    X1P = linspace(0.75 * (blsprice(S0, S0, r, T2-T1, sigma2)+...
        S0 .* exp(-r .*(T2-T1)) - S0), 1.25 * (blsprice(S0, S0, r, T2-T1, sigma2)...
        + S0 .* exp(-r .*(T2-T1)) - S0), 11);
      
    for i = 1:11
        for j = 1:11
        VaR(i, j) = prctile(((-max(X2(i)-ST1T2(:,2), 0) + X1P(j))...
       .* max(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) - X2(i) ...
       .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X1P(j), 0)) ./...
       (-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) - X2(i) .* exp(-r .*(T2-T1))...
       + ST1T2(:, 1)+X1P(j)) - ...
       PPLattice(S0, X1P(j), X2(i), r, T1, T2, sigma1, sigma2, 365),5);
        
        Sharpe(i, j) = mean(((-max(X2(i)-ST1T2(:,2), 0) + X1P(j)) .* ...
            max(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) - ...
            X2(i) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X1P(j), 0)) ./...
            ((-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) - X2(i) ...
            .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X1P(j)) .* ...
            PPLattice(S0, X1P(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r)...
            ./ std(((-max(X2(i)-ST1T2(:,2), 0) + X1P(j)) ...
            .* max(-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) - X2(i)...
            .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X1P(j), 0)) ./...
            ((-blsprice(ST1T2(:, 1), X2(i), r, T2-T1, sigma2) - ...
            X2(i) .* exp(-r .*(T2-T1)) + ST1T2(:, 1)+X1P(j)) .* ...
            PPLattice(S0, X1P(j), X2(i), r, T1, T2, sigma1, sigma2, 365)) - 1 - 2 * r);
        end
    end 
end

% Get Matrix for CC, CP, PC, PP; Client 1, 2, 3, 4

[VaR, Sharpe] = sensitivityCC(50, 1, 2, 0.025, 0.025+0.03, 0.025+0.005, 0.15, 0.3, 777, 100000)
[VaR, Sharpe] = sensitivityCC(50, 1, 2, 0.025, 0.025-0.03, 0.025-0.01, 0.2, 0.18, 777, 100000)
[VaR, Sharpe] = sensitivityCC(50, 1, 2, 0.025, 0.025-0.03, 0.025+0.03, 0.18, 0.12, 777, 100000)
[VaR, Sharpe] = sensitivityCC(50, 1, 2, 0.025, 0.025+0.02, 0.025+0.02, 0.35, 0.1, 777, 100000)

[VaR, Sharpe] = sensitivityCP(50, 1, 2, 0.025, 0.025+0.03, 0.025+0.005, 0.15, 0.3, 777, 100000)
[VaR, Sharpe] = sensitivityCP(50, 1, 2, 0.025, 0.025-0.03, 0.025-0.01, 0.2, 0.18, 777, 100000)
[VaR, Sharpe] = sensitivityCP(50, 1, 2, 0.025, 0.025-0.03, 0.025+0.03, 0.18, 0.12, 777, 100000)
[VaR, Sharpe] = sensitivityCP(50, 1, 2, 0.025, 0.025+0.02, 0.025+0.02, 0.35, 0.1, 777, 100000)

[VaR, Sharpe] = sensitivityPC(50, 1, 2, 0.025, 0.025+0.03, 0.025+0.005, 0.15, 0.3, 777, 100000)
[VaR, Sharpe] = sensitivityPC(50, 1, 2, 0.025, 0.025-0.03, 0.025-0.01, 0.2, 0.18, 777, 100000)
[VaR, Sharpe] = sensitivityPC(50, 1, 2, 0.025, 0.025-0.03, 0.025+0.03, 0.18, 0.12, 777, 100000)
[VaR, Sharpe] = sensitivityPC(50, 1, 2, 0.025, 0.025+0.02, 0.025+0.02, 0.35, 0.1, 777, 100000)

[VaR, Sharpe] = sensitivityPP(50, 1, 2, 0.025, 0.025+0.03, 0.025+0.005, 0.15, 0.3, 777, 100000)
[VaR, Sharpe] = sensitivityPP(50, 1, 2, 0.025, 0.025-0.03, 0.025-0.01, 0.2, 0.18, 777, 100000)
[VaR, Sharpe] = sensitivityPP(50, 1, 2, 0.025, 0.025-0.03, 0.025+0.03, 0.18, 0.12, 777, 100000)
[VaR, Sharpe] = sensitivityPP(50, 1, 2, 0.025, 0.025+0.02, 0.025+0.02, 0.35, 0.1, 777, 100000)


end

