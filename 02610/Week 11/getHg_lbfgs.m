function Hg = getHg_lbfgs(df,S,Y,gamma)

[n,m] = size(S);
rho = zeros(m,1);
for i = 1:m
    rho(i) = 1/(Y(:,i)'*S(:,i));
end

alpha =zeros(m,1);

% step 1
q = df;

% first loop
for i = m:-1:1
    alpha(i) = rho(i)*S(:,i)'*q;
    q = q-alpha(i)*Y(:,i);
end

% Multiply by Initial Hessian
r = gamma*q;

% second loop
for i = 1:m
    beta = rho(i)*Y(:,i)'*r;
    r = r + S(:,i)*(alpha(i)-beta);
end
% 
Hg=r;