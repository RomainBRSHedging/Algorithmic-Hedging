import numpy as np
from scipy.stats import norm



RNG = np.random.default_rng()


class Black_Scholes:

    def __init__(self,N,M, sigma, r,q, K,T):
        self.N = N
        self.M = M
        self.sigma = sigma
        self.r = r
        self.q = q
        self.K = K
        self.T = T
    

    def BS_simul(self,N,M, sigma, r,q, K,T, S0 ):
        Wt = RNG.normal(0,1,M)*np.sqrt(T/N)
        St = S0
        for i in range(N):
            St = St*np.exp( (r - q -sigma**2/2)*(T/N) + sigma*Wt)
        return St

    def Prix_call_MC(self, S, r,q, K,T ):
        payoff = np.where(S-K >0, S-K, 0)
        return np.mean(payoff)*np.exp(-(r - q)*T)

    def prix_BS(self, S0,sigma, r,q, K,T ):
        d1 = (np.log(S0/K) + r - q - (sigma**2)*T/2. )/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)#,  K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1) 
    
    def greeks_call(self,S0,sigma, r,q, K,T  ):
        d1 = (np.log(S0/K) + r - q - (sigma**2)*T/2. )/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return np.exp(-q*T)*norm.cdf(d1), norm.pdf(d1)* S0*sigma*np.sqrt(T), S0 * T**0.5 * norm.pdf(d1)
    




class Heston(Black_Scholes):
    def __init__(self, N, M, sigma, r, q, K, T,epsilon_H, rho, theta,k,P ):
        super().__init__(N, M, sigma, r, q, K, T)
        self.epsilon_H = epsilon_H
        self.rho = rho
        self.theta = theta
        self.k = k
        self.P = P

    def Heston_simul(self, N, M, S0, Sigma0, r, epsilon, rho, theta, k, P, dt):
        Stp = np.empty((M,N+1))
        Stp[:,0] = S0
        Sigmatp = np.empty((M,N+1))
        Sigmatp[:,0] = Sigma0
        Sigmaplus = np.empty((M,N+1))
        Sigmaplus[:,0] = Sigma0

        for i in range(1, N+1):
            dWt1 = RNG.normal(0,1,size=(M,N+1))*np.sqrt(dt)
            dWt2 = RNG.normal(0,1,size=(M,N+1))*np.sqrt(dt)

            Stp[:,i] = Stp[:,i-1] * np.exp(r - Sigmaplus[:,i-1]*dt + epsilon*np.sqrt(Sigmaplus[:,i-1]*dt)*dWt1[:,i-1] ) 
            Sigmatp[:,i] = Sigmatp[:,i-1] + k*(theta - .5*Sigmaplus[:,i-1])*dt + epsilon*np.sqrt(Sigmaplus[:,i-1]*dt)* (rho*dWt1[:,i-1] + (1 - rho**2)*dWt2[:,i-1])
            Sigmaplus[:,i] = [max(0.,Sigmatp[j,i]) for j in range(M)]
            Sigmatp = Sigmatp + P

        return Stp
    
    def prix_call_put_Heston(self, S):
        payoffcall_heston = np.where(S - K >0, S-K, 0)
        payoffput_heston = np.where(K - S >0,K - S, 0)
        return np.mean(payoffcall_heston)*np.exp(-r*T), np.mean(payoffput_heston)*np.exp(-r*T)
    
    def greeks_FD_MC(self, S):
        payoffcallh = np.where(S-K > 0, S-K  ,0)
        payoffcallminush = np.where(S-(T/N) -K > 0, S-(T/N)-K  ,0)
        payoffcallplush = np.where(S+(T/N) -K > 0, S+(T/N)-K  ,0)

        payoffputh = np.where(K-S > 0, K-S, 0)
        payoffputminush = np.where(K- (S-(T/N)) > 0, K- (S-(T/N)), 0)
        payoffputplush = np.where(K- (S+(T/N)) > 0, K- (S+(T/N)), 0)

        prixcallh = np.mean(payoffcallh)*np.exp(-r*T)
        prixcallplush = np.mean(payoffcallplush)*np.exp(-r*T)
        prixcallminush = np.mean(payoffcallminush)*np.exp(-r*T)

        prixputh = np.mean(payoffputh)*np.exp(-r*T)
        prixputplush = np.mean(payoffputplush)*np.exp(-r*T)
        prixputminush = np.mean(payoffputminush)*np.exp(-r*T)

        deltacallFD = (prixcallplush - prixcallminush)/(2*(T/N))
        deltaputFD =  (prixputplush - prixputminush)/(2*(T/N))
        gammacallFD = (prixcallplush + prixcallminush -2*prixcallh)/((T/N)**2)
        gammaputFD = (prixputplush + prixputminush -2*prixputh)/((T/N)**2)

        return deltacallFD,deltaputFD, gammacallFD, gammaputFD




N = 10
M = 1000
sigma = .1
r = .01
q = 0
S0 = 100
K = S0
K1 = S0 + 5
K2 = S0 + 10
T = 1



###Short 1000 call###
###Hedging this position###

Pricer = Black_Scholes(N,M, sigma, r,q, K,T )
prix_call_BS = Pricer.prix_BS(S0,sigma, r,q, K,T )
Delta, Gamma, Vega = Pricer.greeks_call(S0,sigma, r,q, K,T )


print('call hedge: ', prix_call_BS*1000)
print(Delta*-1000)
print(Gamma*-1000)
print(Vega*-1000)

print('-'*60)

#for hedging purpose
prix_call_BS2 = Pricer.prix_BS(S0,sigma, r,q, K1,T)
Delta1, Gamma1, Vega1 = Pricer.greeks_call(S0,sigma, r,q, K1,T )
print('Call 1 :' ) 
print(Delta1)
print(Gamma1)
print(Vega1)

print('-'*60)

#for hedging purpose
prix_call_BS3 = Pricer.prix_BS(S0,sigma, r,q, K2,T)
Delta2, Gamma2, Vega2 = Pricer.greeks_call(S0,sigma, r,q, K2,T )
print('Call 2 :' ) 
print(Delta2)
print(Gamma2)
print(Vega2)

greeks = np.array([[Gamma1,Gamma2], [Vega1, Vega2]]) #Call 1 & 2
portfolio_Greeks = np.array([[Gamma*1000], [Vega*1000]]) #Short Position

print('-'*60)

inv = np.linalg.inv(np.round(greeks, 2))
print('Inv Matrix :')
print(inv)

print('-'*60)

W = np.dot(inv, portfolio_Greeks) # W such as the short position is hedged:  [[Gamma1, Gamma2],[Vega1, Vega2]]*[W1 W2]**T  =  [Gamma, Vega]**T

print(W)

print('-'*60)

print( np.round( np.dot(np.round(greeks,2), W) - portfolio_Greeks) ) # Test that our position in Gamma & Vega is hedged

print('-'*60)

#Now that our position in Gamma & Vega are hedged, we just need to hedge our position in Delta.
#So we have to be long delta (long_d) in U Asset
greeks = np.array([[Delta1, Delta2],[Gamma1,Gamma2], [Vega1, Vega2]])
portfolio_Greeks = np.array([[Delta*-1000],[Gamma*-1000], [Vega*-1000]]) 
long_d = np.array([[670], [0], [0]])
print(np.round(np.dot( np.round(greeks,2), W ) + portfolio_Greeks + long_d)) # Hedging Gamma & Vega + long position will hedge our short position


#Now we can do it with Heston Model class and the greeks calculated with finite difference above.