import numpy as np
import matplotlib.pyplot as plt


def generate_BS_paths(S, T, mu, C, steps, N):
    """
    Generates stock price paths with covariance matrix C and mean=mu, N times for legnth T and step size dt
    """
    v=C.diagonal()
    Npaths=C.shape[0]
    dt = T/steps
    size = (Npaths, steps+1, N)
    prices = np.zeros(size)
    S_t = S
    prices[:, 0, :] = S_t
    for n in range (N):
        S_t = S
        for t in range(steps):
            WT = np.random.multivariate_normal(np.array(np.zeros(Npaths)), 
                                            cov = C) * np.sqrt(dt) 
            
            S_t = S_t*(np.exp((mu-0.5*v**2)*dt+ v*WT) ) 
            prices[:, t+1, n] = S_t
            
    return prices
'''
c=np.array([[1,-1],
            [-1,1]])
m=np.array([.1,.1])


prices=generate_BS_paths(S=100, T=1, mu=m,  C=c, steps=100, N=1)
print(prices.shape[0])

print(np.var(prices[0,:,0]))
plt.figure(figsize=(7,6))
plt.plot(prices[0])
plt.plot(prices[1])

plt.title('BS Price Paths Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()'''

