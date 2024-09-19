import numpy as np

# Compute order of convergence function
def compute_order(x, xstar):
    diff1 = np.abs(x[1::] - xstar)
    diff2 = np.abs(x[0:-1] - xstar)
    
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    
    _lambda = np.exp(fit[1]) 
    alpha = fit[0] 

    return alpha, _lambda


def fixedpt(f, x0, tol, Nmax):
    x_approximations = [x0] 
    count = 0

    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        x_approximations.append(x1) 
        if abs(x1 - x0) < tol:
            print("count:", count)
            return np.array(x_approximations), 0
        x0 = x1
    return np.array(x_approximations), 1

def test():
    tol = 1e-10
    Nmax = 100
    f = lambda x: (10/(x+4))**.5
    x0 = 1.5

    x_seq, _ = fixedpt(f, x0, tol, Nmax)
    
    xstar = 1.3652300134140976  # Approximate known fixed point

    alpha, _lambda = compute_order(x_seq, xstar)
    
    print("alpha:", alpha)
    print("lambda:", _lambda)

test()


def aitken_acceleration(p_seq, tol, Nmax):
    accelerated_seq = []
    count = 0
    while count < len(p_seq) - 2:
        p_n = p_seq[count]
        p_n1 = p_seq[count + 1]
        p_n2 = p_seq[count + 2]
        
        denominator = p_n2 - 2 * p_n1 + p_n
        if abs(denominator) < tol:  # Avoid division by zero or small numbers
            break
        
        p_hat_n = p_n - (p_n1 - p_n)**2 / denominator
        accelerated_seq.append(p_hat_n)
        
        count += 1
        if count >= Nmax:
            break

    return np.array(accelerated_seq)

# Example to apply Aitken's delta^2 method
def test_aitken():
    tol = 1e-6
    Nmax = 100
    f = lambda x: 1 + 0.5 * np.sin(x)  # Test function
    x0 = 0.0

    # Generate fixed-point sequence
    x_seq, _ = fixedpt(f, x0, tol, Nmax)

    # Apply Aitken's delta^2 method
    accelerated_seq = aitken_acceleration(x_seq, tol, Nmax)

    # Print results
    print("Original fixed-point approximations:\n", x_seq)
    print("Accelerated sequence (Aitken's ∆²):\n", accelerated_seq)

test_aitken()



