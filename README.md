# **IPNN - Continuous Indeterminate Probability Neural Network**

Pytorch implementation of paper:  **Continuous Indeterminate Probability Neural Network**,  
By Yang Tao

## **CIPNN Equation** 


\begin{multline}
    \label{eq:P_Y_X_via_Z_norm}
    P^{Z} \left ( y_{l} \mid x_{n+1} \right ) \\
    \begin{aligned}
    &\approx \frac{1}{C}\sum_{c=1}^{C} \left ( \frac{ \sum_{k=1}^{n} \left (  y_{l}(k)\cdot \prod_{i}^{N} \mathcal{N} \left ( g\left ( \epsilon_{c},\theta_{n+1}^{i}  \right );\theta_{k}^{i}  \right )  \right ) }
    {\sum_{k=1}^{n} \left ( \prod_{i}^{N} \mathcal{N} \left ( g\left ( \epsilon_{c},\theta_{n+1}^{i} \right );\theta_{k}^{i}  \right ) \right ) } \right )
    \end{aligned}
    \end{multline}
