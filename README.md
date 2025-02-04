# sentiment-analysis-volatility-forecasting

This is our repository for our FYP project, sentiment analysis volatility forecasting.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/hwh1771/sentiment-analysis-volatility-forecasting.git
   ```
2. Install packages
   ```sh
   pip install -r requirements.txt
   ```



### GARCH X

Our model incorporates the exogenous term in the GARCH model's volatility component. 

#### Code
Our GARCH model implementation is in the `Garch.py` file. Usage:
```python
from Garch import GARCH
garch = GARCH(p=1,q=1,z=1)
y = [0.1,0.3, 0.5]
x = np.array([0.4, 0.2, 0.3]).reshape(3,1)  # Shape should be (T, n) for T time steps and n variables.
res = garch.train(y, x=x)
```






<p align="right">(<a href="#readme-top">back to top</a>)</p>