from dataclasses import dataclass
import math

# from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go
import pandas.core.series
import scipy.stats as stats

# aestimer = [stats.levy_stable.rvs(alpha = 1.5, beta = .5, loc = .1, scale = .03) for _ in range(1000)]
# stats.levy_stable.fit(aestimer, loc=.1, scale=.03)


@dataclass(init=True, repr=True)
class PricingGBM:
    """Pricing of an asset after a period time with the Geometric Brownion Motion & Monte Carlo simulations
        current_price: float -> current asset price or assets_price
        mu: float -> interest free rate term
        sigma: float -> volatility term
        nb_steps: int -> number of steps that will be used by the Geometric Brownian Motion to simulate path. If dt != -1, nb_steps will be ignored.
        maturity: float -> end of simulation FROM NOW
        dt: float = -1  -> duration of step
        nb_simulations: int = 100 -> nb_simulations 
        drift: float = "" -> drif term of the Geometric Brownian Motion
    """

    current_price: float  # current asset price or assets_price
    mu: float  # drift term
    sigma: float  # volatility term
    nb_steps: int  # number of steps that will be used by the Geometric Brownian Motion to simulate path. If dt != -1, nb_steps will be ignored.
    maturity: float  # end of simulation FROM NOW
    dt: float = -1  # duration of step
    nb_simulations: int = 100
    drift: float = ""

    ##############################################
    # After the end of the instance construction #
    ##############################################
    def __post_init__(self):
        # create the time grid
        self.get_time_grid()
        # if drift not in args, we calculate the drift of the Geometric Brownian Motion 
        if self.drift == "":
            self.drift = self.mu - (self.sigma**2) / 2

    @staticmethod
    def calibrate_params_from_times_series(time_series):
        """Returns mu, sigma from normal distribution

        Args:
            time_series (list): time series to fit within the normal distribution

        Returns:
            mu : float -> loc of normal distribution
            sigma : float -> standard deviation of normal distribution
        """
        if type(time_series) == pandas.core.series.Series:
            time_series = time_series.tolist()
        mu, sigma = stats.norm.fit(time_series)

        return mu, sigma

    @classmethod
    def create_pricing_GBM_from_time_series(
        cls,
        time_series,
        nb_steps,
        maturity,
        returns=[],
        dt=-1,
        nb_simulations=100,
        drift="",
    ):
        """Create pricer based on the Geometric Brownian Motion from a time series.

        Args:
            time_series (list|iterable): prices series 
            nb_steps (int): number of steps for the simulation
            maturity (int|flaot): end of the simulation (may be deprecated)
            returns (list, optional): returns series in order to calibrate the normal distribution. Defaults to [].
            dt (int, optional): nb_steps/maturity, optional. Defaults to -1.
            nb_simulations (int, optional): number of simulations. Defaults to 100.
            drift (str, optional): mu - .5*sigma**2. Defaults to "".

        Returns:
            PricingGBM: Instance of the Pricer for Geometric Brownian Motion according to your args
        """
        ##########################################
        ### Calibration of normal distribution ###
        ##########################################

        ### if returns is not empty, we calibrate the normal distrbution from returns 
        if len(returns) > 0:
            mu, sigma = cls.calibrate_params_from_times_series(returns)
        ### else we calibrate the normal distrbution from time_series (time_series should be normally distributed)
        else:
            mu, sigma = cls.calibrate_params_from_times_series(
                time_series=time_series
            )

        current_price = 0

        ##########################
        ### Get the last price ###
        ##########################
        if type(time_series) == np.ndarray:
            current_price = time_series[len(time_series) - 1]
        if type(time_series) == pandas.core.series.Series:
            current_price = time_series[len(time_series) - 1]
        else:
            current_price = time_series[-1]

        #########################
        ### Return the pricer ###
        #########################
        return PricingGBM(
            current_price=current_price,
            mu=mu,
            sigma=sigma,
            nb_steps=nb_steps,
            maturity=maturity,
            dt=dt,
            nb_simulations=nb_simulations,
            drift=drift,
        )

    def simulate_multiple_path_gbm_sol(self):
        """
        Master function to simulate <nb_simulations> from the Geometric Brownian Motion solution

        Affect the attribute simulations to the instance as : 
        self.simulations = {
            "time": self.time_grid,
            "simulations": [
                self.simulate_one_path_gbm_sol() for _ in range(self.nb_simulations)
            ],
        }
        """
        self.simulations = {
            "time": self.time_grid,
            "simulations": [
                self.simulate_one_path_gbm_sol() for _ in range(self.nb_simulations)
            ],
        }

    def simulate_one_path_gbm_sol(self):
        """
        Returns:
            gbmOnePath: One path of simulation of the Geometric Brownian Motion
        """
        gbmOnePath = [self.current_price]
        for _ in self.time_grid[1:]:
            gbmOnePath.append(
                max(gbmOnePath[-1] * self.factor_brownian_motion_sol(), 0)
            )
        return gbmOnePath

    def factor_brownian_motion_sol(self):
        """
        Returns:
           factor_brownian_motion : a step of the simulation to multply to the precedent value 
        """
        if self.dt > 0:
            return np.exp(
                self.drift * self.dt + self.sigma * np.sqrt(self.dt) * stats.norm.rvs()
            )

        else:
            return np.exp(
                self.drift * (self.maturity / self.nb_steps)
                + self.sigma
                * np.sqrt((self.maturity / self.nb_steps))
                * stats.norm.rvs()
            )

    def see_simulations(self):
        """Helper function to see the simulated paths
        """
        fig = go.Figure()
        [
            fig.add_trace(go.Scatter(x=self.simulations["time"], y=une_simulation))
            for une_simulation in self.simulations["simulations"]
        ]
        fig.update_layout(showlegend=False)
        fig.show()
        pass

    def simulate_and_see(self):
        """Helper function to simulate multiple paths of the Geometric Brownian Motion and see the results
        """
        ########################
        ## Simulate the paths ##
        ########################

        self.simulate_multiple_path_gbm_sol()
        
        ######################
        ## See simulations ##
        ######################
        self.see_simulations()

    def get_time_grid(self):
        """Create time grid for the paths
        """
        if self.dt != -1:
            nb_points = self.maturity // self.dt
            time_grid = [self.dt * i for i in range(nb_points)]

        else:
            time_grid = [
                self.maturity * i / self.nb_steps for i in range(self.nb_steps + 1)
            ]

        self.time_grid = time_grid


@dataclass
class PricingGBMPortfolio(PricingGBM):
    """Pricing of a portfolio after a period time with the Geometric Brownion Motion & Monte Carlo simulations
        current_price: float -> current asset price or assets_price
        mu: float -> interest free rate term
        sigma: float -> volatility term
        nb_steps: int -> number of steps that will be used by the Geometric Brownian Motion to simulate path. If dt != -1, nb_steps will be ignored.
        maturity: float -> end of simulation FROM NOW
        correlation_matrix : np.array -> correlation matrix of the portfolio
        dt: float = -1  -> duration of step
        nb_simulations: int = 100 -> nb_simulations 
        drift: float = "" -> drif term of the Geometric Brownian Motion
    """
    correlation_matrix: np.array = np.array([])
    nb_steps = 1
    def __post_init__(self):
        super().__post_init__()
        ###################
        ## Create steps ##
        ###################
        if self.dt < 0:
            self.dt = self.maturity / self.nb_steps
        
        ########################################################
        ## Create Cholesky matrix to compute correlated paths ##
        ########################################################
        self.L = np.linalg.cholesky(self.correlation_matrix)



    def gbm_sol_matrix_normal(
        self,
    ):  
        """Create paths for the assets portfolio according to the Black&Scholes formula and correlation (Cholesky matrix)
        """
        return self.current_price * np.exp(
            self.drift * self.maturity
            + self.sigma
            * np.sqrt(self.maturity)
            * np.matmul(
                self.L, stats.norm.rvs(size=(len(self.current_price), self.nb_simulations))
            )
        )

    def paths_of_portfolio(self):
        """Returns all the simulated end price of the portfolio  
        """
        return np.sum(self.gbm_sol_matrix_normal(), axis=0)
    
    
    


if __name__ == "__main__":
    print("----------------------------------------------------------------------")
    print("Welcome on the pricing module based on the Geometrical Brownian Motion")
    print("----------------------------------------------------------------------")

    print("----------------------------------------------------------------------")
    print("A demo simulation will be launched")
    print("----------------------------------------------------------------------")

    
    ### Datas for the test ###

    import yfinance as yf


    sp500 = yf.Ticker("^GSPC")
    hist_sp500 = sp500.history("1y")
    hist_sp500["returns"] = hist_sp500.Close / hist_sp500.Close.shift(1) - 1
    returns_serie = hist_sp500.dropna().returns

    #################################################


    ### Pricing with Geometric Brownian Motion ###
    pricer = PricingGBM.create_pricing_GBM_from_time_series(
        hist_sp500.Close,
        nb_steps=100,
        maturity=1,
        returns=returns_serie,
        nb_simulations=1000,
    )
    pricer.simulate_and_see()

    #################################################

    print("----------------------------------------------------------------------")
    print("End of the demo, bye.\n\n Author : Adrien Calas - Le23RayGorbella")
    print("----------------------------------------------------------------------")

    
    
    
