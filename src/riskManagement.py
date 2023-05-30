from dataclasses import dataclass
import collections
from scipy.stats import norm
import numpy as np

from pricingGBM import PricingGBM, PricingGBMPortfolio


@dataclass
class RiskManagementOneAsset:
    """Class to manage risk on one asset (work in progress)
    Available :
    Value at Risk & Conditional Value at Risk (Expected shortfall) for historic, normal & Monte-Carlo method
    Args :
        time_series: collections.abc.Iterable -> time series of earning & losses for this asset
        mode: str = "historic" -> estimation method : historic, normal or Monte-Carlo
        input_type: str = "returns" -> type of time series in args (if value, the args returns will be used if necessarily)
        returns: list = -1 -> earnings/returns from the time series
    """

    time_series: collections.abc.Iterable  # time series of earning & losses for this asset
    mode: str = "historic"  # historic, normal, Monte-Carlo
    input_type: str = "returns"
    returns: list = -1

    def value_at_risk(
        self, nb_past_days, confiance, volatility=None, nb_days=1, nb_simulations=1000
    ):
        """Calculate the value at risk with the according method of the instance <self.mode>"""
        if self.mode == "historic":
            return self.value_at_risk_historic(nb_past_days, confiance, nb_days=nb_days)
        if self.mode == "normal":
            return self.value_at_risk_normal(
                confiance=confiance,
                nb_days=nb_days,
                nb_past_days=nb_past_days,
                volatility=volatility,
            )
        if self.mode == "Monte-Carlo":
            return self.value_at_risk_monte_carlo(
                confiance=confiance,
                nb_steps=nb_days,
                nb_simulations=nb_simulations,
            )

    def conditional_value_at_risk(
        self, nb_past_days, confiance, volatility=None, nb_days=1, nb_simulations=1000
    ):
        """Calculate the conditional value at risk with the according method of the instance <self.mode>"""
        if self.mode == "historic":
            return self.conditional_value_at_risk_historic(
                nb_past_days, confiance, nb_days=nb_days
            )
        if self.mode == "Monte-Carlo":
            return self.conditional_value_at_risk_monte_carlo(
                confiance=confiance,
                nb_steps=nb_days,
                nb_simulations=nb_simulations,
            )
        if self.mode == "normal":
            return self.conditional_value_at_risk_normal(
                confiance=confiance,
                nb_days=nb_days,
                nb_past_days=nb_past_days,
                volatility=volatility,
            )

    def value_at_risk_monte_carlo(
        self,
        confiance,
        nb_steps=1,
        nb_simulations=10000,
    ):
        """Calculate the value at risk from the monte-carlo method"""
        pricerGBM, end_price_simulations, index = self.monte_carlo_paths(
            confiance, nb_steps, nb_simulations
        )

        return end_price_simulations[index - 1] / pricerGBM.current_price - 1

    def conditional_value_at_risk_monte_carlo(
        self,
        confiance,
        nb_steps=1,
        nb_simulations=10000,
    ):
        """Calculate the conditional value at risk from the monte-carlo method"""

        ###############################################################
        ## Get end prices simulations and index of the value at risk ##
        ###############################################################

        pricerGBM, end_price_simulations, index = self.monte_carlo_paths(
            confiance, nb_steps, nb_simulations
        )

        return np.mean(end_price_simulations[:index]) / pricerGBM.current_price - 1

    def monte_carlo_paths(self, confiance, nb_steps, nb_simulations):
        """Returns the pricer, the end prices simulated and the index of the value at risk for the confiance level

        Args:
            confiance (float): confiance looking for the value at risk like .95 for 95% confident
            nb_steps (int): nb_steps of simulations
            nb_simulations (int): nb paths of simulations

        """

        ##############################################
        ## Initialize the Geometric Brownian pricer ##
        ##############################################

        if self.input_type != "value":
            pricerGBM = PricingGBM.create_pricing_GBM_from_time_series(
                self.time_series,
                nb_steps=nb_steps,
                maturity=nb_steps,
                nb_simulations=nb_simulations,
            )
        else:
            pricerGBM = PricingGBM.create_pricing_GBM_from_time_series(
                self.time_series,
                returns=self.returns,
                nb_steps=nb_steps,
                maturity=nb_steps,
                nb_simulations=nb_simulations,
            )

        ########################
        ## Simulate the paths ##
        ########################

        pricerGBM.simulate_multiple_path_gbm_sol()
        les_simulations = pricerGBM.simulations["simulations"]

        end_price_simulations = sorted([path[-1] for path in les_simulations])
        index = int((1 - confiance) * len(end_price_simulations))
        return pricerGBM, end_price_simulations, index

    def conditional_value_at_risk_historic(self, nb_past_days, confiance, nb_days=1):
        """Returns the conditional value at risk historic

        Args:
            nb_past_days (int): number of historical values/returns to keep from today
            confiance (_type_): confiance level for the value at risk
            nb_days (int, optional): number of days for the conditional value at risk. Defaults to 1.

        """
        etude = sorted(self.time_series[-nb_past_days:])
        index = int((1 - confiance) * nb_past_days)
        return np.sqrt(nb_days) * np.mean(etude[:index])

    def value_at_risk_historic(self, nb_past_days, confiance, nb_days=1):
        """Returns the conditional value at risk historic

        Args:
            nb_past_days (int): number of historical values/returns to keep from today
            confiance (_type_): confiance level for the value at risk
            nb_days (int, optional): number of days for the conditional value at risk. Defaults to 1.

        """
        etude = sorted(self.time_series[-nb_past_days:])
        index = int((1 - confiance) * nb_past_days)
        return np.sqrt(nb_days) * etude[index - 1]

    def value_at_risk_normal(
        self, confiance, nb_days, nb_past_days=None, volatility=None
    ):
        """Returns the value at risk accroding to the normal method

        Args:
            confiance (float):  confiance level for the value at risk
            nb_days (int):  number of days for the conditional value at risk.
            nb_past_days (int, optional):  number of historical values/returns to keep from today
            volatility (float, optional): historical volatility, default to None
        """
        return np.sqrt(nb_days) * self.value_at_risk_normal_1_day(
            confiance=confiance, nb_past_days=nb_past_days, volatility=volatility
        )

    def conditional_value_at_risk_normal(
        self, confiance, nb_days, nb_past_days=None, volatility=None
    ):
        """Returns the conditional value at risk accroding to the normal method

        Args:
            confiance (float):  confiance level for the value at risk
            nb_days (int):  number of days for the conditional value at risk.
            nb_past_days (int, optional):  number of historical values/returns to keep from today
            volatility (float, optional): historical volatility, default to None
        """
        return np.sqrt(nb_days) * self.conditional_value_at_risk_normal_1_day(
            confiance=confiance, nb_past_days=nb_past_days, volatility=volatility
        )

    def value_at_risk_normal_1_day(self, confiance, nb_past_days=None, volatility=None):
        """Returns value at risk accroding to the normal method for one day

        Args:
            confiance (float):  confiance level for the value at risk
            nb_past_days (int, optional):  number of historical values/returns to keep from today
            volatility (float, optional): historical volatility, default to None
        """
        if nb_past_days != None:
            etude = (
                sorted(self.returns[-nb_past_days:])
                if self.mode == "value"
                else sorted(self.time_series[-nb_past_days:])
            )
        else:
            etude = (
                sorted(self.returns)
                if self.mode == "value"
                else sorted(self.time_series[-nb_past_days:])
            )
        if volatility == None:
            mu, volatility = norm.fit(etude)

        quantile = norm.isf(confiance)

        return quantile * volatility - mu

    def conditional_value_at_risk_normal_1_day(
        self, confiance, nb_past_days=None, volatility=None
    ):
        """Returns the conditional value at risk accroding to the normal method for one day

        Args:
            confiance (float):  confiance level for the value at risk
            nb_past_days (int, optional):  number of historical values/returns to keep from today
            volatility (float, optional): historical volatility, default to None
        """
        if nb_past_days != None:
            etude = (
                sorted(self.returns[-nb_past_days:])
                if self.mode == "value"
                else sorted(self.time_series[-nb_past_days:])
            )
        else:
            etude = (
                sorted(self.returns)
                if self.mode == "value"
                else sorted(self.time_series[-nb_past_days:])
            )
        if volatility == None:
            mu, volatility = norm.fit(etude)

        return (
            -(1 / (1 - confiance)) * norm.pdf(norm.ppf(1 - confiance)) * volatility - mu
        )


@dataclass
class RiskManagementPortfolio:
    """Class to manage risk on a portfolio (work in progress)
    Available :
    Value at Risk for normal & Monte-Carlo method
    Args :
        assets_current_prices: list -> assets current prices in the portfolio
        correlation_matrix: np.array -> correlation matrix of the portfolio
        assets_volatilities: np.array -> assets volatilities of the portfolio
        assets_time_series = np.array([]) -> assets time series values
        mu: float = 0.1 -> interest risk free rate
        drift: float = "" -> drift of the geoetric brownian motion
        mode: str = "Monte-Carlo" -> method of the risk estimator : historic (unavailable), normal, Monte-Carlo
        returns: list = -1 -> returns/earnings list
        input_type: str = "returns" -> assets times series type if values or returns (earnings)
        nb_days: int = 1 -> number of days
    """

    assets_current_prices: list
    correlation_matrix: np.array
    assets_volatilities: np.array
    assets_time_series = np.array([])
    mu: float = 0.1
    drift: float = ""
    mode: str = "Monte-Carlo"  # historic, normal, Monte-Carlo
    returns: list = -1
    input_type: str = "returns"
    nb_days: int = 1

    def __post_init__(self):
        ### if no time series as been specified, affect the current prices to the time series
        if len(self.assets_time_series) == 0:
            self.assets_time_series = self.assets_current_prices

    @staticmethod
    def init_and_instantiate_RK_PF(
        portfolio_time_series, interest_free_rate, mode, nb_days, input_type
    ):
        """Create an instance of the RiskManagementPortfolio from the following args

        Args:
            portfolio_time_series (list): assets values or earnings through past days
            interest_free_rate (float): interest risk free rate
            mode (str) = "Monte-Carlo" -> method of the risk estimator : historic (unavailable), normal, Monte-Carlo
            nb_days (int) -> number of days
            input_type: str = "returns" -> assets times series type if values or returns (earnings)

        Returns:
            RiskManagementPortfolio: instance of the class RiskManagementPortfolio
        """
        import pandas as pd

        ####################################################
        ## Create the correlation matrix of the portfolio ##
        ####################################################
        cov_matrix = np.corrcoef(portfolio_time_series)

        ################################################
        ## Create the returns series of the portfolio ##
        ################################################

        array_time_series = np.array(
            [
                (pd.Series(asset) / pd.Series(asset).shift(1) - 1).tolist()
                for asset in portfolio_time_series
            ]
        )
        array_time_series = [
            [value for value in values if np.isnan(value) == False]
            for values in array_time_series
        ]

        ##########################################
        ## Assets volatilities of the portfolio ##
        ##########################################

        assets_volatilities = np.sqrt(np.var(array_time_series, axis=1))

        # TODO : use np.reshape lazy man
        assets_volatilities = np.array([[vol] for vol in assets_volatilities])

        assets_current_prices = np.array(
            [[asset[-1]] for asset in portfolio_time_series]
        )

        ############################
        ## Assets currents prices ##
        ############################

        return RiskManagementPortfolio(
            assets_current_prices=assets_current_prices,
            correlation_matrix=cov_matrix,
            assets_volatilities=assets_volatilities,
            mu=interest_free_rate,
            mode=mode,
            nb_days=nb_days,
            input_type=input_type,
        )

    def value_at_risk(self, nb_past_days, confiance, nb_days=1, nb_simulations=1000):
        """Return the value at risk according to the method in the attribute <mode>

        Args:
            nb_past_days (int): number of past days to keep from the original series
            confiance (float): confiance level for the computation of the value at risk
            nb_days (int, optional): number of days for the value at risk computation. Defaults to 1.
            nb_simulations (int, optional): number of simulations for the Monte-Carlo method. Defaults to 1000.

        """
        if self.mode == "Monte-Carlo":
            return self.value_at_risk_monte_carlo(
                confiance=confiance,
                nb_steps=nb_days,
                nb_simulations=nb_simulations,
            )
        if self.mode == "normal":
            return self.value_at_risk_normal(
                confiance=confiance,
                nb_days=nb_days,
            )

    def value_at_risk_normal(self, confiance, nb_days):
        """Returns the value at risk according to the normal method

        Args:
            confiance (float): confiance level for the computation of the value at risk
            nb_days (_type_): number of days for the value at risk computation. Defaults to 1.

        """
        return np.sqrt(nb_days) * self.value_at_risk_1_day(confiance)

    def value_at_risk_1_day(self, confiance):
        """Returns the value at risk according to the normal method for one day
        Args:
            confiance (float): confiance level for the computation of the value at risk

        """
        quantile = norm.isf(confiance)

        variances = self.assets_current_prices * self.assets_volatilities * quantile
        return -np.sqrt(
            np.sum(variances**2) + 2 * np.sum(self.correlation_matrix - 1)
        ) / np.sum(self.assets_current_prices)

    def value_at_risk_monte_carlo(
        self,
        confiance,
        nb_steps=1,
        nb_simulations=10000,
    ):
        """Returns the value at risk for Monte-Carlo method

        Args:
            confiance (float): confiance level for the computation of the value at risk
            nb_steps (int, ): number of steps like number of days. Defaults to 1.
            nb_simulations (int, optional): number of simulated paths. Defaults to 10000.

        """
        portfolio, end_price_simulations, index = self.monte_carlo_paths(
            confiance, nb_steps, nb_simulations
        )

        return end_price_simulations[index - 1] / np.sum(portfolio.current_price) - 1

    def monte_carlo_paths(self, confiance, nb_steps, nb_simulations):
        """Returns the portfolio (PricingGBMPortfolio), the end prices of the simulated paths and 
        the index for the Monte-Carlo paths

        Args:
            confiance (float): confiance level for the computation of the value at risk
            nb_steps (int, ): number of steps like number of days. Defaults to 1.
            nb_simulations (int, optional): number of simulated paths. Defaults to 10000.

        """
        if self.input_type != "value":
            portfolio = PricingGBMPortfolio(
                current_price=self.assets_current_prices,
                sigma=self.assets_volatilities,
                mu=self.mu,
                drift=self.drift,
                nb_steps=nb_steps,
                maturity=self.nb_days,
                nb_simulations=nb_simulations,
                correlation_matrix=self.correlation_matrix,
            )
        else:
            portfolio = PricingGBMPortfolio(
                current_price=self.assets_current_prices,
                sigma=self.assets_volatilities,
                mu=self.mu,
                drift=self.drift,
                nb_steps=nb_steps,
                maturity=self.nb_days,
                correlation_matrix=self.correlation_matrix,
                nb_simulations=nb_simulations,
            )

        #################################################
        ## Simulate paths from the PricingGBMPortfolio ##
        #################################################
        les_simulations = portfolio.paths_of_portfolio()

        end_price_simulations = sorted(les_simulations)
        index = int((1 - confiance) * len(end_price_simulations))

        return portfolio, end_price_simulations, index

if __name__=="__main__":
    ### Datas for the test ###
    print("----------------------------------------------------------------------")
    print("Welcome on the risk management module ")
    print("----------------------------------------------------------------------")

    import yfinance as yf


    sp500 = yf.Ticker("^GSPC")
    hist_sp500 = sp500.history("1y")
    hist_sp500["returns"] = hist_sp500.Close / hist_sp500.Close.shift(1) - 1
    returns_serie = hist_sp500.dropna().returns
    
    ### One asset ### 

    rkManageHistoric = RiskManagementOneAsset(returns_serie, input_type="returns")
    rkManageNormal = RiskManagementOneAsset(
        returns_serie,
        mode="normal",
    )
    rkManageMonteCarlo = RiskManagementOneAsset(
        hist_sp500.Close, "Monte-Carlo", returns=returns_serie, input_type="value"
    )

    print(
        f"VaR Historic 1 day : {rkManageHistoric.value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"VaR Normal 1 day  : {rkManageNormal.value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"VaR Monte-Carlo 1 day : {rkManageMonteCarlo.value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )

    print(
        f"Cond-VaR Historic 1 day : {rkManageHistoric.conditional_value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"Cond-VaR Normal 1 day  : {rkManageNormal.conditional_value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"Cond-VaR Monte-Carlo 1 day : {rkManageMonteCarlo.conditional_value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )

    ### Portfolio ### 

    import yfinance as yf
    import numpy as np

    ## Datas for the test
    sp500 = yf.Ticker("^GSPC")
    hist_sp500 = sp500.history("1y")
    aapl = yf.Ticker("AAPL")
    hist_aapl = aapl.history("1y")
    hist_sp500["returns"] = hist_sp500.Close / hist_sp500.Close.shift(1) - 1
    ###############################@


    pf_risk_Test = RiskManagementPortfolio.init_and_instantiate_RK_PF(
        portfolio_time_series=np.array(
            [
                hist_sp500.Close.tolist(),
                hist_aapl.Close.tolist(),
            ]
        ),
        interest_free_rate=0.05 / 252,
        mode="Monte-Carlo",
        nb_days=1,
        input_type="value",
    )
    print("Correlation matrix :")
    print(pf_risk_Test.correlation_matrix)
    print(
        "Value-at-risk 1 day Monte-Carlo",
        pf_risk_Test.value_at_risk(
            nb_past_days=1,  # not important here
            confiance=0.95,
        ),
    )

    pf_risk_Test = RiskManagementPortfolio.init_and_instantiate_RK_PF(
        portfolio_time_series=np.array(
            [
                hist_sp500.Close.tolist(),
                hist_aapl.Close.tolist(),
            ]
        ),
        interest_free_rate=0.05 / 252,
        mode="normal",
        nb_days=1,
        input_type="value",
    )
    print("Correlation matrix :")
    print(pf_risk_Test.correlation_matrix)
    print(
        "Value-at-risk 1 day normal method",
        pf_risk_Test.value_at_risk(
            nb_past_days=1,  # not important here
            confiance=0.95,
        ),
    )

    print("----------------------------------------------------------------------")
    print("End of the demo, bye.\n\n Author : Adrien Calas - Le23RayGorbella - le23rg@icloud.com")
    print("----------------------------------------------------------------------")