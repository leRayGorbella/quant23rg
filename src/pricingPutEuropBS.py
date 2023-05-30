from pricingBS import PricingBS
from dataclasses import dataclass
from scipy import stats, optimize
import numpy as np
import math

# from pricingCallEuropBS import PricingCallEuropBS


@dataclass
class PricingPutEuropBS(PricingBS):
    """Pricer for european option put
    s0: float -> Initial asset value
    strike: float ->  Strike of the derivated product
    dt: float -> interval of time in years
    interest_rate: float -> interest rate
    volatility: float -> volatility (sigma)

    """

    def payoff_sigma_fixed(self):
        """Return fair price according to (BS-Merton formula) of an european put"""
        return -self.s0 * stats.norm.cdf(-self.d1) + self.strike * math.exp(
            -self.interest_rate * self.dt
        ) * stats.norm.cdf(-self.d2)

    def payoff_sigma_strike_unfixed(
        self,
    ):
        """Helper function for the function get_sigma

        Returns:
            lambda (sigma, strike) function
        """
        d1_sigma = lambda sigma, strike: (
            math.log(self.s0 / strike)
            + (self.interest_rate + (sigma**2) / 2) * self.dt
        ) / (sigma * math.sqrt(self.dt))
        d2_sigma = lambda sigma, strike: d1_sigma(
            sigma=sigma, strike=strike
        ) - sigma * math.sqrt(self.dt)
        payoff_sigma_unfixed = lambda sigma, strike: -self.s0 * stats.norm.cdf(
            -d1_sigma(sigma=sigma, strike=strike)
        ) + strike * math.exp(-self.interest_rate * self.dt) * stats.norm.cdf(
            -d2_sigma(sigma=sigma, strike=strike)
        )

        return payoff_sigma_unfixed

    def implied_volatility_only_put(self, marketPrices, strikes):
        """Returns only implied volatilities of the puts, will be merged with the calls imlpied volatilities

        Args:
            marketPrices (list): put prices according to their strikes
            strikes (list): strikes

        Returns:
            Implied Volatilities of the puts
            -> Best sigmas to minimize the gap between Black&Scholes formula and market prices of the puts
        """
        return [
            self.get_sigma(price, strike)
            for price, strike in zip(marketPrices, strikes)
        ]

    def get_sigma(self, price, strike):
        """Best sigma (implied volatility) candidate to minimize the gap between the put market price and the Black&Scholes formula.

        Args:
            price (float): put market price
            strike (float): strike of the put

        Returns:
            float: best sigma (implied volatility) candidate to minimize the gap between the put market price and the Black&Scholes formula.
        """
        res, infodict, ier, mesg = optimize.fsolve(
            lambda sigma, strike: self.payoff_sigma_strike_unfixed()(sigma, strike)
            - price,
            self.volatility,
            args=(strike,),
            full_output=True,
        )
        res = res[0] if ier == 1 else -1
        return res


if __name__ == "__main__":
    ### Pricing European Options ###
    print("----------------------------------------------------------------------")
    print("Welcome on the european put pricer module ")
    print("----------------------------------------------------------------------")

    import pandas as pd

    spy_opt = pd.read_csv(
        "ressources/spy-options.csv",
    ).dropna()
    spy_opt.columns = [
        col + "_call" if ".1" not in col and col != "Strike" else col
        for col in spy_opt.columns
    ]
    spy_opt.columns = [col.replace(".1", "_put") for col in spy_opt.columns]
    spy_opt["IV_call"] = spy_opt["IV_call"].apply(
        lambda x: float(x.replace("%", "")) / 100
    )
    spy_opt["IV_put"] = spy_opt["IV_put"].apply(
        lambda x: float(x.replace("%", "")) / 100
    )
    spy_opt["Last_call"] = spy_opt["Last_call"].apply(lambda x: float(x))
    spy_opt["Last_put"] = spy_opt["Last_put"].apply(lambda x: float(x))
    call_example = spy_opt.sort_values("Volume_call", ascending=False).iloc[0]

    # Last_call              27.9
    # Bid_call              27.85
    # Ask_call              28.18
    # Change_call            27.9
    # Volume_call             962
    # Open Int_call         2,741
    # IV_call              0.1815
    # Last Trade_call    05/26/23
    # Strike                410.0
    # Last_put              12.45
    # Bid_put               12.24
    # Ask_put                12.4
    # Change_put            12.45
    # Volume_put            1,234
    # Open Int_put         20,292
    # IV_put               0.1831
    # Last Trade_put     05/26/23
    # Name: 47, dtype: object

    s0 = 420.02  # 29 may 2023
    dt = 146 / 252  # in years, expiration date : 20 october 2023
    call_pricing = PricingPutEuropBS(
        s0=s0,
        strike=call_example.Strike,
        dt=dt,
        interest_rate=0.05 * dt,  # fed rate multiplied by our period of time
        volatility=0.1326,
    )
    print(call_pricing.payoff_sigma_fixed())
    print("----------------------------------------------------------------------")
    print("End of the demo, bye.\n\n Author : Adrien Calas - Le23RayGorbella - le23rg@icloud.com")
    print("----------------------------------------------------------------------")