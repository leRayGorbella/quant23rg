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
