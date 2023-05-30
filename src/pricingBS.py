from dataclasses import dataclass
import math 

@dataclass(init=True, repr=True)
class PricingBS:

    """Master class for pricing derivated products
    
    s0: float -> Initial asset value
    strike: float ->  Strike of the derivated product
    dt: float -> interval of time in years
    interest_rate: float -> interest rate  
    volatility: float -> volatility (sigma)  
    """

    s0: float  # Initial asset value
    strike: float  # Strike of the derivated product
    dt: float  # interval of time in years
    interest_rate: float  # interest rate if const
    volatility: float  # volatility (sigma) if const

    def __post_init__(self):
        self.d1: float = (
            math.log(self.s0 / self.strike)
            + (self.interest_rate + (self.volatility**2) / 2) * self.dt
        ) / (self.volatility * math.sqrt(self.dt))
        self.d2: float = self.d1 - self.volatility * math.sqrt(self.dt)
