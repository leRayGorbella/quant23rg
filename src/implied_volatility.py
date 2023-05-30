from pricingBS import PricingBS
from dataclasses import dataclass
from scipy import stats, optimize
import numpy as np
import math
from pricingCallEuropBS import PricingCallEuropBS

from pricingPutEuropBS import PricingPutEuropBS


@dataclass
class ImpliedVolatility(PricingBS):
    """Class for the calculation of the implied volatilities of european options

    s0: float -> Initial asset value
    strike: float ->  Strike of the derivated product
    dt: float -> interval of time in years
    interest_rate: float -> interest rate
    volatility: float -> volatility (sigma)

    """

    marketPrices_call = []
    marketPrices_put = []
    strikes = []
    marketVols = []

    def show_implied_vol_and_compare(
        self, marketPrices_call, marketPrices_put, strikes, marketVols=""
    ):
        """Calculate and show the implied volatilities based on the market prices of calls and puts
        marketPrices_call : calls prices
        marketPrices_put : puts prices
        strikes : strikes
        marketVols : implied volatilities sample from market
        """
        self.show_implied_volatility(
            marketPrices_call=marketPrices_call,
            marketPrices_put=marketPrices_put,
            strikes=strikes,
            marketVols=marketVols,
        )

    def show_implied_volatility(
        self, marketPrices_call, marketPrices_put, strikes, marketVols=""
    ):
        """Calculate and show the implied volatilities based on the market prices of calls and puts
        marketPrices_call : calls prices
        marketPrices_put : puts prices
        strikes : strikes
        marketVols : implied volatilities sample from market
        """
        import plotly.graph_objects as go

        ##############################
        ## Get implied volatilities ##
        ##############################     
        pts = self.implied_volatility(
            marketPrices_call=marketPrices_call,
            marketPrices_put=marketPrices_put,
            strikes=strikes,
        )

        ####################################
        ## Trace the implied volatilities ##
        ####################################
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=pts[0], y=pts[1], name="Simulated"),
        )

        if marketVols != "":
            ## add market IVs if in args
            fig.add_trace(go.Scatter(x=pts[0], y=marketVols, name="Market"))
        # show fig
        fig.show()

    def implied_volatility(
        self,
        marketPrices_call,
        strikes,
        marketPrices_put,
    ):
        """Calculate the implied volatilities based on the market prices of calls and puts
        marketPrices_call : calls prices
        marketPrices_put : puts prices
        strikes : strikes
        """
        
        ################################
        ## Initialize pricer for call ##
        ################################
        callPricer = PricingCallEuropBS(
            self.s0,
            strike=self.strike,
            dt=self.dt,
            interest_rate=self.interest_rate,
            volatility=self.volatility,
        )
        
        ###############################
        ## Initialize pricer for put ##
        ###############################
        putPricer = PricingPutEuropBS(
            self.s0,
            strike=self.strike,
            dt=self.dt,
            interest_rate=self.interest_rate,
            volatility=self.volatility,
        )
        
        ###########################
        ## Estimate IVs for call ##
        ###########################
        ivs_call = callPricer.implied_volatility_only_call(
            marketPrices=marketPrices_call, strikes=strikes
        )

        ##########################
        ## Estimate IVs for put ##
        ##########################
        ivs_put = putPricer.implied_volatility_only_put(
            marketPrices=marketPrices_put, strikes=strikes
        )

        ########################
        ## Find the ATM index ##
        ########################
        array = np.asarray(strikes)
        ix_strike_atm = (np.abs(array - self.s0)).argmin()

        ################################################################
        ## Concatenate list of IVs to get a correct estimation of IVs ##
        ################################################################

        ivs = ivs_put[:ix_strike_atm] + ivs_call[ix_strike_atm:]

        #######################
        ## Keep last good IV ##
        #######################               
        ivs_retraites = []

        for ix, iv in enumerate(ivs):
            # if iv ==-1 the pricer solver didnt find a solution for the strike 
            if iv == -1:
                lalist = ivs[:ix]
                lalist = [old_iv for old_iv in lalist if old_iv != -1]
                if lalist:
                    ivs_retraites.append(lalist[-1])
            else:
                ivs_retraites.append(iv)

        return (strikes, ivs_retraites)
