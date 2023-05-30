from quant23rg.pricingBS import PricingBS
from dataclasses import dataclass
from scipy import stats, optimize
import numpy as np
import math
from quant23rg.pricingCallEuropBS import PricingCallEuropBS

from quant23rg.pricingPutEuropBS import PricingPutEuropBS


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

if __name__=="__main__" : 
    print("----------------------------------------------------------------------")
    print("Welcome on the implied volatilities module ")
    print("----------------------------------------------------------------------")

    print("----------------------------------------------------------------------")
    print("A demo simulation will be launched")
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
    spy_opt["IV_call"] = spy_opt["IV_call"].apply(lambda x: float(x.replace("%", "")) / 100)
    spy_opt["IV_put"] = spy_opt["IV_put"].apply(lambda x: float(x.replace("%", "")) / 100)
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

    ivs = ImpliedVolatility(
        s0=s0,
        strike=call_example.Strike,
        dt=dt,
        interest_rate=0.05 * dt,  # fed rate  multiplied by our period of time
        volatility=0.1326,
    )

    ivs.show_implied_vol_and_compare(
        marketPrices_call=spy_opt["Last_call"].tolist(),
        marketPrices_put=spy_opt["Last_put"].tolist(),
        strikes=spy_opt.Strike.tolist(),
        marketVols=spy_opt.IV_call.tolist(),
    )
    print("----------------------------------------------------------------------")
    print("End of the demo, bye.\n\n Author : Adrien Calas - Le23RayGorbella - le23rg@icloud.com")
    print("----------------------------------------------------------------------")