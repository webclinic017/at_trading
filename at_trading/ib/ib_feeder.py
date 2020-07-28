import threading
import time
from typing import List, Dict
import pandas as pd
from ibapi.contract import Contract

from at_trading.ib.ib_api import ATIBApi, gen_contract, gen_contract_id_only, TickType


class ATIBFeeder(object):
    def __init__(self, env='prod'):
        self.instrument_dict = None
        self.env = env
        self.app = ATIBApi()

    def connect(self):
        if self.env == 'dev':
            self.app.connect_local(port=7497)
        else:
            self.app.connect_local()
        api_thread = threading.Thread(target=self.app.run, daemon=True)
        api_thread.start()
        self.app.wait_till_connected()

    def run_ib_api(self, func_name: str, func_param: Dict = None):
        if func_param is None:
            param = {}
        else:
            param = func_param
        obj_func = getattr(self.app, func_name)
        for inst_key, inst in iter(self.instrument_dict.items()):
            obj_func(inst, **param)

    def populate_instrument_dict(self, group_name):
        self.instrument_dict = self.get_instrument_by_group(group_name)

    # TODO: move this to strategy/signals
    def get_instrument_by_group(self, group_name):
        result_instrument_dict = {}
        if group_name == 'vol_futures':
            # first get spx index then get monthly vix futures
            result_instrument_dict['spx'] = gen_contract(
                'SPX', 'IND', 'CBOE', 'USD')
            result_vix_futures = self.app.req_contract_details_futures('VIX', exchange='CFE',
                                                                       summary_only=True)
            result_vix_futures_df = pd.DataFrame(result_vix_futures).sort_values(
                'contract.lastTradeDateOrContractMonth')
            # take 5 front month contract
            result_vix_futures_df = result_vix_futures_df[result_vix_futures_df['marketName'] == 'VX'].head(5)
            for irow, data in result_vix_futures_df.iterrows():
                result_instrument_dict[data['contract.localSymbol'].lower()] = gen_contract_id_only(data['contract.conId'])
        return result_instrument_dict


if __name__ == '__main__':
    feeder = ATIBFeeder()
    feeder.connect()
    feeder.populate_instrument_dict('vol_futures')
    feeder.run_ib_api('req_tick_by_tick_data', {'tick_type': TickType.mid})
    time.sleep(1)
    # app.cancel_mkt_data_all()
    feeder.app.disconnect()
