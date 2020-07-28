import threading
from unittest import TestCase

from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.contract import Contract

from at_trading.ib.ib_api import ATIBApi
import time


class TestATIBApi(TestCase):
    def setUp(self) -> None:
        """
        TODO: use mock later, for now, just do simple setup
        """
        self.app = ATIBApi()
        self.app.connect_local(port=7497)
        api_thread = threading.Thread(target=self.app.run, daemon=True)
        api_thread.start()
        self.account_id = 'DU1589832'
        self.app.wait_till_connected()

    def tearDown(self) -> None:
        time.sleep(3)
        # app.cancel_mkt_data_all()
        self.app.disconnect()

    # TODO: we might not be checking the data returned, but list all the functions
    # that are implemented
    def test_req_pnl_blocking(self):
        result = self.app.req_pnl_blocking(self.account_id, '')
        print(result)

    def test_req_pnl_single_blocking(self):
        contract = Contract()
        contract.symbol = "AAPL"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = 'USD'

        result = self.app.req_contract_details(contract)
        contract_id = result[0]['contract']['conId']
        contract_pnl = self.app.req_pnl_single_blocking(self.account_id, '', contract_id)
        print(contract_pnl)

    def test_req_account_updates_blocking(self):
        result = self.app.req_account_updates_blocking(self.account_id)
        print(result)

    def test_req_account_summary_blocking(self):
        result = self.app.req_account_summary_blocking('All', AccountSummaryTags.AllTags)
        print(result)

    def test_req_positions_blocking(self):
        self.app.req_positions_blocking()
