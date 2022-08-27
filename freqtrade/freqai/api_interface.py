import logging
import time
from typing import Any, Callable, Dict, Tuple

# import datetime
import dateutil.parser
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from freqtrade.exceptions import OperationalException

from freqtrade.freqai.data_drawer import FreqaiDataDrawer
import numpy as np
from typing import Callable
import datetime
import dateutil.parser
import time
import san
from san.graphql import execute_gql
from freqtrade.configuration import TimeRange
from san import Batch
from freqtrade.exchange import timeframe_to_seconds

logger = logging.getLogger(__name__)


class FreqaiAPI:
    """
    Class designed to enable FreqAI "poster" instances to share predictions via API with other
    FreqAI "getter" instances.
    :param: config: dict = user provided config containing api token and url information
    :param: data_drawer: FreqaiDataDrawer = persistent data storage associated with current
    FreqAI instance.
    :param: payload_fun: Callable = User defined schema for the "poster" FreqAI instance.
    Defined in the IFreqaiModel (inherited prediction model class such as CatboostPredictionModel)
    """

    def __init__(self, config: dict, data_drawer: FreqaiDataDrawer,
                 payload_func: Callable, mode: str):

        self.config = config
        self.freqai_config = config.get('freqai', {})
        self.api_token = self.freqai_config.get('freqai_api_token')
        self.api_base_url = self.freqai_config.get('freqai_api_url')
        self.post_url = f"{self.api_base_url}pairs"
        self.dd = data_drawer
        self.create_api_payload = payload_func
        if mode == 'getter':
            self.headers = {
                "X-BLOBR-KEY": self.api_token,
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Authorization": self.api_token,
                "Content-Type": "application/json"
            }
        self.api_dict: Dict[str, Any] = {}
        self.num_posts = 0
        self.santiment_api_key = self.freqai_config['feature_parameters'].get('santiment_api_key')

    def start_fetching_from_api(self, dataframe: DataFrame, pair: str) -> DataFrame:

        fetch_new = self.check_if_new_fetch_required(dataframe, pair)
        if fetch_new:
            response = self.fetch_all_pairs_from_api(dataframe)
            if not response:
                self.create_null_api_dict(pair)
            else:
                self.parse_response(response)
        self.make_return_dataframe(dataframe, pair)
        return self.dd.attach_return_values_to_return_dataframe(pair, dataframe)

    def post_predictions(self, dataframe: DataFrame, pair: str) -> None:
        """
        FreqAI "poster" instance will call this function to post predictions
        to an API. API schema is user defined in the IFreqaiModel.create_api_payload().
        API schema is flexible but must follow standardized method where
        f"{self.post_url}/{pair}" retrieves the predictions for current candle of
        the specified pair. Additionally, the schema must contain "returns"
        which defines the return strings expected by the getter.
        """
        subpair = pair.split('/')
        pair = f"{subpair[0]}{subpair[1]}"

        get_url = f"{self.post_url}/{pair}"

        payload = self.create_api_payload(dataframe, pair)

        if self.num_posts < len(self.config['exchange']['pair_whitelist']):
            response = requests.request("GET", get_url, headers=self.headers)
            self.num_posts += 1
            if response.json()['data'] is None:
                requests.request("POST", self.post_url, json=payload, headers=self.headers)
            else:
                requests.request("PATCH", get_url, json=payload, headers=self.headers)
        else:
            requests.request("PATCH", get_url, json=payload, headers=self.headers)

    def check_if_new_fetch_required(self, dataframe: DataFrame, pair: str) -> bool:

        if not self.api_dict:
            return True

        subpair = pair.split('/')
        coin = f"{subpair[0]}{subpair[1]}"
        candle_date = dataframe['date'].iloc[-1]
        ts_candle = candle_date.timestamp()
        ts_dict = dateutil.parser.parse(self.api_dict[coin]['updatedAt']).timestamp()

        if ts_dict < ts_candle:
            logger.info('Local dictionary outdated, fetching new predictions from API')
            return True
        else:
            return False

    def fetch_all_pairs_from_api(self, dataframe: DataFrame) -> dict:

        candle_date = dataframe['date'].iloc[-1]
        get_url = f"{self.post_url}"
        n_tries = 0
        ts_candle = candle_date.timestamp()
        ts_pair = ts_candle - 1
        ts_pair_oldest = int(ts_candle)

        while 1:
            response = requests.request("GET", get_url, headers=self.headers).json()['data']
            for pair in response:
                ts_pair = dateutil.parser.parse(pair['updatedAt']).timestamp()
                if ts_pair < ts_pair_oldest:
                    ts_pair_oldest = ts_pair
                    outdated_pair = pair['name']
            if ts_pair_oldest < ts_candle:
                logger.warning(
                    f'{outdated_pair} is not uptodate, waiting on API db to update before'
                    ' retrying.')
                n_tries += 1
                if n_tries > 5:
                    logger.warning(
                        'Tried to fetch updated DB 5 times with no success. Returning null values'
                        ' back to strategy')
                    return {}
                time.sleep(5)
            else:
                logger.info('Successfully fetched updated DB')
                break

        return response

    def parse_response(self, response_dict: dict) -> None:

        for coin_pred in response_dict:
            coin = coin_pred['name']
            self.api_dict[coin] = coin_pred   # {}
            # for return_str in coin_pred['returns']:
            #     coin_dict[coin][return_str] = coin_pred[return_str]

    def make_return_dataframe(self, dataframe: DataFrame, pair: str) -> None:

        subpair = pair.split('/')
        coin = f"{subpair[0]}{subpair[1]}"

        if coin not in self.api_dict:
            raise OperationalException(
                'Getter is looking for a coin that is not available at this API. '
                'Ensure whitelist only contains available coins.')

        if pair not in self.dd.model_return_values:
            self.set_initial_return_values(pair, self.api_dict[coin], len(dataframe.index))
        else:
            self.append_model_predictions_from_api(pair, self.api_dict[coin], len(dataframe.index))

    def set_initial_return_values(self, pair: str, response_dict: dict, len_df: int) -> None:
        """
        Set the initial return values to a persistent dataframe so that the getter only needs
        to retrieve a single data point per candle.
        """
        mrv_df = self.dd.model_return_values[pair] = DataFrame()

        for expected_str in response_dict['returns']:
            return_str = expected_str['name']
            mrv_df[return_str] = np.ones(len_df) * response_dict[return_str]

    def append_model_predictions_from_api(self, pair: str,
                                          response_dict: dict, len_df: int) -> None:
        """
        Function to append the api retrieved predictions to the return dataframe, but
        also detects if return dataframe should change size. This enables historical
        predictions to be viewable in FreqUI.
        """

        length_difference = len(self.dd.model_return_values[pair]) - len_df
        i = 0

        if length_difference == 0:
            i = 1
        elif length_difference > 0:
            i = length_difference + 1

        mrv_df = self.dd.model_return_values[pair] = self.dd.model_return_values[pair].shift(-i)

        for expected_str in response_dict['returns']:
            return_str = expected_str['name']
            mrv_df[return_str].iloc[-1] = response_dict[return_str]

        if length_difference < 0:
            prepend_df = pd.DataFrame(
                np.zeros((abs(length_difference) - 1, len(mrv_df.columns))), columns=mrv_df.columns
            )
            mrv_df = pd.concat([prepend_df, mrv_df], axis=0)

    def create_null_api_dict(self, pair: str) -> None:
        """
        Set values in api_dict to 0 and return to user. This is only used in case the API is
        unresponsive, but  we still want FreqAI to return to the strategy to continue handling 
        open trades.
        """
        subpair = pair.split('/')
        pair = f"{subpair[0]}{subpair[1]}"

        for expected_str in self.api_dict[pair]['returns']:
            return_str = expected_str['name']
            self.api_dict[pair][return_str] = 0

    # Santiment

    def create_metric_update_tracker(self) -> list:

        metrics_to_get = ['active_addresses_1h', "daily_active_addresses", "transaction_volume",
                          "active_withdrawals_5m"]  # "transaction_volume", "active_withdrawals_5m"]
        slugs = ['bitcoin', 'ethereum']
        metric_slug = []

        for metric in metrics_to_get:
            for slug in slugs:
                metric_slug.append(f'{metric}/{slug}')
                self.dd.metric_update_tracker[f'{metric}/{slug}'] = {
                    'timestamp': 0, 'minInterval': self.config['timeframe']}
        self.metric_slug_final = metric_slug.copy()

        return metric_slug

    def get_and_store_last_updated_timestamp(self, metric, slug):

        execute_str = ('{'
                       f'getMetric(metric: "{metric}"){{'
                       f'lastDatetimeComputedAt(slug: "{slug}")'
                       r'}}')
        res = execute_gql(execute_str)
        time_updated = res['getMetric']['lastDatetimeComputedAt']
        self.dd.metric_update_tracker[f'{metric}/{slug}']['timestamp'] = dateutil.parser.parse(
            time_updated).timestamp()

    def check_if_needs_update(self, metric, slug) -> bool:

        updated_ts = self.dd.metric_update_tracker[f'{metric}/{slug}']['timestamp']
        interval = self.dd.metric_update_tracker[f'{metric}/{slug}']['minInterval']
        to_update_ts = updated_ts + timeframe_to_seconds(interval)
        now_timestamp = datetime.datetime.now().timestamp()

        if now_timestamp < to_update_ts:
            logger.info(f'Not pulling new data yet for {metric}/{slug}')
            return False
        else:
            logger.info(f'Pulling new value for {metric}/{slug}')
            self.dd.metric_update_tracker[f'{metric}/{slug}'] = to_update_ts
            return True

    def prepare_historic_dataframe(self, metric, slug, start, stop) -> bool:
        projects = san.get("projects/all")
        skip = False
        if not projects['slug'].str.contains(slug).any():
            logger.warning(f'{slug} not in projects list.')
            return True

        metrics = san.available_metrics_for_slug(slug)

        if metric not in metrics:
            logger.warning(f'{metric} not in available {slug} metrics list. Skipping.')
            self.metric_slug_final.remove(f'{metric}/{slug}')
            return True

        meta_dict = san.metadata(
            metric,
            arr=['isAccessible', 'isRestricted', 'restrictedFrom', 'restrictedTo', 'minInterval']
            )

        if not meta_dict['isAccessible']:
            logger.warning(f'{metric} not accessible with current plan. Skipping.')
            self.metric_slug_final.remove(f'{metric}/{slug}')
            return True

        if meta_dict['isRestricted']:
            restricted_from = dateutil.parser.parse(meta_dict['restrictedFrom'])  # .timestamp()
            restricted_to = dateutil.parser.parse(meta_dict['restrictedTo'])  # .timestamp()
            if restricted_from.timestamp() > start.timestamp():
                logger.warning(f'Not enough data at start for {metric}/{slug}')
                self.metric_slug_final.remove(f'{metric}/{slug}')
                return True
            if restricted_to.timestamp() < stop.timestamp():
                logger.warning(f'Not enough data at end for {metric}/{slug}')
                self.metric_slug_final.remove(f'{metric}/{slug}')
                return True

        self.dd.metric_update_tracker[f'{metric}/{slug}']['minInterval'] = meta_dict['minInterval']

        return skip

    def download_external_data_from_santiment(self, timerange: TimeRange) -> None:

        key = next(iter(self.dd.historic_data))
        build_historic_df = False
        if self.dd.historic_external_data.empty:
            build_historic_df = True
            self.dd.historic_external_data['datetime'] = self.dd.historic_data[key][self.config['timeframe']]['date']
            metric_slug = self.create_metric_update_tracker()
        else:
            metric_slug = self.metric_slug_final
        start = datetime.datetime.utcfromtimestamp(timerange.startts)
        stop = datetime.datetime.utcfromtimestamp(timerange.stopts)

        san.ApiConfig.api_key = self.santiment_api_key

        batch = Batch()

        # build batch to send to sanpi
        for key in metric_slug:
            slug = key.split('/')[1]
            metric = key.split('/')[0]

            if build_historic_df:
                skip = self.prepare_historic_dataframe(metric, slug, start, stop)
                if skip:
                    continue
                self.get_and_store_last_updated_timestamp(metric, slug)
            else:
                if not self.check_if_needs_update(metric, slug):
                    continue

            batch.get(
                f'{metric}/{slug}',
                from_date=start,
                to_date=stop,
                interval=self.config['timeframe'],
            )

        response = batch.execute()
        metric_dict = dict(zip(self.metric_slug_final, response))

        if build_historic_df:
            for metric in self.metric_slug_final:
                metric_dict[metric].rename(columns={'value': metric}, inplace=True)
                self.dd.historic_external_data = pd.merge(
                    self.dd.historic_external_data, metric_dict[metric], how='outer', on='datetime').ffill()
        else:
            for metric in self.metric_slug_final:
                metric_dict[metric].rename(columns={'value': metric}, inplace=True)
            pd.concat([metric_dict[key] for key in metric_dict], axis=0)
            df = pd.DataFrame.from_dict(metric_dict)
            self.dd.historic_external_data = pd.concat(
                [self.dd.historic_external_data, df], axis=0, ignore_index=True)
