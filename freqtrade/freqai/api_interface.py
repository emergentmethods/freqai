import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

# import datetime
import dateutil.parser
import numpy as np
import pandas as pd
import requests
import san
from pandas import DataFrame
from san import AsyncBatch
from san.graphql import execute_gql

from freqtrade.configuration import TimeRange
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


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
        self.santiment_api_key = self.freqai_config.get(
            'santiment_config', {}).get('santiment_api_key')
        self.metric_slug_temporary: list = []
        self.moving_avg_window = self.freqai_config.get(
            'santiment_config', {}).get('moving_average_window', 5)

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

    # SANTIMENT API INTERFACE
    def graphql_timeseries(self, metric, slug, start, stop, interval):
        execute_str = ('{'
                       f'getMetric(metric: "{metric}"){{'
                       f'timeseriesData(slug: "{slug}"'
                       f'from: "{start}"'
                       f'to: "{stop}"'
                       f'interval: "{interval}"){{'
                       r'datetime '
                       r'value}'
                       r'}}')

        return execute_gql(execute_str)

    def create_metric_update_tracker(self) -> list:

        metrics_to_get = self.freqai_config['santiment_config']['metrics']
        # ["daily_active_addresses", "transaction_volume",
        #                   "active_withdrawals_5m", 'active_addresses_1h']
        slugs = self.freqai_config['santiment_config']['slugs']  # ['bitcoin', 'ethereum']
        metric_slug = []

        for metric in metrics_to_get:
            for slug in slugs:
                metric_slug.append(f'{metric}/{slug}')
                self.dd.metric_update_tracker[f'{metric}/{slug}'] = {
                    'datetime': datetime.now(tz=timezone.utc),
                    'minInterval': self.config['timeframe']}
        self.metric_slug_final = metric_slug.copy()

        return metric_slug

    def get_and_store_last_updated_timestamp(self, metric, slug):

        # execute_str = ('{'
        #                f'getMetric(metric: "{metric}"){{'
        #                f'lastDatetimeComputedAt(slug: "{slug}")'
        #                r'}}')
        # res = execute_gql(execute_str)
        time_updated = datetime.now(tz=timezone.utc)  # res['getMetric']['lastDatetimeComputedAt']
        self.dd.metric_update_tracker[f'{metric}/{slug}']['datetime_updated'] = time_updated

    def check_if_needs_update(self, metric, slug):

        try:
            updated_dt = self.dd.metric_update_tracker[f'{metric}/{slug}']['datetime_updated']
        except TypeError:
            logger.info('someshit happening')
        interval = self.dd.metric_update_tracker[f'{metric}/{slug}']['minInterval']
        to_update_dt = updated_dt + timedelta(seconds=timeframe_to_seconds(interval))
        now_dt = datetime.now(timezone.utc)

        if now_dt < to_update_dt:
            until_update = (to_update_dt - now_dt)  # / 3600
            logger.info(
                f'Not pulling new data yet for {metric}/{slug}, still {until_update}')
            return None
        else:
            logger.info(f'Trying to pulling new value for {metric}/{slug}')
            # self.dd.metric_update_tracker[f'{metric}/{slug}']['datetime'] = to_update_dt
            self.metric_slug_temporary.append(f'{metric}/{slug}')
            return updated_dt

    def prepare_historic_dataframe(self, metric: str, slug: str,
                                   start: datetime, stop: datetime) -> bool:

        skip = False
        if slug in ["gold", "s-and-p-500"]:
            metrics = ["price_usd"]
        else:
            if metric == "price_usd":
                self.metric_slug_final.remove(f'{metric}/{slug}')
                return True
            projects = san.get("projects/all")
            if not projects['slug'].str.contains(slug).any():
                logger.warning(f'{slug} not in projects list.')
                self.metric_slug_final.remove(f'{metric}/{slug}')
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
            if meta_dict['restrictedFrom']:
                restricted_from = dateutil.parser.parse(meta_dict['restrictedFrom'])
            else:
                restricted_from = None
            if meta_dict['restrictedTo']:
                restricted_to = dateutil.parser.parse(meta_dict['restrictedTo'])
            else:
                restricted_to = None
            if restricted_from and restricted_from > start:
                logger.warning(f'Not enough data at start for {metric}/{slug}')
                self.metric_slug_final.remove(f'{metric}/{slug}')
                return True
            if restricted_to and restricted_to < stop:
                logger.warning(f'Not enough data at end for {metric}/{slug}')
                self.metric_slug_final.remove(f'{metric}/{slug}')
                return True

        self.dd.metric_update_tracker[f'{metric}/{slug}']['minInterval'] = meta_dict['minInterval']

        return skip

    def download_external_data_from_santiment(self, dk: FreqaiDataKitchen,
                                              timerange: TimeRange = TimeRange()) -> None:

        begin = time.time()
        build_historic_df = False
        if self.dd.historic_external_data.empty:
            build_historic_df = True
            pair = self.find_pair_with_most_data()
            hist_df = self.dd.historic_data[pair][self.config['timeframe']]
            self.dd.historic_external_data['datetime'] = hist_df['date']
            metric_slug = self.create_metric_update_tracker()
            start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
            stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
        else:
            metric_slug = self.metric_slug_final
            self.metric_slug_temporary = []
            stop = datetime.now(timezone.utc)

        san.ApiConfig.api_key = self.santiment_api_key

        batch = AsyncBatch()

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
                start = self.check_if_needs_update(metric, slug)
                if not start:
                    continue

            batch.get(
                f'{metric}/{slug}',
                from_date=start,
                to_date=stop,
                # transform={"type": "moving_average",
                #            "moving_average_base": self.moving_avg_window},
                interval=self.dd.metric_update_tracker[f'{metric}/{slug}']['minInterval']
            )

        if batch.queries:
            response = batch.execute()
        else:
            logger.info('Nothing to fetch externally, ffilling dataframe')
            self.ffill_historic_values()
            return

        if build_historic_df:
            self.build_historic_external_data(response, dk)
        else:
            self.append_new_row_to_historic_external_data(response, dk)

        end = time.time()
        logger.info(f'Total time spent fetching Santiment data {end-begin:.2f} seconds')

    def build_historic_external_data(self, response: dict, dk: FreqaiDataKitchen):
        """
        Build the persistent historic_external_data dataframe using user defined
        training duration.
        """
        logger.info(f'External successfully fetching metrics for {self.metric_slug_final}')
        metric_dict = dict(zip(self.metric_slug_final, response))
        to_remove = []
        for metric in self.metric_slug_final:
            metric_dict[metric].rename(columns={'value': metric}, inplace=True)
            # minInterval = self.dd.metric_update_tracker[metric]['minInterval']
            # metric_dict[metric].index = (metric_dict[metric].index +
            #               timedelta(seconds=timeframe_to_seconds(minInterval)))
            if (metric_dict[metric][metric] == 0.0).all():
                logger.info(f'{metric} is all zeros, removing it from metric '
                            'list and never fetching again.')
                to_remove.append(metric)
                continue
            dt = datetime.fromtimestamp(
                int(metric_dict[metric].iloc[-1].name.timestamp()), timezone.utc)
            self.dd.metric_update_tracker[f'{metric}']['datetime_updated'] = dt
            # Santiment stores their data at beginning of candle that contained the data.
            # Everyone else stores it at end (As they should) so we fix santiment data for them.

            self.dd.historic_external_data = pd.merge(
                self.dd.historic_external_data, metric_dict[metric],
                how='left', on='datetime'
                ).ffill()
        for item in to_remove:
            self.metric_slug_final.remove(item)

    def append_new_row_to_historic_external_data(self, response, dk):
        """
        Append new row to the historic_external_data dataframe. This
        function checks the response from santiment if there was new data
        or not, and ffills values accordingly.

        Note: santiment data is stored at the back of the bucket, not the front.
        TODO consider storing all their values 1 "minInterval" forward (front
        of the bucket). But it should be ensured that this minInterval is properly
        handled across all metrics.
        """
        metric_dict = dict(zip(self.metric_slug_temporary, response))
        hist_df = self.dd.historic_external_data
        metric_df = pd.DataFrame(np.nan, index=hist_df.index[-1:], columns=hist_df.columns)
        hist_df = pd.concat([hist_df, metric_df], ignore_index=True, axis=0)

        for metric in self.metric_slug_temporary:

            if metric_dict[metric].empty:
                logger.info(f'Unable to pull new data for {metric}, ffilling')
                continue
            dt = datetime.fromtimestamp(
                int(metric_dict[metric].iloc[-1].name.timestamp()), timezone.utc)

            # santiment stores data at beginning of candle where it was computed.
            # minInterval = self.dd.metric_update_tracker[metric]['minInterval']
            # dt += timedelta(seconds=timeframe_to_seconds(minInterval))

            if dt == self.dd.metric_update_tracker[f'{metric}']['datetime_updated']:
                logger.info(
                    f'no new data available for {metric} from santiment, waiting since {dt}')
                continue

            metric_dict[metric].rename(columns={'value': metric}, inplace=True)

            self.dd.metric_update_tracker[f'{metric}']['datetime_updated'] = dt
            self.dd.metric_update_tracker[f'{metric}']['datetime_grabbed'] = datetime.now(
                tz=timezone.utc)

            idx = hist_df[hist_df['datetime'] >= dt].index
            hist_df[metric].iloc[idx] = metric_dict[metric].iloc[-1]
            logger.info(f'Successfully pulled new data for {metric}')
        hist_df['datetime'].iloc[-1] = hist_df['datetime'].iloc[-2] + timedelta(
            seconds=timeframe_to_seconds(self.config['timeframe']))
        hist_df.fillna(method='ffill', inplace=True)
        self.dd.historic_external_data = hist_df

    def ffill_historic_values(self):
        """
        Only used if no queries were actually made to santiment.
        """
        index = self.dd.historic_external_data.index[-1:]
        columns = self.dd.historic_external_data.columns
        df = self.dd.historic_external_data
        new_date = df['datetime'].iloc[-1] + \
            timedelta(seconds=timeframe_to_seconds(self.config['timeframe']))
        last_row = pd.DataFrame(np.nan, index=index, columns=columns)
        self.dd.historic_external_data = pd.concat(
            [self.dd.historic_external_data, last_row], ignore_index=True, axis=0)
        self.dd.historic_external_data.iloc[-1] = df.iloc[-1]
        self.dd.historic_external_data['datetime'].iloc[-1] = new_date
        return

    def find_pair_with_most_data(self):
        length = 0
        max_pair = ''
        for pair in self.dd.historic_data:
            pair_len = len(self.dd.historic_data[pair][self.config["timeframe"]])
            if pair_len > length:
                max_pair = pair
                length = pair_len

        return max_pair

    def shift_and_concatenate_df(self, dataframe: pd.DataFrame, shifts: int) -> pd.DataFrame:
        columns = []
        for metric in dataframe.columns:
            for shift in range(1, shifts + 1):
                columns.append(f'{metric}_shift_{shift}')
        shifted_df = DataFrame(
            np.zeros((len(dataframe.index), len(dataframe.columns) * shifts)),
            columns=columns, index=dataframe.index)
        base_tf_seconds = timeframe_to_seconds(self.config['timeframe'])
        for metric in dataframe.columns:
            minInt = self.dd.metric_update_tracker[f'{metric}']['minInterval']
            num_base_candles = timeframe_to_seconds(minInt) / base_tf_seconds
            for shift in range(1, shifts + 1):
                candle_shift = int(shift * num_base_candles)
                shifted_df[f'{metric}_shift_{shift}'] = dataframe[metric].shift(candle_shift)

        dataframe = pd.concat([dataframe, shifted_df], axis=1)

        return dataframe
