import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

# import datetime
import dateutil.parser
import numpy as np
import pandas as pd
# import requests
import san
from pandas import DataFrame
from san import AsyncBatch
from san.graphql import execute_gql

from freqtrade.configuration import TimeRange
# from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class FreqaiAPI:
    """
    Class designed to interact with the santiment data source
    """

    def __init__(self, config: dict, data_drawer: FreqaiDataDrawer):

        self.config = config
        self.freqai_config = config.get('freqai', {})
        self.dd = data_drawer
        self.api_dict: Dict[str, Any] = {}
        self.num_posts = 0
        self.santiment_api_key = self.freqai_config.get(
            'santiment_config', {}).get('santiment_api_key')
        self.metric_slug_temporary: list = []
        self.moving_avg_window = self.freqai_config.get(
            'santiment_config', {}).get('moving_average_window', 5)

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
        slugs = self.freqai_config['santiment_config']['slugs']
        metric_slug = []

        for metric in metrics_to_get:
            for slug in slugs:
                metric_slug.append(f'{metric}/{slug}')
                # self.dd.metric_update_tracker[f'{metric}/{slug}'] = {
                #     'datetime_updated': datetime.now(tz=timezone.utc),
                #     'minInterval': self.config['timeframe'],
                #     'datetime_grabbed': datetime.now(tz=timezone.utc)}
        self.dd.metric_slug_final = metric_slug.copy()

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
            self.metric_slug_temporary.append(f'{metric}/{slug}')
            return updated_dt

    def prepare_historic_dataframe(self, metric: str, slug: str,
                                   start: datetime, stop: datetime) -> bool:

        skip = False
        if slug in ["gold", "s-and-p-500"]:
            metrics = ["price_usd"]
        else:
            if metric == "price_usd":
                self.dd.metric_slug_final.remove(f'{metric}/{slug}')
                return True
            projects = san.get("projects/all")
            if not projects['slug'].str.contains(slug).any():
                logger.warning(f'{slug} not in projects list.')
                self.dd.metric_slug_final.remove(f'{metric}/{slug}')
                return True
            metrics = san.available_metrics_for_slug(slug)

        if metric not in metrics:
            logger.warning(f'{metric} not in available {slug} metrics list. Skipping.')
            self.dd.metric_slug_final.remove(f'{metric}/{slug}')
            return True

        meta_dict = san.metadata(
            metric,
            arr=['isAccessible', 'isRestricted', 'restrictedFrom', 'restrictedTo', 'minInterval']
            )

        if not meta_dict['isAccessible']:
            logger.warning(f'{metric} not accessible with current plan. Skipping.')
            self.dd.metric_slug_final.remove(f'{metric}/{slug}')
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
                self.dd.metric_slug_final.remove(f'{metric}/{slug}')
                return True
            if restricted_to and restricted_to < stop:
                logger.warning(f'Not enough data at end for {metric}/{slug}')
                self.dd.metric_slug_final.remove(f'{metric}/{slug}')
                return True

        minInt_sec = timeframe_to_seconds(meta_dict['minInterval'])
        stratInt_sec = timeframe_to_seconds(self.config["timeframe"])
        maxInt = self.freqai_config['santiment_config']['maxInt']
        maxInt_sec = timeframe_to_seconds(self.freqai_config['santiment_config']['maxInt'])
        if minInt_sec > maxInt_sec:
            logger.warning(f'Removed {metric}/{slug} since its minimum interval was greater'
                           f' than user requested {meta_dict["minInterval"]} > {maxInt}')
            self.dd.metric_slug_final.remove(f'{metric}/{slug}')
            return True
        if minInt_sec < stratInt_sec:
            logger.warning(f'{metric}/{slug} since its minimum interval was less'
                           f' than  than strat tf. {minInt_sec} < {stratInt_sec}. '
                           "Changing minInterval to strat tf.")
            meta_dict['minInterval'] = self.config['timeframe']
            # self.dd.metric_slug_final.remove(f'{metric}/{slug}')
            # return True

        self.dd.metric_update_tracker[f'{metric}/{slug}'] = {}
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
            metric_slug = self.dd.metric_slug_final
            self.metric_slug_temporary = []
            start = None
            # stop = self.dd.current_candle
            stop = datetime.now(timezone.utc)

        san.ApiConfig.api_key = self.santiment_api_key

        batch = AsyncBatch()

        get_many_dict = self.build_get_many_dict(metric_slug, build_historic_df, start, stop)

        for metric in get_many_dict.keys():
            slug1 = get_many_dict[metric]["slugs"][0]
            start = datetime.fromtimestamp(
                get_many_dict[metric]["start"], tz=timezone.utc) - timedelta(hours=24)
            batch.get_many(
                metric,
                slugs=get_many_dict[metric]["slugs"],
                from_date=start,
                to_date=stop,
                # transform={"type": "moving_average",
                #            "moving_average_base": self.moving_avg_window},
                interval=self.dd.metric_update_tracker[f'{metric}/{slug1}']['minInterval']

            )

        if batch.queries:
            try:
                response = batch.execute()
            except Exception as e:
                # response = None
                logger.exception(f"Santiment api fetch encountered error {e}")

        else:
            logger.info('Nothing to fetch externally, ffilling dataframe')
            self.ffill_historic_values()
            return

        if build_historic_df and response is not None:
            self.build_historic_external_data(response, get_many_dict)
        elif response is not None:
            self.append_new_row_to_historic_external_data(response, get_many_dict)

        end = time.time()
        logger.info(f'Total time spent fetching Santiment data {end-begin:.2f} seconds')

    def build_get_many_dict(self, metric_slug, build_historic_df, start, stop):
        # build dict for metric: [slugs] to feed to sanpy get_many
        get_many_dict = {}
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

            if metric not in get_many_dict.keys():
                get_many_dict[metric] = {"slugs": [slug], "start": start.timestamp()}
            else:
                get_many_dict[metric]["slugs"].append(slug)
                if start.timestamp() < get_many_dict[metric]["start"]:
                    # get data from earliest needed time on a metric (Across slugs)
                    get_many_dict[metric]["start"] = start.timestamp()

        return get_many_dict

    def build_historic_external_data(self, response: dict, get_many_dict):
        """
        Build the persistent historic_external_data dataframe using user defined
        training duration.
        """
        logger.info(f'External successfully fetching metrics for {get_many_dict.keys()}')
        metric_dict = dict(zip(get_many_dict.keys(), response))
        # metric_dict = dict(zip(self.dd.metric_slug_final, response))
        to_remove = []
        for metric in get_many_dict.keys():
            # metric_dict[metric].rename(columns={'value': metric}, inplace=True)
            for slug in metric_dict[metric].columns:
                metric_slug = f"{metric}/{slug}"
                metric_dict[metric].rename(columns={f"{slug}": metric_slug}, inplace=True)
                if ((metric_dict[metric][metric_slug] == 0.0).all() or
                        metric_dict[metric][metric_slug].isnull().all()):
                    logger.info(f'{metric_slug} is all zeros or NaNs, removing it from metric '
                                'list and never fetching again.')
                    to_remove.append(metric_slug)
                    continue
                dt = datetime.fromtimestamp(
                    int(metric_dict[metric].iloc[-1].name.timestamp()), timezone.utc)
                self.dd.metric_update_tracker[f'{metric_slug}']['datetime_updated'] = dt
                # Santiment stores their data at beginning of candle that contained the data.
                # Everyone else stores it at end (As they should) so we fix santiment data for them.

                self.dd.historic_external_data = pd.merge(
                    self.dd.historic_external_data,
                    metric_dict[metric][metric_slug],
                    how='left',
                    on='datetime'
                    ).ffill()
        for item in to_remove:
            self.dd.metric_slug_final.remove(item)

    def append_new_row_to_historic_external_data(self, response, get_many_dict):
        """
        Append new row to the historic_external_data dataframe. This
        function checks the response from santiment if there was new data
        or not, and ffills values accordingly.

        Note: santiment data is stored at the back of the bucket, not the front.
        TODO consider storing all their values 1 "minInterval" forward (front
        of the bucket). But it should be ensured that this minInterval is properly
        handled across all metrics.
        """
        # metric_dict = dict(zip(self.metric_slug_temporary, response))
        metric_dict = dict(zip(get_many_dict.keys(), response))
        hist_df = self.dd.historic_external_data
        # metric_df = pd.DataFrame(np.nan, index=hist_df.index[-1:], columns=hist_df.columns)
        # hist_df = pd.concat([hist_df, metric_df], ignore_index=True, axis=0)

        # for metric in self.metric_slug_temporary:
        for metric in get_many_dict.keys():
            # metric_dict[metric].rename(columns={'value': metric}, inplace=True)
            for slug in metric_dict[metric].columns:
                metric_slug = f"{metric}/{slug}"
                metric_dict[metric].rename(columns={f"{slug}": metric_slug}, inplace=True)

                if metric_dict[metric][metric_slug].empty:
                    logger.info(f'Unable to pull new data for {metric}, ffilling')
                    continue
                dt = datetime.fromtimestamp(
                    int(metric_dict[metric].iloc[-1].name.timestamp()), timezone.utc)

                # santiment stores data at beginning of candle where it was computed.
                # minInterval = self.dd.metric_update_tracker[metric]['minInterval']
                # dt += timedelta(seconds=timeframe_to_seconds(minInterval))

                if dt == self.dd.metric_update_tracker[f'{metric_slug}']['datetime_updated']:
                    logger.info(
                        f"no new data available for {metric_slug} from santiment, "
                        f"waiting since {dt}")
                    continue

                # metric_dict[metric].rename(columns={'value': metric}, inplace=True)

                self.dd.metric_update_tracker[f'{metric_slug}']['datetime_updated'] = dt
                self.dd.metric_update_tracker[f'{metric_slug}']['datetime_grabbed'] = datetime.now(
                    tz=timezone.utc)

                series_df = metric_dict[metric][metric_slug].reset_index()
                series_df.columns = ['datetime', metric_slug]
                # first time around, we need a column ready?
                # if metric_slug not in hist_df:
                #     hist_df[metric_slug] = np.nan
                merged_df = pd.merge(hist_df, series_df, on='datetime',
                                     how='outer', suffixes=('', '_from_series'))
                merged_df[metric_slug] = merged_df[f"{metric_slug}_from_series"].combine_first(
                    merged_df[metric_slug])
                merged_df = merged_df.drop(columns=[f"{metric_slug}_from_series"])

                hist_df = merged_df

                logger.info(f'Successfully pulled new data for {metric_slug}')

        # hist_df['datetime'].iloc[-1] = hist_df['datetime'].iloc[-2] + timedelta(
        #     seconds=timeframe_to_seconds(self.config['timeframe']))
        hist_df.fillna(method='ffill', inplace=True)
        self.dd.historic_external_data = hist_df
        self.dd.save_historic_external_data_to_disk()
        self.dd.save_metric_update_tracker_to_disk()

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
                columns.append(f'{metric}_shift-{shift}')
        shifted_df = DataFrame(
            np.zeros((len(dataframe.index), len(dataframe.columns) * shifts)),
            columns=columns, index=dataframe.index)
        base_tf_seconds = timeframe_to_seconds(self.config['timeframe'])
        for metric in dataframe.columns:
            minInt = self.dd.metric_update_tracker[f'{metric}']['minInterval']
            num_base_candles = timeframe_to_seconds(minInt) / base_tf_seconds
            for shift in range(1, shifts + 1):
                candle_shift = int(shift * num_base_candles)
                shifted_df[f'{metric}_shift-{shift}'] = dataframe[metric].shift(candle_shift)

        dataframe = pd.concat([dataframe, shifted_df], axis=1)

        return dataframe
