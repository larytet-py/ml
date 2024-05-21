# Run ClickHouse

```
docker run -d \
  --name ml-clickhouse-server \
  --ulimit nofile=262144:262144 \
  -p 8123:8123 \
  -p 9000:9000 \
  -v $PWD/clickhouse:/var/lib/clickhouse \
  clickhouse/clickhouse-server

docker exec -it ml-clickhouse-server clickhouse-client --receive_timeout=60000 --send_timeout=600
```


Run the WEB server http://127.0.0.1:8080

```sh
python3 web/web.py --debug_level=DEBUG
```


# Logs

```
docker exec -it ml-clickhouse-server bash -c "tail -f /var/log/clickhouse-server/clickhouse-server.err.log"
docker exec -it ml-clickhouse-server bash -c "tail -f /var/log/clickhouse-server/clickhouse-server.log"
```


# Links

* https://www.youtube.com/watch?v=BRUlSm4gdQ4
* https://quantpedia.com/machine-learning-execution-time-in-asset-pricing/
* https://www.youtube.com/watch?v=LTI9i_Njj3U
* https://www.youtube.com/watch?v=WfrGj3qvDEQ
* https://en.wikipedia.org/wiki/Autoregressive_moving-average_model
* https://arxiv.org/pdf/2301.12561
* https://shakti.com/

# Books 

* Option Volatility and Pricing by Sheldon Natenberg: https://amzn.to/3hqRglr
* Dynamic Hedging: https://amzn.to/2TVvgGr
* Frequently Asked Questions in Quantitative Finance (Second Edition) by Paul Wilmott: https://amzn.to/36rvg3E
* Python for Data Analysis: https://amzn.to/2T6F1Bm
* Introduction to Linear Algebra: https://amzn.to/3qXc47m
* Advances in Active Portfolio Management: https://amzn.to/3xwSfpX
* Technical Analysis is Mostly Bullshit: https://amzn.to/2TU3M41

# SQL tips


Limit the number of partitions in the ClickHouse, keep sorted by timestamp

```SQL
CREATE TABLE trades_BTC
(
    id UInt64,
    price Decimal(18, 8),
    qty Decimal(12, 8),
    base_qty Decimal(12, 8),
    time UInt64,
    is_buyer_maker Boolean,
    unknown_flag Boolean DEFAULT True,
    timestamp DateTime64(3) -- Millisecond precision
)
ENGINE = MergeTree
PARTITION BY toDate(toDateTime(time / 1000))
ORDER BY (timestamp)
SETTINGS index_granularity = 8192;
```


Verify that the ascending order of the column `time` 

```SQL
SELECT 
    id,
    time,
    runningDifference(time) AS time_diff
FROM (
    SELECT 
        id,
        time
    FROM trades_ETH
    ORDER BY id, time
)
WHERE time_diff < 0;
```

Aggregation of tick data in 1 minute OHLCVs

```SQL
CREATE TABLE ohlc_M1_BTC
(
    timestamp DateTime64(3),
    open_price Decimal(18, 8),
    high_price Decimal(18, 8),
    low_price Decimal(18, 8),
    close_price Decimal(18, 8),
    num_trades UInt64
) 
ENGINE = MergeTree()
ORDER BY timestamp;

INSERT INTO ohlc_M1_BTC
SELECT
    toStartOfInterval(timestamp, INTERVAL 60 SECOND) AS timestamp,
    any(price) AS open_price,
    max(price) AS high_price,
    min(price) AS low_price,
    anyLast(price) AS close_price,
    count() AS num_trades
FROM trades_BTC
GROUP BY timestamp
ORDER BY timestamp;
```

Generate trades density 

```SQL
CREATE TABLE trades_density
ENGINE = MergeTree()
ORDER BY timestamp AS
SELECT
    toStartOfInterval(timestamp, INTERVAL 300 SECOND) AS timestamp,
    (toFloat64(anyLast(price)) - toFloat64(any(price))) / toFloat64(any(price)) AS roc,
    count() as count,
    if(
        anyLast(price) = any(price),
        0,
        log(count() / (abs(toFloat64(anyLast(price)) - toFloat64(any(price))) / toFloat64(any(price))))
    ) AS density
FROM trades_BTC
WHERE timestamp BETWEEN '2021-01-01 00:00:00' AND '2024-04-25 11:38:58'
GROUP BY timestamp
ORDER BY timestamp ASC
```

# Performnace test 

```sql
CREATE TABLE ohlcv_M1_BTC
(
    time UInt64,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    num_trades Int
) 
ENGINE = MergeTree()
ORDER BY time;

INSERT INTO ohlcv_M1_BTC SELECT
    toUnixTimestamp64Milli(CAST(toStartOfInterval(timestamp, toIntervalSecond(60)), 'DateTime64')) AS time,
    toFloat64(any(price)) AS open,
    toFloat64(max(price)) AS high,
    toFloat64(min(price)) AS low,
    toFloat64(anyLast(price)) AS close,
    count() AS num_trades
FROM trades_BTC
GROUP BY time
ORDER BY time ASC;
```

```sh
time docker exec -it ml-clickhouse-server clickhouse-client --receive_timeout=60000 --send_timeout=600 --query="WITH
    rolling_metrics AS (
        SELECT
            time,
            sum(num_trades) OVER (
                ORDER BY time
                ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
            ) AS rolling_num_trades,
            abs(
                (max(close) OVER (
                    ORDER BY time
                    ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
                ) - min(open) OVER (
                    ORDER BY time
                    ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
                )) / nullif(min(open) OVER (
                    ORDER BY time
                    ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
                ), 0)
            ) AS rolling_roc
        FROM ohlcv_M1_BTC
    )
SELECT
    time,
    if(
        rolling_roc = 0,
        last_value(log(rolling_num_trades / nullif(rolling_roc, 0))) OVER (
            ORDER BY time
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        log(
            rolling_num_trades / nullif(rolling_roc, 0)
        )
    ) AS forward_filled_density
FROM rolling_metrics
ORDER BY time ASC;" --format CSV > output.csv
```


Histogram 

```sql
SELECT histogram(5)(base_qty) AS base_qty_histogram
FROM
(
    SELECT toFloat64(base_qty) AS base_qty
    FROM trades_BTC
) FORMAT CSV;


WITH 
    20 AS bins,
    40 AS max_width,
    (SELECT count(*) FROM trades_BTC) AS total_rows
SELECT
    bin_range_start,
    bin_range_end,
    height,
    bar(height, 0, total_rows/(bins/3), max_width) AS bar
FROM
(
    SELECT
        bin_tuple.1 AS bin_range_start,
        bin_tuple.2 AS bin_range_end,
        bin_tuple.3 AS height
    FROM
    (
        SELECT arrayJoin(hist) AS bin_tuple
        FROM
        (
            SELECT histogram(bins)(log10(toFloat64(base_qty))) AS hist
            FROM trades_BTC
        )
    )
)
ORDER BY bin_range_start;
```