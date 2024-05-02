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


Limit the number of partitions in the ClickHouse 

```SQL
CREATE TABLE btc_trades
(
    id UInt64,
    price Decimal(18, 8),
    qty Decimal(12, 8),
    base_qty Decimal(12, 8),
    time UInt64,
    is_buyer_maker Boolean,
    unknown_flag Boolean DEFAULT True
)
ENGINE = MergeTree
PARTITION BY toDate(toDateTime(time / 1000))
ORDER BY (id)
SETTINGS index_granularity = 8192;
```


# Logs

```
docker exec -it ml-clickhouse-server bash -c "tail -f /var/log/clickhouse-server/clickhouse-server.err.log"
docker exec -it ml-clickhouse-server bash -c "tail -f /var/log/clickhouse-server/clickhouse-server.log"
```


# SQL tips

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
