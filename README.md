# Run ClickHouse

```
docker run -d --name ml-clickhouse-server --ulimit nofile=262144:262144 clickhouse/clickhouse-server
docker exec -it ml-clickhouse-server clickhouse-client
```


Limit the number of partitions in the ClickHouse 

```SQL
CREATE TABLE trades
(
    EventTime DateTime64(3),  -- Supporting millisecond precision
    Price Decimal64(8),
    Volume Decimal64(8),
    Value Decimal64(8),
    TimestampMs UInt64,
    Flag1 Boolean,
    Flag2 Boolean
)
ENGINE = MergeTree
PARTITION BY toDate(EventTime)  -- Daily partitioning
ORDER BY (toDate(EventTime), EventTime)
SETTINGS index_granularity = 8192;
```


