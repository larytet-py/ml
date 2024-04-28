# Run ClickHouse

```
docker run -d --name ml-clickhouse-server --ulimit nofile=262144:262144 clickhouse/clickhouse-server
docker exec -it ml-clickhouse-server clickhouse-client
```


Limit the number of partitions in the ClickHouse 

```SQL
CREATE TABLE trades
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


