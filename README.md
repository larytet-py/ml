# Run ClickHouse

```
docker run -d \
  --name ml-clickhouse-server \
  --ulimit nofile=262144:262144 \
  -e CLICKHOUSE_USER=default \
  -e CLICKHOUSE_PASSWORD=password \
  -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
  -p 8123:8123 \
  -p 9000:9000 \
  -v "$PWD/clickhouse:/var/lib/clickhouse" \
  clickhouse/clickhouse-server

docker exec -it ml-clickhouse-server clickhouse-client --receive_timeout=60000 --send_timeout=600 --multiquery
```



## Python Virtual Environment and Server

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Python 3.11
python3 web/web.py --debug_level=DEBUG
```

Run the WEB server http://127.0.0.1:8080

# Logs

```
docker exec -it ml-clickhouse-server bash -c "tail -f /var/log/clickhouse-server/clickhouse-server.err.log"
docker exec -it ml-clickhouse-server bash -c "tail -f /var/log/clickhouse-server/clickhouse-server.log"
```

# Configuration 

```json
{
    "panels": [
        {
            "symbol": "BTC",
            "title": "BTC 5s",
            "endpoints": [
                {
                    "url": "/ohlc_data",
                    "type": "line"
                }
            ]
        },
        {
            "symbol": "BTC",
            "title": "Autocorrelation",
            "endpoints": [
                {
                    "url": "/autocorrelation",
                    "type": "line",
                    "parameters": {"window_size": 300}
                }
            ]
        }
    ]
}
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
