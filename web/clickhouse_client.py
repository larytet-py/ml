import clickhouse_connect


class ClickhouseClient:
    def __init__(
        self,
        username: str = "default",
        password: str = "password",
        host: str | None = None,
    ):
        self.username = username
        self.password = password
        self.host = host

    def get_client(self, settings=None):
        connection_args = {
            "settings": settings or {},
            "username": self.username,
            "password": self.password,
        }
        if self.host is not None:
            connection_args["host"] = self.host
        return clickhouse_connect.get_client(**connection_args)
