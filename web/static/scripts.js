// jQuery is a popular JavaScript library for simplifying HTML document traversing,
// event handling, and AJAX interactions. The "$" symbol is an alias for "jQuery".
// Everything inside $(function() {...}) will run once the document is fully loaded.
$(function() {
    // Selects the HTML element with the ID 'startTimePicker' and applies a date range picker plugin.
    // This plugin provides a user-friendly interface for date picking in web forms.
    $('#startTimePicker').daterangepicker({
        // Configures the date range picker to behave as a single date picker with time selection.
        singleDatePicker: true,
        timePicker: true,
        timePicker24Hour: true, // Sets time format to 24-hour clock.
        timePickerIncrement: 15, // Sets the increment of minutes selection to 15 minutes.
        startDate: moment().subtract(24, 'days'), // Default starting date is 4 days before the current day.
        locale: {
            format: 'YYYY-MM-DDTHH:mm:ss.SSS[Z]'
        }
    });
    // Sets the default value of the HTML input element with ID 'duration' to 2 (min).
    $('#duration').val(2);

    // Calls a function to load configuration and data when the page is ready.
    loadConfigAndData();
});

function intervalToSeconds(interval) {
    switch (interval) {
        case '1s': return 1;
        case '5s':   return 5;
        case '30s':  return 30;
        case '1min': return 60;
        case '5min': return 5*60;
        case '30min': return 30*60;
        case '1h': return 60*60;
        default: return 300;
    }
}

function loadConfigAndData() {
    // Fetch API is used to make an HTTP request to retrieve 'panels.json'.
    fetch('static/panels.json')
        .then(response => response.json())
        .then(panels => {
            panels.forEach(panel => {
                var containerId = 'container_' + panel.title.replace(/[^a-zA-Z0-9]/g, '_');
                var containerDiv = $('<div>').attr('id', containerId).css({
                    height: '600px',
                    minWidth: '310px',
                    marginBottom: '20px'
                });
                $('body').append(containerDiv);
                loadData(panel, containerId);
            });
        })
        .catch(error => {
            console.error('Error loading panel config:', error);
        });
}

// Defines a function to load data for a specific panel.
function loadData(panel, containerId) {
    var startTime = moment.utc($('#startTimePicker').val());
    var intervalInSeconds = intervalToSeconds($('#interval').val());
    var duration = intervalInSeconds * parseInt($('#duration').val());
    var endTime = moment(startTime).add(duration, 'seconds');

    panel.endpoints.forEach(endpoint => {
        fetchData(panel.symbol, 
            startTime.format('YYYY-MM-DDTHH:mm:ss.SSS[Z]'), 
            endTime.format('YYYY-MM-DDTHH:mm:ss.SSS[Z]'), 
            endpoint.type, 
            endpoint.url, 
            intervalInSeconds, 
            containerId, 
            panel.title,
            endpoint.parameters
        );
    });
}

function fetchData(symbol, startDate, endDate, type, url, interval, containerId, title) {
    console.log(`Fetching data: ${symbol} from ${url}, Start: ${startDate}, End: ${endDate}, Interval: ${interval} seconds`);

    let queryParams = new URLSearchParams({
        symbol: encodeURIComponent(symbol),
        start: encodeURIComponent(startDate),
        end: encodeURIComponent(endDate),
        interval: encodeURIComponent(interval)
    });

    // Append additional parameters if they exist
    if (parameters) {
        Object.keys(parameters).forEach(key => {
            queryParams.append(key, parameters[key]);
        });
    }

    fetch(`${url}?${queryParams.toString()}`)
        .then(response => response.json())
        .then(data => {
            // Processes the received data based on the type of data representation (line or OHLC chart).
            // The `seriesData` constant is assigned the result of the `map` method called on the `data` array.
            // `map` is used to transform each item in the original `data` array into a new format suitable for chart representation.
            // Each `item` in `data` is expected to contain data points with properties like time, price, and possibly open, high, low, close values for OHLC charts.
            // The arrow function `(item => { ... })` passed to `map` is executed for each element of `data`:
            // - Inside the arrow function, conditional logic checks the 'type' to format the data appropriately:
            //   - If `type` is 'line', the function transforms the item into a new array containing the item's time as a timestamp and its price.
            //   - If `type` is 'ohlc', the function transforms the item into a new array with the item's time as a timestamp followed by its open, high, low, and close values.
            // This transformation is crucial for the Highcharts library used later to render the data accurately in the specified chart type.
            // The `map` method returns a new array consisting of these transformed items, which is then assigned to `seriesData`.
            const seriesData = data.map(item => {
                if (type === 'line' || type === 'scatter') {
                    return [item[0], item[1]];
                } else if (type === 'candlestick') {
                    return [item[0], item[1], item[2], item[3], item[4]];
                }
            // The .sort((a, b) => a[0] - b[0]) function is used to sort an array of arrays (or objects) based on their first elements.
            // In this code, each array's first element is a timestamp, and the sort function arranges them in ascending order.
            // The arrow function (a, b) => a[0] - b[0] takes two parameters, 'a' and 'b', which represent two elements of the array being sorted.
            // The function calculates the difference between the first element of 'a' and 'b':
            // - If the result is negative, 'a' is placed before 'b'.
            // - If the result is positive, 'a' is placed after 'b'.
            // - If the result is zero, the order of 'a' and 'b' relative to each other does not change.
            // This sorting method ensures that the data is ordered chronologically, which is crucial for correctly displaying data in time-sensitive charts.
            }).sort((a, b) => a[0] - b[0]);

            if (seriesData.length == 0) {
                console.log('Got no data. Ignore');
                return;
            }

            Highcharts.stockChart(containerId, {
                rangeSelector: {
                    buttons: [{
                        type: 'second',
                        count: 1,
                        text: '1s'
                    }, {
                        type: 'minute',
                        count: 1,
                        text: '1m'
                    }, {
                        type: 'minute',
                        count: 5,
                        text: '5m'
                    }, {
                        type: 'minute',
                        count: 15,
                        text: '15m'
                    }, {
                        type: 'minute',
                        count: 30,
                        text: '30m'
                    }, {
                        type: 'hour',
                        count: 1,
                        text: '1h'
                    }, {
                        type: 'all',
                        text: 'All'
                    }],
                    inputEnabled: true,
                    selected: 1
                },
                title: {
                    text: title
                },
                xAxis: {
                    crosshair: true,
                    type: 'datetime',
                    // Ensure tickInterval is set correctly; here it should likely be in milliseconds if showing seconds
                    tickInterval: 1,
                    events: {
                        setExtremes: function(e) {
                            syncExtremes(e);
                        }
                    }
                },
                yAxis: {
                    title: {
                        text: 'Price'
                    }
                },
                series: [{
                    type: type,
                    name: title,
                    data: seriesData,
                    tooltip: {
                        valueDecimals: 2
                    },
                    dataGrouping: {
                        enabled: true
                    },
                    color: type === 'line' ? undefined : 'red',
                    upColor: type === 'candlestick' ? 'green' : undefined,
                    lineColor: type === 'candlestick' ? 'black' : undefined,
                    upLineColor: type === 'candlestick' ? 'black' : undefined,
                    marker: { 
                        enabled: true,
                        radius: 3, 
                        fillColor: 'blue' 
                    }
                }],
                tooltip: {
                    formatter: function() {
                        const dateStr = Highcharts.dateFormat('%d-%m-%Y %H:%M:%S', new Date(this.x));
                        switch (this.series.options.type) {
                            case 'scatter':
                                return dateStr + '<br>Value: ' + this.y;
                            case 'candlestick':
                                return dateStr +
                                    '<br>Open: ' + this.point.open +
                                    '<br>High: ' + this.point.high +
                                    '<br>Low: ' + this.point.low +
                                    '<br>Close: ' + this.point.close;
                            default:
                                return dateStr + '<br>Value: ' + this.y;
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

function syncExtremes(e) {
    var thisChart = this.chart;
    if (e.trigger !== 'syncExtremes') {
        Highcharts.each(Highcharts.charts, function(chart) {
            if (chart !== thisChart) {
                if (chart.xAxis[0].setExtremes) {
                    chart.xAxis[0].setExtremes(e.min, e.max, undefined, false, { trigger: 'syncExtremes' });
                }
            }
        });
    }
}