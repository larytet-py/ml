// jQuery is a popular JavaScript library for simplifying HTML document traversing,
// event handling, and AJAX interactions. The "$" symbol is an alias for "jQuery".
// Everything inside $(function() {...}) will run once the document is fully loaded.
$(function() {
    var timeFormat = 'YYYY-MM-DDTHH:mm:ss.SSS[Z]'
    // Default starting date is 24 days before the current day.
    var startTimePickerVal = Cookies.get('startTimePicker') || moment().subtract(24, 'days').format(timeFormat);
    var intervalVal = Cookies.get('interval') || '1s';
    var durationVal = Cookies.get('duration') || 2000;

    // Selects the HTML element with the ID 'startTimePicker' and applies a date range picker plugin.
    // This plugin provides a user-friendly interface for date picking in web forms.
    $('#startTimePicker').daterangepicker({
        // Configures the date range picker to behave as a single date picker with time selection.
        singleDatePicker: true,
        timePicker: true,
        timePicker24Hour: true, // Sets time format to 24-hour clock.
        timePickerIncrement: 5, // Sets the increment of minutes selection to 15 minutes.
        startDate: startTimePickerVal,
        locale: {
            format: timeFormat
        }
    });

    // Sets the default value of the HTML input element with ID 'duration' to 2 (min).
    $('#interval').val(intervalVal);
    $('#duration').val(durationVal);

    // Calls a function to load configuration and data when the page is ready.
    loadConfigAndData();

    // Save settings to cookies when they change
    $('#startTimePicker').on('apply.daterangepicker', function(ev, picker) {
        Cookies.set('startTimePicker', picker.startDate.format(timeFormat));
    });
    $('#interval').on('change', function() {
        Cookies.set('interval', $(this).val());
    });
    $('#duration').on('change', function() {
        Cookies.set('duration', $(this).val());
    });

    $('#loadButton').on('click', function() {
        loadConfigAndData();
    });
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
            endpoint.parameters || {}
        );
    });
}

function fetchData(symbol, startDate, endDate, type, url, interval, containerId, title, parameters) {
    console.log(`Fetching data: ${symbol} from ${url}, Start: ${startDate}, End: ${endDate}, Interval: ${interval} seconds`);

    let queryParams = createQueryParams(symbol, startDate, endDate, interval, parameters);

    fetch(`${url}?${queryParams.toString()}`)
        .then(response => response.json())
        .then(data => {
            const seriesData = processSeriesData(data, type);
            if (seriesData.length === 0) {
                console.log('Got no data. Ignore');
                return;
            }

            let chart = Highcharts.charts.find(c => c.renderTo.id === containerId);
            if (chart) {
                addSeriesToChart(chart, type, title, seriesData);
            } else {
                createNewChart(containerId, type, title, seriesData);
            }
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

function createQueryParams(symbol, startDate, endDate, interval, parameters) {
    let queryParams = new URLSearchParams({ symbol, start: startDate, end: endDate, interval });
    if (parameters) {
        Object.keys(parameters).forEach(key => {
            queryParams.append(key, parameters[key]);
        });
    }
    return queryParams;
}

function processSeriesData(data, type) {
    return data.map(item => {
        if (type === 'line' || type === 'scatter') {
            return [item[0], item[1]];
        } else if (type === 'candlestick') {
            return [item[0], item[1], item[2], item[3], item[4]];
        }
    }).sort((a, b) => a[0] - b[0]);
}

function addSeriesToChart(chart, type, title, seriesData) {
    chart.addSeries({
        type: type,
        name: title,
        data: seriesData,
        tooltip: {
            valueDecimals: 2
        },
        dataGrouping: {
            enabled: true
        },
        color: type === 'line' ? 'lightblue' : 'red',
        upColor: type === 'candlestick' ? 'green' : undefined,
        lineColor: type === 'candlestick' ? 'black' : undefined,
        upLineColor: type === 'candlestick' ? 'black' : undefined,
        marker: {
            enabled: type === 'line' ? false : true,
            radius: 3,
            fillColor: 'blue'
        }
    });
}

function createNewChart(containerId, type, title, seriesData) {
    Highcharts.stockChart(containerId, {
        rangeSelector: {
            buttons: getRangeSelectorButtons(),
            inputEnabled: true,
            selected: 1
        },
        title: {
            text: title
        },
        xAxis: {
            crosshair: true,
            type: 'datetime',
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
                enabled: type === 'line' ? false : true,
                radius: 3,
                fillColor: 'blue'
            }
        }],
        tooltip: {
            formatter: function() {
                return formatTooltip(this);
            }
        }
    });
}

function getRangeSelectorButtons() {
    return [
        { type: 'second', count: 1, text: '1s' },
        { type: 'minute', count: 1, text: '1m' },
        { type: 'minute', count: 5, text: '5m' },
        { type: 'minute', count: 15, text: '15m' },
        { type: 'minute', count: 30, text: '30m' },
        { type: 'hour', count: 1, text: '1h' },
        { type: 'all', text: 'All' }
    ];
}

function formatTooltip(point) {
    const dateStr = Highcharts.dateFormat('%d-%m-%Y %H:%M:%S', new Date(point.x));
    switch (point.series.options.type) {
        case 'scatter':
            return `${dateStr}<br>Value: ${point.y}`;
        case 'candlestick':
            return `${dateStr}<br>Open: ${point.point.open}<br>High: ${point.point.high}<br>Low: ${point.point.low}<br>Close: ${point.point.close}`;
        default:
            return `${dateStr}<br>Value: ${point.y}`;
    }
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

function clearCharts() {
    // Remove all div elements with IDs starting with 'container_' that are specifically meant for charts
    $('div[id^="container_"]').remove();

    // Destroy all existing Highcharts instances
    Highcharts.charts.forEach((chart, index) => {
        if (chart) {
            chart.destroy();
            Highcharts.charts[index] = null; // Clear the reference
        }
    });

    // Remove null values from Highcharts.charts array
    Highcharts.charts = Highcharts.charts.filter(chart => chart !== null);
}
