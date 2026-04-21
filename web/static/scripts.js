// jQuery is a popular JavaScript library for simplifying HTML document traversing,
// event handling, and AJAX interactions. The "$" symbol is an alias for "jQuery".
// Everything inside $(function() {...}) will run once the document is fully loaded.
$(function() {
    var timeFormat = 'YYYY-MM-DDTHH:mm:ss.SSS[Z]'
    // Backward-compatible fallback from old global cookies.
    var legacyStartTime = Cookies.get('startTimePicker');
    var legacyInterval = Cookies.get('interval');
    var legacyDuration = Cookies.get('duration');

    // Default starting date is 24 days before the current day.
    var startTimePickerVal = legacyStartTime || moment().subtract(24, 'days').format(timeFormat);
    var intervalVal = legacyInterval || '1s';
    var durationVal = legacyDuration || 2000;

    // Selects the HTML element with the ID 'startTimePicker' and applies a date range picker plugin.
    // This plugin provides a user-friendly interface for date picking in web forms.
    $('#startTimePicker').daterangepicker({
        // Configures the date range picker to behave as a single date picker with time selection.
        singleDatePicker: true,
        timePicker: true,
        timePicker24Hour: true, // Sets time format to 24-hour clock.
        timePickerIncrement: 1, // Sets the increment of minutes selection to 15 minutes.
        startDate: startTimePickerVal,
        locale: {
            format: timeFormat
        }
    });

    // Sets the default value of the HTML input element with ID 'duration' to 2 (min).
    $('#interval').val(intervalVal);
    $('#duration').val(durationVal);

    initializeConfigurationSelector()
        .then(function() {
            return synchronizeUiSettingsWithSelectedConfiguration();
        })
        .then(function(panels) {
            loadConfigAndData(panels);
        })
        .catch(function(error) {
            console.error('Error loading available configurations:', error);
        });

    // Save settings by symbol when they change.
    $('#startTimePicker').on('apply.daterangepicker', function(ev, picker) {
        saveUiSettingsForCurrentSymbol();
    });
    $('#interval').on('change', function() {
        saveUiSettingsForCurrentSymbol();
    });
    $('#duration').on('change', function() {
        saveUiSettingsForCurrentSymbol();
    });
    $('#configSelector').on('change', function() {
        Cookies.set('panelConfiguration', $(this).val());
        synchronizeUiSettingsWithSelectedConfiguration()
            .then(function(panels) {
                loadConfigAndData(panels);
            })
            .catch(function(error) {
                console.error('Error loading panel config while synchronizing settings:', error);
            });
    });

    $('#loadButton').on('click', function() {
        saveUiSettingsForCurrentSymbol();
        loadConfigAndData();
    });
});

var currentConfigurationSymbol = null;

function getSymbolSettingsCookie() {
    var rawValue = Cookies.get('symbolUiSettings');
    if (!rawValue) {
        return {};
    }
    try {
        var parsed = JSON.parse(rawValue);
        return typeof parsed === 'object' && parsed !== null ? parsed : {};
    } catch (error) {
        console.warn('Could not parse symbolUiSettings cookie, resetting.', error);
        return {};
    }
}

function setSymbolSettingsCookie(allSettings) {
    Cookies.set('symbolUiSettings', JSON.stringify(allSettings));
}

function getDefaultIntervalForSymbol(symbol) {
    return symbol === 'BTC' ? '1s' : '5min';
}

function getDefaultDurationForSymbol() {
    return 2000;
}

function getCurrentSymbolFromPanels(panels) {
    if (!Array.isArray(panels) || panels.length === 0) {
        return 'DEFAULT';
    }
    return (panels[0].symbol || 'DEFAULT').toUpperCase();
}

function applyUiSettingsForSymbol(symbol) {
    var allSettings = getSymbolSettingsCookie();
    var symbolSettings = allSettings[symbol] || {};
    var timeFormat = 'YYYY-MM-DDTHH:mm:ss.SSS[Z]';
    var startDate = symbolSettings.date || symbolSettings.startTimePicker || moment().subtract(24, 'days').format(timeFormat);
    var interval = symbolSettings.resolution || symbolSettings.interval || getDefaultIntervalForSymbol(symbol);
    var duration = symbolSettings.intervalSize || symbolSettings.duration || getDefaultDurationForSymbol();

    $('#startTimePicker').data('daterangepicker').setStartDate(startDate);
    $('#startTimePicker').val(startDate);
    $('#interval').val(interval);
    $('#duration').val(duration);
}

function saveUiSettingsForCurrentSymbol() {
    if (!currentConfigurationSymbol) {
        return;
    }
    var allSettings = getSymbolSettingsCookie();
    var startDate = moment.utc($('#startTimePicker').val()).format('YYYY-MM-DDTHH:mm:ss.SSS[Z]');
    var interval = $('#interval').val();
    var duration = $('#duration').val();
    allSettings[currentConfigurationSymbol] = {
        // Preferred keys.
        date: startDate,
        resolution: interval,
        intervalSize: duration,
        // Backward-compatible keys.
        startTimePicker: startDate,
        interval: interval,
        duration: duration
    };
    setSymbolSettingsCookie(allSettings);
}

function fetchPanelConfiguration(configurationName) {
    return fetch('/panel_configuration?name=' + encodeURIComponent(configurationName))
        .then(function(response) {
            if (!response.ok) {
                throw new Error('Failed to fetch panel configuration: ' + configurationName);
            }
            return response.json();
        });
}

function synchronizeUiSettingsWithSelectedConfiguration() {
    var selectedConfiguration = $('#configSelector').val();
    if (!selectedConfiguration) {
        return Promise.resolve([]);
    }
    return fetchPanelConfiguration(selectedConfiguration).then(function(panels) {
        currentConfigurationSymbol = getCurrentSymbolFromPanels(panels);
        applyUiSettingsForSymbol(currentConfigurationSymbol);
        return panels;
    });
}

function initializeConfigurationSelector() {
    return fetch('/panel_configurations')
        .then(function(response) {
            if (!response.ok) {
                throw new Error('Failed to fetch panel configurations');
            }
            return response.json();
        })
        .then(function(payload) {
            var configurations = payload.configurations || [];
            var selector = $('#configSelector');
            selector.empty();

            configurations.forEach(function(configurationName) {
                selector.append(
                    $('<option>').val(configurationName).text(configurationName)
                );
            });

            if (configurations.length === 0) {
                throw new Error('No panel configurations were returned by the server');
            }

            var savedConfiguration = Cookies.get('panelConfiguration');
            var selectedConfiguration = savedConfiguration && configurations.includes(savedConfiguration)
                ? savedConfiguration
                : (payload.default || configurations[0]);

            selector.val(selectedConfiguration);
            Cookies.set('panelConfiguration', selectedConfiguration);
        });
}

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

function createPanel(containerId) {
    var savedHeight = Cookies.get(containerId + '_height');
    var containerDiv = $('<div>').attr('id', containerId).addClass('resizable-panel').css({
        height: savedHeight ? savedHeight + 'px' : '300px',
        minWidth: '310px',
        marginBottom: '20px',
        resize: 'vertical', // Enable vertical resizing
        overflow: 'auto', // Allow content to be scrolled if needed
        position: 'relative' // Positioning of components
    });
    $('body').append(containerDiv);
}

var resizing = false;

function addResizerHandle(containerId) {
    var containerDiv = $('#' + containerId);

    var resizerDiv = $('<div>').addClass('resizer').css({
        width: '100%',
        height: '2px',
        cursor: 'ns-resize', 
        backgroundColor: '#888',
        position: 'absolute',
        bottom: 0,
        zIndex: 9999
    });

    containerDiv.append(resizerDiv);

    var startY, startHeight, resizing = false;

    // Handle resizing
    resizerDiv.on('mousedown', mouseDownHandler);

    function mouseDownHandler(e) {
        e.preventDefault();
        resizing = true;
        startY = e.pageY;
        startHeight = containerDiv.height();
        $(document).on('mousemove', mouseMoveHandler);
        $(document).on('mouseup', mouseUpHandler);
    }

    function mouseMoveHandler(e) {
        if (resizing) {
            var height = startHeight + e.pageY - startY;
            containerDiv.css('height', height + 'px');
        }
    }

    function mouseUpHandler() {
        if (resizing) {
            resizing = false;
            $(document).off('mousemove', mouseMoveHandler);
            $(document).off('mouseup', mouseUpHandler);

            // Save the current height of the panel to cookies
            var containerId = containerDiv.attr('id');
            Cookies.set(containerId + '_height', containerDiv.height());            
        }
    }
}


function loadConfigAndData(preloadedPanels) {
    var selectedConfiguration = $('#configSelector').val();
    if (!selectedConfiguration) {
        console.error('No panel configuration selected');
        return;
    }

    clearCharts();
    var panelPromise = preloadedPanels ? Promise.resolve(preloadedPanels) : fetchPanelConfiguration(selectedConfiguration);
    panelPromise
        .then(panels => {
            currentConfigurationSymbol = getCurrentSymbolFromPanels(panels);
            panels.forEach(panel => {
                var containerId = 'container_' + panel.title.replace(/[^a-zA-Z0-9]/g, '_');
                createPanel(containerId);
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
                //addSeriesToChart(chart, type, title, seriesData);
            } else {
                createNewChart(containerId, type, title, seriesData);
                addResizerHandle(containerId);
            }
        })
        .catch(error => {
            console.error('Error fetching data:', error);
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
                setExtremes: syncExtremes
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
    const dateStr = Highcharts.dateFormat('%d-%m-%Y %H:%M:%S.%L', new Date(point.x));
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
        Highcharts.charts.forEach(function(chart) {
            if (!chart || chart === thisChart || !chart.xAxis || !chart.xAxis[0]) {
                return;
            }
            if (chart.xAxis[0].setExtremes) {
                chart.xAxis[0].setExtremes(e.min, e.max, undefined, false, { trigger: 'syncExtremes' });
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
