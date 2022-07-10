import vegaEmbed from 'vega-embed'


spec = {
    "config": {"view": {"continuousWidth": 400, "continuousHeight": 300}},
    "vconcat": [
      {
        "layer": [
          {
            "data": {"name": "data-47c9ee55e0e18e98ff0830236bcebaef"},
            "mark": "line",
            "encoding": {
              "x": {
                "axis": {"grid": false, "title": ""},
                "field": "Time",
                "type": "temporal"
              },
              "y": {
                "axis": {"title": ""},
                "field": "Equity",
                "scale": {"domain": [150, 2350]},
                "type": "quantitative"
              }
            },
            "transform": [{"filter": {"field": "Equity", "valid": true}}]
          },
          {
            "data": {"name": "data-ac0604903442ed28ac7caeaeddb9ad1b"},
            "mark": {"type": "rule", "size": 0.2},
            "encoding": {
              "color": {"value": "red"},
              "y": {"field": "y", "type": "quantitative"}
            }
          },
          {
            "data": {"name": "data-47c9ee55e0e18e98ff0830236bcebaef"},
            "mark": {"type": "rule", "size": 10},
            "encoding": {
              "opacity": {"value": 0},
              "tooltip": [
                {
                  "field": "Time",
                  "format": "%d %b %Y, %H:%M",
                  "type": "temporal"
                },
                {"field": "Equity", "format": ".2f", "type": "quantitative"}
              ],
              "x": {"field": "Time", "type": "temporal"}
            },
            "selection": {
              "selector001": {
                "type": "single",
                "nearest": true,
                "on": "mouseover",
                "fields": ["Time"],
                "empty": "none"
              }
            }
          },
          {
            "data": {"name": "data-47c9ee55e0e18e98ff0830236bcebaef"},
            "mark": {"type": "rule", "opacity": 0.25, "size": 0.25},
            "encoding": {"x": {"field": "Time", "type": "temporal"}},
            "transform": [{"filter": {"selection": "selector001"}}]
          },
          {
            "data": {"name": "data-47c9ee55e0e18e98ff0830236bcebaef"},
            "mark": {"type": "point", "filled": true},
            "encoding": {
              "opacity": {
                "condition": {"value": 1, "selection": "selector001"},
                "value": 0
              },
              "x": {
                "axis": {"grid": false, "title": ""},
                "field": "Time",
                "type": "temporal"
              },
              "y": {
                "axis": {"title": ""},
                "field": "Equity",
                "scale": {"domain": [150, 2350]},
                "type": "quantitative"
              }
            },
            "transform": [{"filter": {"field": "Equity", "valid": true}}]
          }
        ],
        "height": 300,
        "title": "Equity",
        "width": 1000
      },
      {
        "layer": [
          {
            "mark": "bar",
            "encoding": {
              "color": {"value": "darkgreen"},
              "opacity": {
                "condition": {"value": 1, "selection": "selector001"},
                "value": 0.5
              },
              "x": {
                "axis": {"grid": false, "title": ""},
                "field": "Time",
                "type": "temporal"
              },
              "y": {
                "axis": {"title": ""},
                "field": "Leverage Used",
                "scale": {"domain": [0, 5.5]},
                "type": "quantitative"
              }
            }
          },
          {
            "mark": {"type": "rule", "size": 10},
            "encoding": {
              "opacity": {"value": 0},
              "tooltip": [
                {
                  "field": "Time",
                  "format": "%d %b %Y, %H:%M",
                  "type": "temporal"
                },
                {
                  "field": "Leverage Used",
                  "format": ".2f",
                  "type": "quantitative"
                }
              ],
              "x": {"field": "Time", "type": "temporal"}
            },
            "selection": {
              "selector001": {
                "type": "single",
                "nearest": true,
                "on": "mouseover",
                "fields": ["Time"],
                "empty": "none"
              }
            }
          },
          {
            "mark": {"type": "rule", "opacity": 0.25, "size": 0.25},
            "encoding": {"x": {"field": "Time", "type": "temporal"}},
            "transform": [{"filter": {"selection": "selector001"}}]
          }
        ],
        "data": {"name": "data-47c9ee55e0e18e98ff0830236bcebaef"},
        "height": 150,
        "title": "Leverage Used",
        "width": 1000
      },
      {
        "layer": [
          {
            "mark": {"type": "line", "color": "red"},
            "encoding": {
              "x": {
                "axis": {"grid": false, "title": ""},
                "field": "Time",
                "type": "temporal"
              },
              "y": {
                "axis": {"title": ""},
                "field": "Drawdown",
                "type": "quantitative"
              }
            },
            "transform": [{"filter": {"field": "Drawdown", "valid": true}}]
          },
          {
            "mark": {"type": "rule", "size": 10},
            "encoding": {
              "opacity": {"value": 0},
              "tooltip": [
                {
                  "field": "Time",
                  "format": "%d %b %Y, %H:%M",
                  "type": "temporal"
                },
                {"field": "Drawdown", "format": ".6f", "type": "quantitative"}
              ],
              "x": {"field": "Time", "type": "temporal"}
            },
            "selection": {
              "selector001": {
                "type": "single",
                "nearest": true,
                "on": "mouseover",
                "fields": ["Time"],
                "empty": "none"
              }
            }
          },
          {
            "mark": {"type": "rule", "opacity": 0.25, "size": 0.25},
            "encoding": {"x": {"field": "Time", "type": "temporal"}},
            "transform": [{"filter": {"selection": "selector001"}}]
          },
          {
            "mark": {"type": "point", "color": "red", "filled": true},
            "encoding": {
              "opacity": {
                "condition": {"value": 1, "selection": "selector001"},
                "value": 0
              },
              "x": {
                "axis": {"grid": false, "title": ""},
                "field": "Time",
                "type": "temporal"
              },
              "y": {
                "axis": {"title": ""},
                "field": "Drawdown",
                "type": "quantitative"
              }
            },
            "transform": [{"filter": {"field": "Drawdown", "valid": true}}]
          }
        ],
        "data": {"name": "data-47c9ee55e0e18e98ff0830236bcebaef"},
        "height": 100,
        "title": "Drawdown",
        "width": 1000
      }
    ],
    "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json",
    "datasets": {
      "data-47c9ee55e0e18e98ff0830236bcebaef": [
        {
          "Time": "2022-01-01T00:00:00+05:00",
          "Equity": 1000,
          "Drawdown": 0,
          "Leverage Used": null
        },
        {
          "Time": "2022-01-02T13:00:00+05:00",
          "Equity": 1000,
          "Drawdown": 0,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-04T03:00:00+05:00",
          "Equity": 1008.83,
          "Drawdown": 0,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-05T17:00:00+05:00",
          "Equity": 1008.83,
          "Drawdown": 0,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-07T07:00:00+05:00",
          "Equity": 575.22,
          "Drawdown": 0.42981473,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-08T21:00:00+05:00",
          "Equity": 575.22,
          "Drawdown": 0.42981473,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-10T11:00:00+05:00",
          "Equity": 566.51,
          "Drawdown": 0.43844849,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-12T01:00:00+05:00",
          "Equity": 566.51,
          "Drawdown": 0.43844849,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-13T15:00:00+05:00",
          "Equity": 700.23,
          "Drawdown": 0.30589891,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-15T05:00:00+05:00",
          "Equity": 700.23,
          "Drawdown": 0.30589891,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-16T19:00:00+05:00",
          "Equity": 720.45,
          "Drawdown": 0.28585589,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-18T09:00:00+05:00",
          "Equity": 720.45,
          "Drawdown": 0.28585589,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-19T23:00:00+05:00",
          "Equity": 678.48,
          "Drawdown": 0.32745854,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-21T13:00:00+05:00",
          "Equity": 678.48,
          "Drawdown": 0.32745854,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-23T03:00:00+05:00",
          "Equity": 254.83,
          "Drawdown": 0.74740045,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-24T17:00:00+05:00",
          "Equity": 254.83,
          "Drawdown": 0.74740045,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-26T07:00:00+05:00",
          "Equity": 514.17,
          "Drawdown": 0.49033038,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-27T21:00:00+05:00",
          "Equity": 514.17,
          "Drawdown": 0.49033038,
          "Leverage Used": 5
        },
        {
          "Time": "2022-01-29T11:00:00+05:00",
          "Equity": 563.87,
          "Drawdown": 0.44106539,
          "Leverage Used": 0
        },
        {
          "Time": "2022-01-31T01:00:00+05:00",
          "Equity": 563.87,
          "Drawdown": 0.44106539,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-01T15:00:00+05:00",
          "Equity": 678.85,
          "Drawdown": 0.32709177,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-03T05:00:00+05:00",
          "Equity": 678.85,
          "Drawdown": 0.32709177,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-04T19:00:00+05:00",
          "Equity": 829.05,
          "Drawdown": 0.17820643,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-06T09:00:00+05:00",
          "Equity": 829.05,
          "Drawdown": 0.17820643,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-07T23:00:00+05:00",
          "Equity": 1100.2,
          "Drawdown": 0,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-09T13:00:00+05:00",
          "Equity": 1100.2,
          "Drawdown": 0,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-11T03:00:00+05:00",
          "Equity": 1095.31,
          "Drawdown": 0.00444464,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-12T17:00:00+05:00",
          "Equity": 1095.31,
          "Drawdown": 0.00444464,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-14T07:00:00+05:00",
          "Equity": 908.35,
          "Drawdown": 0.17437738,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-15T21:00:00+05:00",
          "Equity": 908.35,
          "Drawdown": 0.17437738,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-17T11:00:00+05:00",
          "Equity": 869.63,
          "Drawdown": 0.20957098,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-19T01:00:00+05:00",
          "Equity": 869.63,
          "Drawdown": 0.20957098,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-20T15:00:00+05:00",
          "Equity": 634.06,
          "Drawdown": 0.4236866,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-22T05:00:00+05:00",
          "Equity": 634.06,
          "Drawdown": 0.4236866,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-23T19:00:00+05:00",
          "Equity": 794.78,
          "Drawdown": 0.27760407,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-25T09:00:00+05:00",
          "Equity": 794.78,
          "Drawdown": 0.27760407,
          "Leverage Used": 5
        },
        {
          "Time": "2022-02-26T23:00:00+05:00",
          "Equity": 1029.61,
          "Drawdown": 0.06416106,
          "Leverage Used": 0
        },
        {
          "Time": "2022-02-28T13:00:00+05:00",
          "Equity": 1029.61,
          "Drawdown": 0.06416106,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-02T03:00:00+05:00",
          "Equity": 2204.15,
          "Drawdown": 0,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-03T17:00:00+05:00",
          "Equity": 2204.15,
          "Drawdown": 0,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-05T07:00:00+05:00",
          "Equity": 1187.91,
          "Drawdown": 0.46105755,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-06T21:00:00+05:00",
          "Equity": 1187.91,
          "Drawdown": 0.46105755,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-08T11:00:00+05:00",
          "Equity": 959.91,
          "Drawdown": 0.56449878,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-10T01:00:00+05:00",
          "Equity": 959.91,
          "Drawdown": 0.56449878,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-11T15:00:00+05:00",
          "Equity": 729.15,
          "Drawdown": 0.6691922,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-13T05:00:00+05:00",
          "Equity": 729.15,
          "Drawdown": 0.6691922,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-14T19:00:00+05:00",
          "Equity": 682.77,
          "Drawdown": 0.69023433,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-16T09:00:00+05:00",
          "Equity": 682.77,
          "Drawdown": 0.69023433,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-17T23:00:00+05:00",
          "Equity": 940.15,
          "Drawdown": 0.57346369,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-19T13:00:00+05:00",
          "Equity": 940.15,
          "Drawdown": 0.57346369,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-21T03:00:00+05:00",
          "Equity": 805.61,
          "Drawdown": 0.63450309,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-22T17:00:00+05:00",
          "Equity": 805.61,
          "Drawdown": 0.63450309,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-24T07:00:00+05:00",
          "Equity": 860.93,
          "Drawdown": 0.60940498,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-25T21:00:00+05:00",
          "Equity": 860.93,
          "Drawdown": 0.60940498,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-27T11:00:00+05:00",
          "Equity": 996.22,
          "Drawdown": 0.54802531,
          "Leverage Used": 0
        },
        {
          "Time": "2022-03-29T01:00:00+05:00",
          "Equity": 996.22,
          "Drawdown": 0.54802531,
          "Leverage Used": 5
        },
        {
          "Time": "2022-03-30T15:00:00+05:00",
          "Equity": 1104.88,
          "Drawdown": 0.4987274,
          "Leverage Used": 0
        }
      ],
      "data-ac0604903442ed28ac7caeaeddb9ad1b": [{"y": 1000}]
    }
}


const opts = {
    defaultStyle: false,
}


vegaEmbed('#vis', spec, opts)