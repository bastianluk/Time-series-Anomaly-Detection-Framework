# TSADF
A Generic Framework for Anomaly Detection in Time-series

## Usage

```
python main.py [-h] -t TSFILE -f TSFREQ [-m METHOD] [-s SEASONALITY]
               [-l LOWBOUNDARY] [-b HIGHBOUNDARY]

optional arguments:
  -h, --help            show this help message and exit
  -t TSFILE, --tsfile TSFILE
                        path to timeseries file ( in CSV format-> ['Time',
                        'Value'] )
  -f TSFREQ, --tsfreq TSFREQ
                        timeseries frequency
  -m METHOD, --method METHOD
                        threshold selection method: possible values -> 'automatic', 'interactive'
  -s SEASONALITY, --seasonality SEASONALITY
                        list of seasonality
  -l LOWBOUNDARY, --lowboundary LOWBOUNDARY
                        lower limit of accepted value
  -b HIGHBOUNDARY, --highboundary HIGHBOUNDARY
                        higher limit of accepted value
```

## Sample Execution
```
python main.py --tsfile sample_data.csv --tsfreq 96 --method interactive
```