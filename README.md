# vw-fish-completions
[Fish](https://fishshell.com/) plugins which adds completions for all of [Vowpal Wabbit](https://vowpalwabbit.org/)'s command line options. Start typing `vw --` and hit tab in Fish to see all available options.

## Installation

```sh
$ fisher install jackgerrits/vw-fish-completions
```

## Demo

[![asciicast](https://asciinema.org/a/ARQKvDlTNswv5a2OXhxuiuHfk.svg)](https://asciinema.org/a/ARQKvDlTNswv5a2OXhxuiuHfk)

## Generation

Use [`vw-dump-options`](https://github.com/VowpalWabbit/vowpal_wabbit/tree/master/utl/dump_options) to dump the current options state at a particular release. Then pass it to `generate.py options.json`
