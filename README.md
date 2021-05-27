# vw-fish-completions
[Fish](https://fishshell.com/) plugins which adds completions for all of [Vowpal Wabbit](https://vowpalwabbit.org/)'s command line options. Start typing `vw --` and hit tab in Fish to see all available options.

## Installation

```sh
$ fisher install jackgerrits/vw-fish-completions
```

## Demo

[![asciicast](https://asciinema.org/a/ARQKvDlTNswv5a2OXhxuiuHfk.svg)](https://asciinema.org/a/ARQKvDlTNswv5a2OXhxuiuHfk)


## Details

To generate the rough completions the VW options framework was used. The patch of code changes to generate the completions is [here](patch.diff). I had to manually edit the generated completions to:

- Fix locations where there was a `'` in the middle of a description
- Add the different valid values for `cb_type`