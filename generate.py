import argparse
import json

file_options = [
    "audit_regressor",
    "cache_file",
    "initial_regressor",
    "input_feature_regularizer",
    "data",
    "output_feature_regularizer_text",
    "output_feature_regularizer_binary",
    "final_regressor",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", help="Options file in JSON format produced by vw-dump-options"
    )
    args = parser.parse_args()

    # Options may exist in more than one group, but we only want to produce a
    # single description for each.
    already_seen_options = set()

    with open(args.input, "r") as f:
        options_obj = json.load(f)

        for group in options_obj["option_groups"]:
            for option in group["options"]:
                if option["name"] in already_seen_options:
                    continue
                already_seen_options.add(option["name"])

                line = f"complete --command vw --long-option {option['name']}"
                if option["short_name"] != "":
                    line += f" --short-option {option['short_name']}"
                help_with_newlines_removed = "".join(option["help"].splitlines()).replace("'", r"\'")
                line += (
                    f" --description '{option['name']}: {help_with_newlines_removed}'"
                )
                if option["name"] in file_options:
                    line += " --force-files"
                else:
                    line += " --no-files"
                if option["type"] != "bool":
                    line += " --require-parameter"
                print(line)
